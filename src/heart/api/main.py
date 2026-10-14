"""Heart disease risk API.

    uvicorn heart.api.main:app --host 0.0.0.0 --port 8000

Endpoints
    POST /predict         one patient  -> prediction, probability, confidence, top factors
    POST /predict/batch   up to 1000 patients
    GET  /health          liveness: the process is up
    GET  /ready           readiness: the model is loaded and can serve
    GET  /model           metadata of the model being served
    GET  /metrics         Prometheus metrics
    GET  /docs            interactive API docs (Swagger UI)

The model is loaded once at startup from MODELS_DIR (default: models/ in the project,
/app/models in the container). Every request is logged as one JSON line.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from heart import __version__
from heart.api.logging import configure_logging
from heart.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    Patient,
    PredictionResponse,
    ReadyResponse,
)
from heart.config import MODELS_DIR
from heart.packaging import ModelPackageError
from heart.predict import Predictor

log = logging.getLogger("heart.api")


class ModelMetrics:
    """Model-level metrics, on top of the HTTP metrics the instrumentator adds.

    Created per app, in that app's registry, so building several apps (as tests do)
    never registers the same metric twice.
    """

    def __init__(self, registry: CollectorRegistry):
        self.predictions = Counter(
            "heart_predictions_total",
            "Predictions served, by predicted class",
            ["label"],
            registry=registry,
        )
        self.probability = Histogram(
            "heart_prediction_probability",
            "Predicted probability of heart disease (watch for drift)",
            buckets=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry,
        )
        self.validation_errors = Counter(
            "heart_validation_errors_total",
            "Requests rejected by input validation",
            ["endpoint"],
            registry=registry,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    models_dir = app.state.models_dir
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = Predictor(models_dir)
        log.info(
            "model loaded",
            extra={"model_version": app.state.predictor.model_version, "dir": str(models_dir)},
        )
    except ModelPackageError as err:
        # Stay up so /health answers and /ready explains the problem
        app.state.load_error = str(err)
        log.error("model not loaded", extra={"error": str(err)})
    yield


router = APIRouter()


def get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, f"Model not loaded: {request.app.state.load_error}")
    return predictor


async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["x-request-id"] = request_id
    if request.url.path not in ("/health", "/ready", "/metrics"):
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else None,
                **getattr(request.state, "log_extra", {}),
            },
        )
    return response


async def validation_error(request: Request, exc: RequestValidationError):
    request.app.state.metrics.validation_errors.labels(endpoint=request.url.path).inc()
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()
    ]
    request.state.log_extra = {"validation_errors": len(errors)}
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


def _respond(request: Request, patients: list[Patient]) -> list[PredictionResponse]:
    predictor = get_predictor(request)
    metrics: ModelMetrics = request.app.state.metrics
    results = predictor.predict([p.model_dump() for p in patients])
    for r in results:
        metrics.predictions.labels(label=r.label).inc()
        metrics.probability.observe(r.probability)
    return [
        PredictionResponse(**vars(r) | {"top_factors": [vars(f) for f in r.top_factors]},
                           model_version=predictor.model_version)
        for r in results
    ]  # fmt: skip


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: Patient, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
    """Predict heart disease risk for one patient."""
    result = _respond(request, [patient])[0]
    # Log the outcome, never the patient's measurements
    request.state.log_extra = {
        "prediction": result.prediction,
        "probability": result.probability,
        "model_version": result.model_version,
    }
    return result


@router.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    """Predict for up to 1000 patients in one call."""
    predictor = get_predictor(request)
    results = _respond(request, batch.patients)
    request.state.log_extra = {
        "batch_size": len(results),
        "positives": sum(r.prediction for r in results),
        "model_version": predictor.model_version,
    }
    return BatchResponse(predictions=results, model_version=predictor.model_version)


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: the process is running (Kubernetes restarts the pod if this fails)."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(request: Request) -> JSONResponse:
    """Readiness probe: 200 only when the model is loaded (traffic is withheld otherwise)."""
    predictor = request.app.state.predictor
    if predictor is None:
        body = ReadyResponse(
            status="not ready", model_loaded=False, detail=request.app.state.load_error
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    body = ReadyResponse(status="ready", model_loaded=True, model_version=predictor.model_version)
    return JSONResponse(content=body.model_dump())


@router.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info(request: Request) -> ModelInfoResponse:
    """Metadata of the model currently being served."""
    meta = get_predictor(request).metadata
    return ModelInfoResponse(**{k: meta[k] for k in ModelInfoResponse.model_fields})


@router.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "heart-disease-api", "docs": "/docs", "health": "/health"}


def create_app(models_dir: Path | str | None = None) -> FastAPI:
    """Build the app. Each call returns an independent app (tests rely on this).

    The model folder comes from the argument, else $MODELS_DIR, else the project's models/.
    """
    app = FastAPI(
        title="Heart Disease Risk API",
        description=(
            "Predicts the presence of heart disease from 13 clinical measurements "
            "(UCI Cleveland dataset). The decision threshold is tuned for at least 90% "
            "recall: in screening, a missed case costs more than a false alarm. "
            "**Not for clinical use.**"
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.models_dir = Path(models_dir or os.environ.get("MODELS_DIR", MODELS_DIR))
    app.include_router(router)
    app.middleware("http")(log_requests)
    app.add_exception_handler(RequestValidationError, validation_error)
    # One registry per app holds both HTTP metrics (request counts, latency by endpoint)
    # and the model metrics; /metrics serves exactly this registry.
    registry = CollectorRegistry()
    app.state.metrics = ModelMetrics(registry)
    Instrumentator(
        excluded_handlers=["/metrics", "/health", "/ready"],
        should_group_status_codes=False,
        registry=registry,
    ).instrument(app).expose(app, include_in_schema=False)
    return app


app = create_app()
