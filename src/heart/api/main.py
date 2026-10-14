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
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    Summary,
)
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
from heart.config import FEATURES, MODELS_DIR, NUMERIC_FEATURES
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
        self.inference_seconds = Histogram(
            "heart_inference_seconds",
            "Time spent in the model itself (preprocessing + predict + explanations)",
            buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry,
        )
        # Live input statistics. A Summary without quantiles is just _sum and _count, so
        # rate(_sum) / rate(_count) is the mean value seen recently, per feature.
        self.feature_value = Summary(
            "heart_input_feature",
            "Numeric feature values received (mean = rate(_sum) / rate(_count))",
            ["feature"],
            registry=registry,
        )
        self.missing_features = Counter(
            "heart_missing_features_total",
            "Features sent as null or left out (the model imputes them)",
            ["feature"],
            registry=registry,
        )
        # What the loaded model is, and its training-time baseline to compare live traffic with
        self.model_info = Gauge(
            "heart_model_info",
            "The model being served (always 1; the labels carry the information)",
            ["model_version", "estimator"],
            registry=registry,
        )
        self.model_loaded = Gauge(
            "heart_model_loaded", "1 if the model is loaded and serving", registry=registry
        )
        self.threshold = Gauge(
            "heart_model_decision_threshold",
            "Probability at or above which a patient is flagged",
            registry=registry,
        )
        self.reference_positive_rate = Gauge(
            "heart_reference_positive_rate",
            "Share of patients flagged on held-out training data (out-of-fold)",
            registry=registry,
        )
        self.reference_mean = Gauge(
            "heart_reference_feature_mean",
            "Training-set mean of each numeric feature",
            ["feature"],
            registry=registry,
        )
        self.reference_std = Gauge(
            "heart_reference_feature_std",
            "Training-set standard deviation of each numeric feature",
            ["feature"],
            registry=registry,
        )

    def describe_model(self, predictor: Predictor) -> None:
        meta = predictor.metadata
        self.model_loaded.set(1)
        self.model_info.labels(meta["model_version"], meta["estimator"]).set(1)
        self.threshold.set(predictor.threshold)
        reference = meta.get("reference")  # absent in models packaged before Phase 8
        if reference:
            self.reference_positive_rate.set(reference["positive_rate"])
            for name, stats in reference["numeric"].items():
                self.reference_mean.labels(name).set(stats["mean"])
                self.reference_std.labels(name).set(stats["std"])

    def observe_inputs(self, records: list[dict]) -> None:
        for record in records:
            for name in FEATURES:
                value = record.get(name)
                if value is None:
                    self.missing_features.labels(name).inc()
                elif name in NUMERIC_FEATURES:
                    self.feature_value.labels(name).observe(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    models_dir = app.state.models_dir
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = Predictor(models_dir)
        app.state.metrics.describe_model(app.state.predictor)
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
    records = [p.model_dump() for p in patients]
    metrics.observe_inputs(records)
    with metrics.inference_seconds.time():
        results = predictor.predict(records)
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
    # CPU and memory of this process (process_cpu_seconds_total, process_resident_memory_bytes).
    # A fresh collector reads the same /proc data the global one does.
    ProcessCollector(registry=registry)
    Instrumentator(
        excluded_handlers=["/metrics", "/health", "/ready"],
        should_group_status_codes=False,
        registry=registry,
    ).instrument(
        app,
        # Per-endpoint latency histogram; the default (0.1, 0.5, 1) is too coarse for a
        # model that answers in a few milliseconds
        latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
    ).expose(app, include_in_schema=False)
    return app


app = create_app()
