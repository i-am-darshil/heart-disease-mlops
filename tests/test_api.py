import json

import pytest
from fastapi.testclient import TestClient

from heart import train
from heart.api import main
from heart.api.schemas import EXAMPLE_PATIENT
from heart.data import load_processed
from tests.test_train import FAST_MODELS


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train.save_outputs(train.run(models=FAST_MODELS, df=load_processed()), models_dir=out)
    return out


@pytest.fixture(scope="module")
def client(models_dir):
    # A fresh app per fixture, so no test can leak app state into another.
    # `with` runs the startup hook, which loads the model.
    with TestClient(main.create_app(models_dir)) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path):
    with TestClient(main.create_app(tmp_path)) as c:
        yield c


def test_predict_returns_prediction_probability_confidence_and_factors(client):
    r = client.post("/predict", json=EXAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart disease", "no heart disease")
    assert 0 <= body["probability"] <= 1 and 0.5 <= body["confidence"] <= 1
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert 1 <= len(body["top_factors"]) <= 3
    assert {"feature", "description", "effect"} <= set(body["top_factors"][0])
    assert body["model_version"]


def test_factors_are_sorted_by_strength(client):
    factors = client.post("/predict", json=EXAMPLE_PATIENT).json()["top_factors"]
    effects = [abs(f["effect"]) for f in factors]
    assert effects == sorted(effects, reverse=True)


def test_predict_matches_the_offline_predictor(client, models_dir):
    from heart.predict import Predictor

    expected = Predictor(models_dir).predict_one(EXAMPLE_PATIENT)
    body = client.post("/predict", json=EXAMPLE_PATIENT).json()
    assert body["probability"] == expected.probability
    assert body["prediction"] == expected.prediction


def test_missing_and_null_fields_are_imputed(client):
    partial = {k: v for k, v in EXAMPLE_PATIENT.items() if k not in ("chol", "fbs")}
    assert client.post("/predict", json=partial | {"ca": None, "thal": None}).status_code == 200
    assert client.post("/predict", json={}).status_code == 200


@pytest.mark.parametrize(
    "bad, field",
    [
        ({"age": 500}, "age"),
        ({"age": -1}, "age"),
        ({"cp": 9}, "cp"),
        ({"ca": 1.5}, "ca"),
        ({"sex": "male"}, "sex"),
        ({"cholesterol": 200}, "cholesterol"),
    ],
)
def test_invalid_input_is_rejected_with_field_errors(client, bad, field):
    r = client.post("/predict", json=EXAMPLE_PATIENT | bad)
    assert r.status_code == 422
    assert field in [e["field"] for e in r.json()["errors"]]


def test_non_json_body_is_rejected(client):
    r = client.post("/predict", content="not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_predicts_every_patient(client):
    patients = json.loads(open("examples/sample_batch.json").read())
    r = client.post("/predict/batch", json={"patients": patients})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == len(patients)


def test_batch_rejects_empty_and_oversized_requests(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
    too_many = {"patients": [EXAMPLE_PATIENT] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["model_loaded"] is True


def test_model_endpoint_describes_the_served_model(client):
    body = client.get("/model").json()
    assert body["estimator"] == "logistic_regression"
    assert 0 < body["decision_threshold"] < 1
    assert len(body["features"]) == 13


def test_without_a_model_health_is_ok_but_ready_and_predict_are_503(client_without_model):
    c = client_without_model
    assert c.get("/health").status_code == 200
    ready = c.get("/ready")
    assert ready.status_code == 503 and "Train a model first" in ready.json()["detail"]
    assert c.post("/predict", json=EXAMPLE_PATIENT).status_code == 503


def test_request_id_is_echoed_or_generated(client):
    assert client.get("/model", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"
    assert client.get("/model").headers["x-request-id"]


def test_metrics_expose_http_and_model_counters(client):
    client.post("/predict", json=EXAMPLE_PATIENT)
    client.post("/predict", json={"age": 999})
    text = client.get("/metrics").text
    assert 'http_requests_total{handler="/predict",method="POST",status="200"}' in text
    assert "heart_predictions_total" in text
    assert "heart_prediction_probability_bucket" in text
    assert 'heart_validation_errors_total{endpoint="/predict"}' in text


def test_metrics_describe_the_served_model_and_its_training_baseline(client, models_dir):
    meta = json.loads((models_dir / "model_metadata.json").read_text())
    text = client.get("/metrics").text
    info = f'estimator="{meta["estimator"]}",model_version="{meta["model_version"]}"'
    assert f"heart_model_info{{{info}}} 1.0" in text
    assert "heart_model_loaded 1.0" in text
    assert f"heart_reference_positive_rate {meta['reference']['positive_rate']}" in text
    assert 'heart_reference_feature_mean{feature="chol"}' in text


def test_metrics_track_input_values_and_missing_features(client):
    def value(text, sample):
        line = next(x for x in text.splitlines() if x.startswith(sample + " "))
        return float(line.split()[-1])

    before = client.get("/metrics").text
    client.post("/predict", json=EXAMPLE_PATIENT | {"chol": 400, "thal": None})
    after = client.get("/metrics").text
    chol_sum = 'heart_input_feature_sum{feature="chol"}'
    missing = 'heart_missing_features_total{feature="thal"}'
    assert value(after, chol_sum) - value(before, chol_sum) == 400
    assert value(after, missing) - (value(before, missing) if missing in before else 0) == 1
    assert "heart_inference_seconds_count" in after


def test_model_loaded_is_zero_without_a_model(client_without_model):
    assert "heart_model_loaded 0.0" in client_without_model.get("/metrics").text


def test_requests_are_logged_as_json_without_patient_data(client, caplog):
    from heart.api.logging import JsonFormatter

    with caplog.at_level("INFO", logger="heart.api"):
        client.post("/predict", json=EXAMPLE_PATIENT, headers={"x-request-id": "log-test"})
    records = [r for r in caplog.records if getattr(r, "request_id", None) == "log-test"]
    assert records, "no request log line"
    entry = json.loads(JsonFormatter().format(records[0]))  # exactly what goes to stdout
    assert entry["msg"] == "request" and entry["level"] == "INFO"
    assert entry["path"] == "/predict" and entry["status"] == 200
    assert "duration_ms" in entry and "prediction" in entry and "ts" in entry
    assert not {"age", "chol", "trestbps"} & set(entry), "patient measurements must not be logged"


def test_openapi_docs_document_every_feature(client):
    schema = client.get("/openapi.json").json()
    patient = schema["components"]["schemas"]["Patient"]["properties"]
    assert len(patient) == 13
    assert "Chest pain type" in patient["cp"]["description"]


def test_json_formatter_keeps_extra_fields_and_drops_uvicorn_colour_copy():
    import logging

    from heart.api.logging import JsonFormatter

    record = logging.makeLogRecord(
        {"name": "x", "levelname": "INFO", "msg": "hi", "path": "/p", "color_message": "\x1b[1mhi"}
    )
    entry = json.loads(JsonFormatter().format(record))
    assert entry["msg"] == "hi" and entry["path"] == "/p"
    assert "color_message" not in entry
