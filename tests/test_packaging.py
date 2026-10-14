import json

import joblib
import numpy as np
import pytest

from heart import packaging, train
from heart.config import FEATURES
from heart.data import load_processed
from heart.predict import Predictor, main
from tests.test_train import FAST_MODELS

SAMPLE = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}  # fmt: skip


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train.save_outputs(train.run(models=FAST_MODELS, df=load_processed()), models_dir=out)
    return out


def test_package_contains_model_metadata_and_metrics(models_dir):
    names = {p.name for p in models_dir.iterdir()}
    assert {"model.joblib", "model_metadata.json", "metrics.json"} <= names


def test_metadata_describes_model_data_and_environment(models_dir):
    meta = json.loads((models_dir / "model_metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert [f["name"] for f in meta["input_schema"]] == FEATURES
    assert 0 < meta["decision_threshold"] < 1
    assert meta["model_sha256"] == packaging.sha256_file(models_dir / "model.joblib")
    assert meta["training_data"]["n_train"] + meta["training_data"]["n_test"] == 303
    assert set(packaging.RUNTIME_LIBRARIES) <= set(meta["environment"])
    assert packaging.python_version_matches(meta)


def test_input_schema_lists_ranges_and_allowed_codes():
    schema = {f["name"]: f for f in packaging.input_schema()}
    assert schema["age"]["min"] == 1 and schema["age"]["max"] == 120
    assert schema["cp"]["allowed"] == [1, 2, 3, 4]
    assert schema["sex"]["kind"] == "binary"
    assert all(f["nullable"] for f in schema.values())


def test_load_model_rejects_a_tampered_model_file(models_dir, tmp_path):
    for name in ("model.joblib", "model_metadata.json"):
        (tmp_path / name).write_bytes((models_dir / name).read_bytes())
    joblib.dump({"not": "the model"}, tmp_path / "model.joblib")
    with pytest.raises(packaging.ModelPackageError, match="checksum"):
        packaging.load_model(tmp_path)


def test_load_model_rejects_mismatched_library_versions(models_dir, tmp_path):
    (tmp_path / "model.joblib").write_bytes((models_dir / "model.joblib").read_bytes())
    meta = json.loads((models_dir / "model_metadata.json").read_text())
    meta["environment"]["scikit-learn"] = "0.0.1"
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(packaging.ModelPackageError, match="scikit-learn trained=0.0.1"):
        packaging.load_model(tmp_path)


def test_load_model_explains_missing_files(tmp_path):
    with pytest.raises(packaging.ModelPackageError, match="Train a model first"):
        packaging.load_model(tmp_path)


def test_predictor_returns_label_probability_and_confidence(models_dir):
    result = Predictor(models_dir).predict_one(SAMPLE)
    assert result.prediction in (0, 1)
    assert result.label in ("heart disease", "no heart disease")
    assert 0 <= result.probability <= 1
    assert result.confidence == pytest.approx(
        result.probability if result.prediction == 1 else 1 - result.probability, abs=1e-4
    )
    assert result.prediction == int(result.probability >= result.threshold)


def test_predictor_matches_the_trained_pipeline_exactly(models_dir):
    predictor = Predictor(models_dir)
    X = load_processed()[FEATURES].head(20)
    records = X.astype(float).to_dict(orient="records")
    expected = predictor.model.predict_proba(X.astype("float64"))[:, 1]
    got = [r.probability for r in predictor.predict(records)]
    np.testing.assert_allclose(got, expected.round(4))


def test_predictor_accepts_nulls_and_missing_features(models_dir):
    predictor = Predictor(models_dir)
    with_nulls = SAMPLE | {"ca": None, "thal": None}
    missing = {k: v for k, v in SAMPLE.items() if k not in ("chol", "fbs")}
    assert predictor.predict_one(with_nulls).prediction in (0, 1)
    assert predictor.predict_one(missing).prediction in (0, 1)


def test_predictor_rejects_unknown_features(models_dir):
    with pytest.raises(ValueError, match="Unknown feature"):
        Predictor(models_dir).predict_one(SAMPLE | {"cholesterol": 200})


def test_cli_predicts_single_record_and_batch(models_dir, tmp_path, capsys):
    single, batch = tmp_path / "one.json", tmp_path / "many.json"
    single.write_text(json.dumps(SAMPLE))
    batch.write_text(json.dumps([SAMPLE, SAMPLE | {"age": 40}]))
    assert main([str(single), "--models-dir", str(models_dir)]) == 0
    assert "prediction" in json.loads(capsys.readouterr().out)
    assert main([str(batch), "--models-dir", str(models_dir)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_cli_exits_nonzero_without_a_model(tmp_path, capsys):
    request = tmp_path / "one.json"
    request.write_text(json.dumps(SAMPLE))
    assert main([str(request), "--models-dir", str(tmp_path)]) == 1
    assert "Train a model first" in capsys.readouterr().err


def test_example_request_files_are_valid_input(models_dir):
    predictor = Predictor(models_dir)
    for path in ("examples/sample_request.json", "examples/sample_batch.json"):
        data = json.loads(open(path).read())
        records = data if isinstance(data, list) else [data]
        assert len(predictor.predict(records)) == len(records)


def test_reproduce_compare_flags_any_difference():
    from heart.reproduce import compare

    meta = {"params": {"C": 1.0}, "estimator": "lr", "decision_threshold": 0.25}
    run = {"sha256": "a" * 64, "proba": np.array([0.1, 0.9]), "metrics": {"x": 1}, "metadata": meta}
    assert all(ok for _, ok, _ in compare(run, dict(run)))

    drifted = run | {"proba": np.array([0.1, 0.9 + 1e-12]), "sha256": "b" * 64}
    failed = {name for name, ok, _ in compare(run, drifted) if not ok}
    assert failed == {"model file (SHA-256)", "predictions, all 303 rows"}
