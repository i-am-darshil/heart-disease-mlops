import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from heart import tracking, train
from heart.config import CHAMPION_ALIAS, FEATURES, REGISTERED_MODEL
from heart.data import load_processed
from tests.test_train import FAST_MODELS


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A throwaway MLflow store per test module, so tests never touch ./mlflow.db."""
    root = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri()
    uri = tracking.setup(f"sqlite:///{root / 'mlflow.db'}", artifact_dir=root / "artifacts")
    yield uri
    mlflow.set_tracking_uri(previous)


@pytest.fixture(scope="module")
def result():
    return train.run(models=FAST_MODELS, df=load_processed())


@pytest.fixture(scope="module")
def logged(store, result, tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    figures = train.save_figures(result, out_dir=out)
    train.save_outputs(result, models_dir=out)
    version = tracking.log_training_run(result, figures, out / "metrics.json", gate_failures=[])
    runs = mlflow.search_runs(experiment_names=[tracking.MLFLOW_EXPERIMENT])
    return version, runs


def test_logs_parent_candidate_and_grid_point_runs(logged):
    _, runs = logged
    counts = runs["tags.stage"].value_counts()
    assert counts["training"] == 1
    assert counts["candidate"] == len(FAST_MODELS)
    assert counts["grid_point"] == 3  # dummy has 1 grid point, LR has 2


def test_parent_run_records_params_metrics_and_lineage(logged):
    _, runs = logged
    parent = runs[runs["tags.stage"] == "training"].iloc[0]
    assert parent["params.selected_model"] == "logistic_regression"
    assert 0 < float(parent["params.threshold"]) < 1
    for metric in ("cv_roc_auc", "cv_roc_auc_std", "test_recall", "test_recall_at_0.5"):
        assert 0 <= parent[f"metrics.{metric}"] <= 1
    assert parent["tags.quality_gate"] == "passed"
    assert parent["tags.raw_data_sha256"]
    assert parent["tags.git_commit"]


def test_parent_run_has_plots_and_metrics_artifacts(logged):
    _, runs = logged
    run_id = runs[runs["tags.stage"] == "training"].iloc[0].run_id
    client = MlflowClient()
    top = {a.path for a in client.list_artifacts(run_id)}
    assert {"metrics.json", "model_comparison.csv", "plots"} <= top
    assert len(client.list_artifacts(run_id, "plots")) == 4


def test_registers_model_and_moves_champion_alias(logged):
    version, _ = logged
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    assert str(mv.version) == version
    assert {"git_commit", "cv_roc_auc", "threshold"} <= set(mv.tags)


def test_champion_matches_trained_model_and_accepts_missing_values(logged, result):
    champion = tracking.load_champion()
    X = tracking.model_input(result.X_test)
    np.testing.assert_allclose(champion.predict_proba(X), result.model.predict_proba(X))

    pyfunc = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}")
    row = X.head(1).assign(ca=np.nan, thal=np.nan)
    assert pyfunc.predict(row).shape == (1,)


def test_gate_failure_is_logged_but_not_registered(store, result, tmp_path):
    before = len(MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'"))
    version = tracking.log_training_run(
        result, [], tmp_path / "missing.json", gate_failures=["too weak"]
    )
    assert version is None
    after = len(MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'"))
    assert after == before
    runs = mlflow.search_runs(
        experiment_names=[tracking.MLFLOW_EXPERIMENT],
        filter_string="tags.quality_gate = 'failed'",
    )
    assert runs.iloc[0]["tags.quality_gate_failures"] == "too weak"


def test_model_input_is_float64_for_every_feature():
    X = tracking.model_input(load_processed())
    assert list(X.columns) == FEATURES
    assert all(dtype == "float64" for dtype in X.dtypes)
