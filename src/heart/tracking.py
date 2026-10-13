"""MLflow experiment tracking for a training run.

One parent run per `make train`, holding the dataset, selection decisions, final
test metrics, plots and the registered model. Underneath it:
- one child run per candidate model (best params, CV mean/std, OOF ROC curve),
  each with one grandchild run per grid-search point, so every configuration
  that was tried is searchable and comparable in the UI.
"""

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from heart import evaluation
from heart.config import (
    CHAMPION_ALIAS,
    CV_FOLDS,
    FEATURES,
    MIN_CV_ROC_AUC,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_SHA256,
    RAW_DATA_URL,
    REGISTERED_MODEL,
    TARGET,
    TARGET_RECALL,
    TEST_SIZE,
)

SKOPS_TRUSTED_TYPES = ["heart.features._onehot_name", "numpy.dtype"]


def setup(tracking_uri: str | None = None, artifact_dir: Path = MLFLOW_ARTIFACT_DIR) -> str:
    """Point MLflow at the tracking store and create the experiment if needed."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # MLflow 3's UI opens an untagged experiment on the GenAI "Traces" view, which looks
    # empty for a classic ML project. Mark it as model development so the Runs table shows.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    return uri


def git_commit() -> str:
    """Commit SHA of the code being trained (CI sets GITHUB_SHA; locally ask git)."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_input(X: pd.DataFrame) -> pd.DataFrame:
    """Float64 view of the features: the model's input contract.

    Every column is declared as double so a missing value (NaN) is valid for any
    feature, including the coded ones. Integer columns can't hold NaN.
    """
    return X[FEATURES].astype("float64")


def _run_tags(stage: str, model_name: str | None = None) -> dict[str, str]:
    tags = {"stage": stage, "git_commit": git_commit()}
    if model_name:
        tags["model_type"] = model_name
    if run_id := os.environ.get("GITHUB_RUN_ID"):
        tags["ci_run_url"] = (
            f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
            f"/actions/runs/{run_id}"
        )
    return tags


def _log_grid_point(model_name: str, row: pd.Series) -> None:
    params = {
        k: v for k, v in row.items() if not k.startswith(("mean_test_", "std_test_", "rank_"))
    }
    with mlflow.start_run(
        run_name=f"{model_name}:{_short(params)}",
        nested=True,
        tags=_run_tags("grid_point", model_name),
    ):
        mlflow.log_params(params)
        mlflow.log_metrics(
            {f"cv_{m}": row[f"mean_test_{m}"] for m in evaluation.METRICS}
            | {f"cv_{m}_std": row[f"std_test_{m}"] for m in evaluation.METRICS}
        )


def _short(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "default"


def _log_candidate(tuned, y_train, tmp: Path) -> None:
    tags = _run_tags("candidate", tuned.name)
    with mlflow.start_run(run_name=tuned.name, nested=True, tags=tags):
        mlflow.log_params(tuned.params)
        mlflow.log_metrics(
            {f"cv_{m}": v for m, v in tuned.cv_mean.items()}
            | {f"cv_{m}_std": v for m, v in tuned.cv_std.items()}
        )
        if tuned.grid_results is not None:
            csv = tmp / f"{tuned.name}_grid.csv"
            tuned.grid_results.to_csv(csv, index=False)
            mlflow.log_artifact(str(csv), artifact_path="grid_search")
            for _, row in tuned.grid_results.iterrows():
                _log_grid_point(tuned.name, row)
        roc = evaluation.plot_roc_curves(
            y_train, {tuned.name: tuned.oof_proba}, tmp / f"{tuned.name}_roc.png"
        )
        mlflow.log_artifact(str(roc), artifact_path="plots")


def log_training_run(result, figures: list[Path], metrics_path: Path, gate_failures: list[str]):
    """Log a complete training run. Registers the model only if the quality gate passed.

    Returns the registered model version (str) or None.
    """
    data = pd.concat([model_input(result.X_train), result.y_train], axis=1)
    with mlflow.start_run(run_name=f"train:{result.selected.name}", tags=_run_tags("training")):
        # Data lineage: what was trained on and where it came from
        with warnings.catch_warnings():
            # MLflow warns that the int target can't hold NaN; the target is never missing.
            # Also hides pandas deprecation noise raised inside MLflow.
            warnings.simplefilter("ignore")
            dataset = mlflow.data.from_pandas(
                data, source=RAW_DATA_URL, name="uci-heart-cleveland", targets=TARGET
            )
        mlflow.log_input(dataset, context="training")
        mlflow.set_tags({"raw_data_sha256": RAW_DATA_SHA256, "dataset_digest": dataset.digest})

        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "target_recall": TARGET_RECALL,
                "min_cv_roc_auc": MIN_CV_ROC_AUC,
                "n_train": len(result.X_train),
                "n_test": len(result.X_test),
                "selected_model": result.selected.name,
                "threshold": round(result.threshold, 4),
            }
            | {f"model_{k}": v for k, v in result.selected.params.items()}
        )
        mlflow.log_metrics(
            {f"cv_{m}": v for m, v in result.selected.cv_mean.items()}
            | {f"cv_{m}_std": v for m, v in result.selected.cv_std.items()}
            | {f"test_{m}": v for m, v in result.test_metrics.items()}
            | {f"test_{m}_at_0.5": v for m, v in result.test_metrics_default_threshold.items()}
        )
        mlflow.set_tag("quality_gate", "failed" if gate_failures else "passed")
        if gate_failures:
            mlflow.set_tag("quality_gate_failures", "; ".join(gate_failures))

        with tempfile.TemporaryDirectory() as tmp:
            for tuned in result.tuned:
                _log_candidate(tuned, result.y_train, Path(tmp))
            comparison = Path(tmp) / "model_comparison.csv"
            train_comparison(result).to_csv(comparison)
            mlflow.log_artifact(str(comparison))

        for fig in figures:
            mlflow.log_artifact(str(fig), artifact_path="plots")
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path))

        if gate_failures:
            return None
        return _log_and_register_model(result)


def train_comparison(result) -> pd.DataFrame:
    rows = {
        t.name: {f"cv_{m}": t.cv_mean[m] for m in evaluation.METRICS}
        | {f"cv_{m}_std": t.cv_std[m] for m in evaluation.METRICS}
        | {"params": json.dumps(t.params, default=str)}
        for t in result.tuned
    }
    return pd.DataFrame(rows).T


def _log_and_register_model(result) -> str:
    X_example = model_input(result.X_test.head(3))
    X_train = model_input(result.X_train)
    with warnings.catch_warnings():
        # Same int-can't-hold-NaN hint, here for the 0/1 prediction output, which is never NaN
        warnings.simplefilter("ignore", UserWarning)
        signature = infer_signature(X_train, result.model.predict(X_train))
    info = mlflow.sklearn.log_model(
        result.model,
        name="model",
        # skops (MLflow's default sklearn format) only loads types it is told to trust,
        # unlike pickle which runs arbitrary code. Our pipeline adds exactly these two.
        serialization_format="skops",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
        signature=signature,
        input_example=X_example,
        registered_model_name=REGISTERED_MODEL,
        metadata={"threshold": result.threshold, "selected_model": result.selected.name},
    )
    version = str(info.registered_model_version)
    client = MlflowClient()
    client.update_model_version(
        REGISTERED_MODEL,
        version,
        description=(
            f"{result.selected.name} {result.selected.params}, threshold {result.threshold:.3f}. "
            f"CV ROC-AUC {result.selected.cv_mean['roc_auc']:.3f}, "
            f"test recall {result.test_metrics['recall']:.3f}."
        ),
    )
    for key, value in {
        "git_commit": git_commit(),
        "cv_roc_auc": f"{result.selected.cv_mean['roc_auc']:.4f}",
        "threshold": f"{result.threshold:.4f}",
    }.items():
        client.set_model_version_tag(REGISTERED_MODEL, version, key, value)
    # Promoting is a single alias move; serving loads models:/<name>@champion
    client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)
    return version


def load_champion():
    """Load the model currently aliased as champion (used by tests and, later, serving)."""
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}")


__all__ = ["setup", "log_training_run", "load_champion", "model_input", "PROCESSED_DATA_PATH"]
