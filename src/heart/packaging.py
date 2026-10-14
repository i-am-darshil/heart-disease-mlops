"""Model packaging: the metadata saved next to the model, and loading it back.

`models/` holds everything needed to use the model without the training code's state:

    model.joblib          the fitted pipeline (preprocessing + classifier + threshold)
    model_metadata.json   what it is, how it was made, and its input contract
    metrics.json          full CV and test results
"""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from heart import __version__
from heart.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    NUMERIC_RANGES,
    PROCESSED_DATA_PATH,
    RAW_DATA_SHA256,
    TARGET_RECALL,
)

MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"

# Libraries whose version must match at load time for the pickled pipeline to behave the same
RUNTIME_LIBRARIES = ["scikit-learn", "numpy", "pandas", "joblib"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_schema() -> list[dict]:
    """The model's input contract: every feature, its type, allowed values, and nullability."""
    fields = []
    for name in FEATURES:
        field = {"name": name, "type": "number", "nullable": True}
        if name in NUMERIC_FEATURES:
            field["kind"] = "numeric"
            field["min"], field["max"] = NUMERIC_RANGES[name]
        else:
            field["kind"] = "binary" if name in BINARY_FEATURES else "categorical"
            field["allowed"] = sorted(CATEGORICAL_VALUES[name])
        fields.append(field)
    return fields


def training_reference(result) -> dict:
    """The training-time baseline the API compares live traffic against (drift monitoring).

    positive_rate: share of patients flagged at the tuned threshold, measured on
    out-of-fold predictions, so it is what to expect on patients the model hasn't seen.
    numeric: mean and standard deviation of each numeric feature in the training set.
    """
    X = result.X_train[NUMERIC_FEATURES]
    return {
        "positive_rate": round(float(np.mean(result.selected.oof_proba >= result.threshold)), 4),
        "numeric": {
            name: {"mean": round(float(X[name].mean()), 4), "std": round(float(X[name].std()), 4)}
            for name in NUMERIC_FEATURES
        },
    }


def build_metadata(result, model_path: Path) -> dict:
    from heart.tracking import git_commit  # imported lazily: pulls in MLflow

    return {
        "model_name": "heart-disease-classifier",
        "model_version": git_commit()[:12],
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model_sha256": sha256_file(model_path),
        "estimator": result.selected.name,
        "params": result.selected.params,
        "decision_threshold": result.threshold,
        "target_recall": TARGET_RECALL,
        "classes": {"0": "no heart disease", "1": "heart disease"},
        "metrics": {
            "cv_roc_auc": result.selected.cv_mean["roc_auc"],
            "cv_roc_auc_std": result.selected.cv_std["roc_auc"],
            "test": result.test_metrics,
        },
        "training_data": {
            "source_sha256": RAW_DATA_SHA256,
            "processed_sha256": sha256_file(PROCESSED_DATA_PATH),
            "n_train": len(result.X_train),
            "n_test": len(result.X_test),
        },
        "features": FEATURES,
        "feature_groups": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "binary": BINARY_FEATURES,
        },
        "input_schema": input_schema(),
        "reference": training_reference(result),
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            **{lib: version(lib) for lib in RUNTIME_LIBRARIES},
        },
    }


class ModelPackageError(RuntimeError):
    pass


def load_model(models_dir: Path = MODELS_DIR, verify: bool = True):
    """Load the pipeline and its metadata, checking the file and library versions match.

    Returns (model, metadata). Raises ModelPackageError if the files are missing, the
    model file was modified or swapped, or the installed library versions differ from
    the ones the model was trained with.
    """
    models_dir = Path(models_dir)
    model_path, metadata_path = models_dir / MODEL_FILE, models_dir / METADATA_FILE
    for path in (model_path, metadata_path):
        if not path.exists():
            raise ModelPackageError(f"{path} not found. Train a model first (./run.sh train).")
    metadata = json.loads(metadata_path.read_text())
    if verify:
        if sha256_file(model_path) != metadata["model_sha256"]:
            raise ModelPackageError(
                f"{model_path} does not match the checksum in {metadata_path}; "
                "it was modified or comes from a different training run."
            )
        mismatched = {
            lib: (metadata["environment"][lib], version(lib))
            for lib in RUNTIME_LIBRARIES
            if metadata["environment"].get(lib) != version(lib)
        }
        if mismatched:
            details = ", ".join(
                f"{lib} trained={trained} installed={installed}"
                for lib, (trained, installed) in mismatched.items()
            )
            raise ModelPackageError(f"Library versions differ from training: {details}")
    return joblib.load(model_path), metadata


def python_version_matches(metadata: dict) -> bool:
    trained = tuple(metadata["environment"]["python"].split(".")[:2])
    return trained == tuple(str(v) for v in sys.version_info[:2])
