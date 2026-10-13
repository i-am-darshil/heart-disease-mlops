"""Train, tune and select the heart disease classifier.

    python -m heart.train

Steps:
1. Stratified 80/20 split. The test set is used once, at the very end.
2. For each candidate, grid-search hyperparameters with stratified 5-fold CV.
3. Pick the simplest model within one standard error of the best CV ROC-AUC.
4. Lower the decision threshold until cross-validated recall reaches TARGET_RECALL.
5. Quality gate: fail (non-zero exit) if the model is too weak, so CI goes red.
6. Evaluate once on the test set, then save the model, metrics and plots.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    FixedThresholdClassifier,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from heart import evaluation
from heart.config import (
    CV_FOLDS,
    FEATURES,
    FIGURES_DIR,
    MIN_CV_ROC_AUC,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TARGET_RECALL,
    TEST_SIZE,
)
from heart.data import load_processed
from heart.models import Candidate, build_pipeline, candidates


@dataclass
class TunedModel:
    name: str
    params: dict
    cv_mean: dict[str, float]
    cv_std: dict[str, float]
    pipeline: object  # best pipeline, refit on the full training set
    oof_proba: np.ndarray  # out-of-fold probabilities on the training set
    grid_results: pd.DataFrame | None = None  # every grid point: params + CV mean/std per metric


@dataclass
class TrainingResult:
    tuned: list[TunedModel]
    selected: TunedModel
    threshold: float
    model: FixedThresholdClassifier
    test_metrics: dict[str, float]
    test_metrics_default_threshold: dict[str, float]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def tune(candidate: Candidate, X: pd.DataFrame, y: pd.Series) -> TunedModel:
    search = GridSearchCV(
        build_pipeline(candidate.estimator),
        param_grid={f"model__{k}": v for k, v in candidate.grid.items()},
        scoring=evaluation.SCORERS,
        refit="roc_auc",
        cv=cv_splitter(),
        n_jobs=-1,
    )
    search.fit(X, y)
    i = search.best_index_
    results = search.cv_results_
    columns = [c for c in results if c.startswith(("param_", "mean_test_", "std_test_"))]
    grid_results = pd.DataFrame({c: results[c] for c in columns}).rename(
        columns=lambda c: c.replace("param_model__", "")
    )
    oof = cross_val_predict(
        clone(search.best_estimator_), X, y, cv=cv_splitter(), method="predict_proba"
    )[:, 1]
    return TunedModel(
        name=candidate.name,
        params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv_mean={m: float(results[f"mean_test_{m}"][i]) for m in evaluation.METRICS},
        cv_std={m: float(results[f"std_test_{m}"][i]) for m in evaluation.METRICS},
        pipeline=search.best_estimator_,
        oof_proba=oof,
        grid_results=grid_results,
    )


def select(tuned: list[TunedModel]) -> TunedModel:
    """One-standard-error rule over candidates listed simplest first.

    With only ~240 training rows, CV scores have noticeable variance. A model that is
    "better" by less than one standard error isn't reliably better, so prefer the simpler one.
    """
    models = [t for t in tuned if t.name != "dummy"]
    best = max(models, key=lambda t: t.cv_mean["roc_auc"])
    se = best.cv_std["roc_auc"] / np.sqrt(CV_FOLDS)
    return next(t for t in models if t.cv_mean["roc_auc"] >= best.cv_mean["roc_auc"] - se)


def quality_gate(selected: TunedModel, tuned: list[TunedModel]) -> list[str]:
    failures = []
    auc = selected.cv_mean["roc_auc"]
    if auc < MIN_CV_ROC_AUC:
        failures.append(f"CV ROC-AUC {auc:.3f} is below the minimum {MIN_CV_ROC_AUC}")
    dummy = next((t for t in tuned if t.name == "dummy"), None)
    if dummy and auc <= dummy.cv_mean["roc_auc"]:
        failures.append(f"CV ROC-AUC {auc:.3f} does not beat the dummy baseline")
    return failures


def run(models: list[Candidate] | None = None, df: pd.DataFrame | None = None) -> TrainingResult:
    df = load_processed() if df is None else df
    X_train, X_test, y_train, y_test = split(df)

    tuned = [tune(c, X_train, y_train) for c in (models or candidates())]
    selected = select(tuned)
    threshold = evaluation.choose_threshold(y_train, selected.oof_proba, TARGET_RECALL)

    # The shipped model carries its own threshold: model.predict() applies it, and
    # model.predict_proba() still returns the raw probabilities
    model = FixedThresholdClassifier(
        selected.pipeline, threshold=threshold, response_method="predict_proba"
    ).fit(X_train, y_train)

    test_proba = model.predict_proba(X_test)[:, 1]
    return TrainingResult(
        tuned=tuned,
        selected=selected,
        threshold=threshold,
        model=model,
        test_metrics=evaluation.compute_metrics(y_test, test_proba, threshold),
        test_metrics_default_threshold=evaluation.compute_metrics(y_test, test_proba, 0.5),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def comparison_table(result: TrainingResult) -> pd.DataFrame:
    rows = []
    for t in result.tuned:
        row = {"model": t.name}
        for m in evaluation.METRICS:
            row[m] = f"{t.cv_mean[m]:.3f} ± {t.cv_std[m]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def save_figures(result: TrainingResult, out_dir: Path = FIGURES_DIR) -> list[Path]:
    selected = result.selected
    test_pred = result.model.predict(result.X_test)
    return [
        evaluation.plot_roc_curves(
            result.y_train,
            {t.name: t.oof_proba for t in result.tuned},
            out_dir / "model_roc_curves.png",
        ),
        evaluation.plot_threshold_tradeoff(
            result.y_train,
            selected.oof_proba,
            result.threshold,
            TARGET_RECALL,
            out_dir / "model_threshold_tradeoff.png",
        ),
        evaluation.plot_confusion_matrix(
            result.y_test,
            test_pred,
            out_dir / "model_confusion_matrix.png",
            f"Test set, threshold {result.threshold:.2f}",
        ),
        evaluation.plot_feature_effects(selected.pipeline, out_dir / "model_feature_effects.png"),
    ]


def save_outputs(result: TrainingResult, models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, models_dir / "model.joblib")
    metrics = {
        "selected_model": result.selected.name,
        "params": result.selected.params,
        "threshold": result.threshold,
        "target_recall": TARGET_RECALL,
        "test": result.test_metrics,
        "test_at_default_threshold": result.test_metrics_default_threshold,
        "cv": {
            t.name: {"params": t.params, "mean": t.cv_mean, "std": t.cv_std} for t in result.tuned
        },
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def main() -> int:
    result = run()
    pd.set_option("display.width", 120)
    print("Cross-validated scores (mean ± std over 5 folds, training set):")
    print(comparison_table(result).to_string())
    print(f"\nSelected: {result.selected.name}  {result.selected.params}")
    print(f"Decision threshold for ≥{TARGET_RECALL:.0%} recall: {result.threshold:.3f}")

    failures = quality_gate(result.selected, result.tuned)
    if failures:
        print("\nQUALITY GATE FAILED:", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    print("Quality gate passed")

    print("\nHeld-out test set:")
    print(f"  threshold {result.threshold:.2f}:", _fmt(result.test_metrics))
    print("  threshold 0.50:", _fmt(result.test_metrics_default_threshold))

    save_outputs(result)
    for path in save_figures(result):
        print(f"Wrote {path}")
    print(f"Wrote {MODELS_DIR / 'model.joblib'} and metrics.json")
    return 0


def _fmt(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())


if __name__ == "__main__":
    sys.exit(main())
