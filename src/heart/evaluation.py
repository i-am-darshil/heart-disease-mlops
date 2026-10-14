"""Metrics and evaluation plots shared by training, the notebook and (Phase 3) MLflow logging."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heart import eda
from heart.config import readable_feature_name

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# GridSearchCV scorers. Precision is spelled out so a heavily regularized model that predicts
# no positives scores 0 instead of raising a warning on every such fold.
SCORERS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Categorical slots 1-3 for the tuned models; the baseline is a neutral reference line
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "gradient_boosting": "#1baf7a",
}
MODEL_LABELS = {
    "dummy": "Dummy baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def choose_threshold(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall still reaches the target (maximizes precision under it)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    # recall[i] is the recall when predicting positive for proba >= thresholds[i]
    meets = recall[:-1] >= target_recall
    return float(thresholds[meets].max())


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(y_true, oof_proba: dict[str, np.ndarray], path: Path) -> Path:
    """Cross-validated (out-of-fold) ROC curve for every candidate."""
    eda._style()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=eda.BASELINE, linewidth=1, linestyle=":")
    for name, proba in oof_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        if name == "dummy":
            style = {"color": eda.INK_MUTED, "linestyle": "--", "linewidth": 1.5}
        else:
            style = {"color": MODEL_COLORS[name], "linewidth": 2}
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]}  (AUC {auc:.3f})", **style)
    ax.set_xlabel("False positive rate (healthy patients flagged)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("ROC curves, 5-fold cross-validation on the training set", loc="left")
    return _save(fig, path)


def plot_threshold_tradeoff(
    y_true, proba, threshold: float, target_recall: float, path: Path
) -> Path:
    eda._style()
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, recall[:-1], color=eda.DISEASE, linewidth=2, label="Recall")
    ax.plot(thresholds, precision[:-1], color=eda.NO_DISEASE, linewidth=2, label="Precision")
    ax.axhline(target_recall, color=eda.INK_MUTED, linestyle=":", linewidth=1)
    ax.axvline(threshold, color=eda.INK, linestyle="--", linewidth=1)
    ax.axvline(0.5, color=eda.BASELINE, linestyle="--", linewidth=1)
    ax.annotate(
        f"chosen threshold {threshold:.2f}",
        (threshold, 0.05),
        xytext=(5, 0),
        textcoords="offset points",
        color=eda.INK,
        fontsize=9,
    )
    ax.annotate(
        "default 0.5",
        (0.5, 0.15),
        xytext=(5, 0),
        textcoords="offset points",
        color=eda.INK_SECONDARY,
        fontsize=9,
    )
    ax.annotate(
        f"target recall {target_recall:.0%}",
        (0.0, target_recall),
        xytext=(4, -12),
        textcoords="offset points",
        ha="left",
        color=eda.INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Decision threshold (predict disease if probability ≥ threshold)")
    ax.set_ylabel("Score (cross-validated)")
    ax.legend(loc="lower left")
    ax.set_title("Precision vs recall as the decision threshold moves", loc="left")
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> Path:
    eda._style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    names = ["No disease", "Disease"]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else eda.INK,
            )
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left")
    return _save(fig, path)


def feature_effects(pipeline) -> pd.Series:
    """Signed coefficients for linear models, impurity importances for tree models."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
    else:
        values = model.feature_importances_
    return pd.Series(values, index=names).sort_values(key=np.abs, ascending=False)


def plot_feature_effects(pipeline, path: Path, top_n: int = 15) -> Path:
    eda._style()
    effects = feature_effects(pipeline)
    effects = effects[effects.abs() > 0.01].head(top_n)[::-1]  # L1 zeroes out unused features
    effects.index = [readable_feature_name(n) for n in effects.index]
    signed = (effects < 0).any()
    if signed:
        colors = [eda.NO_DISEASE if v < 0 else eda.DISEASE for v in effects]
        xlabel = "Coefficient on standardized / one-hot features (+ raises risk)"
    else:
        colors = [eda.NO_DISEASE] * len(effects)
        xlabel = "Impurity-based importance"
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(effects.index, effects.values, color=colors, height=0.65)
    ax.axvline(0, color=eda.BASELINE, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Top {len(effects)} features, final model", loc="left")
    return _save(fig, path)
