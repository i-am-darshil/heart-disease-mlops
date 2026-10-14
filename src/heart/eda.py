"""EDA figures, shared by the notebook and `make eda` so the report always uses fresh plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed (works in CI and Docker)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from heart.config import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_LABELS,
    FEATURE_NAMES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET,
)
from heart.data import load_processed  # noqa: E402

# Palette: categorical slots 1-2 for the two classes, gray-midpoint diverging map for correlations
NO_DISEASE = "#2a78d6"
DISEASE = "#eb6834"
CLASS_COLORS = {0: NO_DISEASE, 1: DISEASE}
CLASS_NAMES = {0: "No disease", 1: "Disease"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#256abf", "#f0efec", "#e34948"])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": BASELINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = [CLASS_NAMES[c] for c in counts.index]
    bars = ax.bar(labels, counts.values, color=[CLASS_COLORS[c] for c in counts.index], width=0.55)
    for bar, n in zip(bars, counts.values, strict=True):
        ax.annotate(
            f"{n}  ({n / counts.sum():.0%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=INK,
        )
    ax.set_title("Class balance: patients with and without heart disease", loc="left")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    return _save(fig, "class_balance", out_dir)


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        bins = np.histogram_bin_edges(df[col].dropna(), bins=20)
        for cls in (0, 1):
            values = df.loc[df[TARGET] == cls, col].dropna()
            ax.hist(values, bins=bins, color=CLASS_COLORS[cls], alpha=0.25)
            ax.hist(values, bins=bins, color=CLASS_COLORS[cls], histtype="step", linewidth=2)
            ax.axvline(values.median(), color=CLASS_COLORS[cls], linestyle="--", linewidth=1.2)
        ax.set_title(FEATURE_NAMES[col], loc="left")
        ax.set_ylabel("Patients")

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS[c], alpha=0.6, edgecolor=CLASS_COLORS[c])
        for c in (0, 1)
    ]
    legend_ax.legend(
        handles, [CLASS_NAMES[c] for c in (0, 1)], loc="center left", fontsize=11, title="Class"
    )
    legend_ax.text(0.02, 0.22, "Dashed line = class median", color=INK_SECONDARY, fontsize=9)
    fig.suptitle(
        "Numeric features by class", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK
    )
    fig.tight_layout()
    return _save(fig, "numeric_distributions", out_dir)


def plot_categorical_disease_rate(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    """Share of patients with disease in each category level, vs the overall rate."""
    cols = CATEGORICAL_FEATURES + BINARY_FEATURES
    overall = df[TARGET].mean()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.flat, cols, strict=True):
        grouped = df.groupby(col, dropna=True)[TARGET].agg(["mean", "size"])
        names = CATEGORY_LABELS.get(col, {})
        labels = [names.get(int(v), str(int(v))) for v in grouped.index]
        y = np.arange(len(grouped))
        ax.barh(y, grouped["mean"], color=DISEASE, height=0.6)
        for yi, (rate, n) in enumerate(zip(grouped["mean"], grouped["size"], strict=True)):
            ax.text(
                rate + 0.02,
                yi,
                f"{rate:.0%}  (n={n})",
                va="center",
                fontsize=8.5,
                color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
                zorder=3,
            )
        ax.axvline(overall, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.set_title(FEATURE_NAMES[col], loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        f"Share of patients with heart disease by category  (dashed line = overall {overall:.0%})",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, "categorical_disease_rate", out_dir)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> Path:
    corr = df.astype(float).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # lower triangle only: no duplicates
    shown = corr.mask(mask)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1)
    n = len(corr)
    for i in range(n):
        for j in range(i + 1):
            value = corr.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.6 else INK,
            )
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.columns)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson correlation", color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title("Feature correlations (including target)", loc="left")
    return _save(fig, "correlation_heatmap", out_dir)


def make_all_figures(df: pd.DataFrame | None = None, out_dir: Path = FIGURES_DIR) -> list[Path]:
    _style()
    df = load_processed() if df is None else df
    return [
        plot_class_balance(df, out_dir),
        plot_numeric_distributions(df, out_dir),
        plot_categorical_disease_rate(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
    ]


if __name__ == "__main__":
    for path in make_all_figures():
        print(f"Wrote {path}")
