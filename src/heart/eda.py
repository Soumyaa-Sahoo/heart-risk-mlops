"""Exploratory data analysis figures.

Usage:
    python -m heart.eda          # writes PNGs to reports/figures/eda/

The notebook notebooks/01_eda.ipynb calls these same functions, so the
figures in the report and the notebook are always identical.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402
from heart.features import band  # noqa: E402

PALETTE = {0: "#4C72B0", 1: "#DD8452"}
TARGET_LABELS = {0: "No disease", 1: "Disease"}

CATEGORY_NAMES = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "exang": {0: "no", 1: "yes"},
}

sns.set_theme(style="whitegrid", context="notebook")


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diagnosis"] = out[config.TARGET].map(TARGET_LABELS)
    return out


def plot_class_balance(df: pd.DataFrame):
    counts = df[config.TARGET].map(TARGET_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v} ({v / counts.sum():.0%})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    return fig


def plot_missing(df: pd.DataFrame):
    miss = df[config.INPUT_FEATURES].isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(miss.index, miss.values, color="#8172B2")
    ax.set_ylabel("% missing")
    ax.set_title("Missing values per feature (before imputation)")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_numeric_distributions(df: pd.DataFrame):
    d = _labelled(df)
    cols = [c for c in config.NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(data=d, x=col, hue="diagnosis", kde=True, element="step",
                     stat="density", common_norm=False, ax=ax,
                     palette=[PALETTE[0], PALETTE[1]], hue_order=list(TARGET_LABELS.values()))
        ax.set_title(col)
    # Last panel: the engineered feature, to motivate it visually.
    d["hr_reserve_ratio"] = d["thalach"] / (220 - d["age"])
    sns.histplot(data=d, x="hr_reserve_ratio", hue="diagnosis", kde=True, element="step",
                 stat="density", common_norm=False, ax=axes.flat[-1],
                 palette=[PALETTE[0], PALETTE[1]], hue_order=list(TARGET_LABELS.values()))
    axes.flat[-1].axvline(0.85, ls="--", color="grey", lw=1)
    axes.flat[-1].set_title("hr_reserve_ratio (engineered)")
    fig.suptitle("Numeric feature distributions by diagnosis", y=1.01)
    fig.tight_layout()
    return fig


def plot_categorical_rates(df: pd.DataFrame):
    """Disease rate within each category level, more informative than raw counts."""
    cols = config.CATEGORICAL_FEATURES + config.BINARY_FEATURES + ["ca"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, cols):
        g = df.groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index()
        names = CATEGORY_NAMES.get(col, {})
        g["level"] = [names.get(int(v), str(int(v))) for v in g[col]]
        ax.bar(g["level"], g["mean"], color="#C44E52")
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.annotate(f"n={n}", (i, m), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(col)
        ax.set_ylabel("Disease rate")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Heart-disease rate by category level", y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame):
    cols = config.INPUT_FEATURES + [config.TARGET]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Spearman correlation (handles ordinal codes better than Pearson)")
    return fig


def plot_outliers(df: pd.DataFrame):
    d = _labelled(df)
    cols = ["trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=d, x="diagnosis", y=col, ax=ax, hue="diagnosis", legend=False,
                    palette=[PALETTE[0], PALETTE[1]], order=list(TARGET_LABELS.values()))
        ax.set_xlabel("")
        ax.set_title(col)
    fig.suptitle("Spread and outliers by diagnosis", y=1.02)
    fig.tight_layout()
    return fig


def plot_clinical_bands(df: pd.DataFrame):
    """Disease rate per guideline risk band: motivates (or not) the 'clinical'
    feature set. A monotonic rise suggests the band carries signal; a flat
    profile suggests the raw value is already enough."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (name, (source, edges)) in zip(axes, config.BAND_EDGES.items()):
        b = band(df[source], edges)
        g = df.assign(b=b).dropna(subset=["b"]).groupby("b")[config.TARGET].agg(["mean", "size"])
        labels = [config.BAND_LABELS[name][int(i)] for i in g.index]
        ax.bar(labels, g["mean"], color="#55A868")
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.annotate(f"n={n}", (i, m), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name} (from {source})")
        ax.set_ylabel("Disease rate")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.suptitle("Disease rate across clinical risk bands", y=1.03)
    fig.tight_layout()
    return fig


def plot_sex_balance(df: pd.DataFrame):
    """Why subgroup evaluation matters: group sizes and prevalence by sex."""
    g = df.groupby("sex")[config.TARGET].agg(["size", "mean"])
    g.index = [config.SUBGROUP_NAMES.get(int(i), str(i)) for i in g.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.5))
    a1.bar(g.index, g["size"], color="#8172B2")
    a1.set_title("Patients by sex")
    a2.bar(g.index, g["mean"], color="#C44E52")
    a2.set_ylim(0, 1)
    a2.set_title("Disease prevalence by sex")
    for ax, col, fmt in [(a1, "size", "{:.0f}"), (a2, "mean", "{:.0%}")]:
        for i, v in enumerate(g[col]):
            ax.annotate(fmt.format(v), (i, v), ha="center", va="bottom")
    fig.tight_layout()
    return fig


FIGURES = {
    "01_class_balance": plot_class_balance,
    "02_missing_values": plot_missing,
    "03_numeric_distributions": plot_numeric_distributions,
    "04_categorical_disease_rates": plot_categorical_rates,
    "05_correlation_heatmap": plot_correlation,
    "06_outliers_boxplots": plot_outliers,
    "07_clinical_risk_bands": plot_clinical_bands,
    "08_sex_balance": plot_sex_balance,
}


def generate_all(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR / "eda") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fig = fn(df)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    df = load_clean()
    for p in generate_all(df):
        print(f"wrote {p}")
    print("\nSummary statistics:\n", df[config.INPUT_FEATURES].describe().T.round(2))


if __name__ == "__main__":
    main()
