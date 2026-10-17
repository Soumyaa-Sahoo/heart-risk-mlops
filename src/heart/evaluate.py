"""Metrics, decision-threshold selection and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI runners and containers have no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Threshold metrics + ROC-AUC + Brier score (lower Brier = better calibrated)."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def subgroup_metrics(y_true, proba, groups, threshold: float,
                     names: dict | None = None) -> dict[str, dict[str, float]]:
    """Per-subgroup metrics (e.g. by sex).

    The UCI data is roughly two-thirds male, so a model can look good overall
    while missing disease in women. ROC-AUC is NaN when a subgroup contains
    only one class (possible in small test splits).
    """
    names = names or {}
    y_true, proba, groups = map(np.asarray, (y_true, proba, groups))
    out = {}
    for g in np.unique(groups[~pd.isna(groups)]):
        m = groups == g
        yt, pr = y_true[m], proba[m]
        yp = (pr >= threshold).astype(int)
        out[names.get(int(g), str(g))] = {
            "n": int(m.sum()),
            "prevalence": float(yt.mean()),
            "recall": recall_score(yt, yp, zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "roc_auc": roc_auc_score(yt, pr) if len(np.unique(yt)) == 2 else float("nan"),
            "brier": brier_score_loss(yt, pr),
        }
    return out


def flatten_subgroups(sub: dict, prefix: str) -> dict[str, float]:
    """{'female': {'recall': .8}} -> {'<prefix>_recall_female': .8}, NaNs dropped,
    plus the absolute recall gap between the first two groups."""
    flat = {f"{prefix}_{k}_{g}": float(v) for g, d in sub.items() for k, v in d.items()
            if k != "n" and not np.isnan(v)}
    groups = list(sub)
    if len(groups) >= 2:
        flat[f"{prefix}_recall_gap"] = abs(sub[groups[0]]["recall"] - sub[groups[1]]["recall"])
    return flat


def choose_threshold(y_true, proba, beta: float = 2.0,
                     grid: np.ndarray | None = None) -> tuple[float, float]:
    """Pick the probability cut-off maximising F-beta on out-of-fold predictions.

    beta=2 weights recall twice as heavily as precision. For a screening tool
    a missed disease case (false negative) is costlier than a false alarm that
    gets resolved by a follow-up test.
    """
    grid = np.round(np.arange(0.20, 0.71, 0.01), 2) if grid is None else grid
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
              for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def save_evaluation_plots(pipeline, X_test, y_test, threshold: float,
                          out_dir: Path, title: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{title}\nthreshold = {threshold:.2f}")
    paths.append(_save(fig, out_dir / "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC curve (test set)")
    paths.append(_save(fig, out_dir / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name=title)
    ax.set_title("Precision-recall curve (test set)")
    paths.append(_save(fig, out_dir / "pr_curve.png"))

    paths.append(save_permutation_importance(pipeline, X_test, y_test, out_dir, title))
    return paths


def save_calibration_plot(y_true, proba, out_dir: Path, title: str, n_bins: int = 8) -> Path:
    """Reliability diagram: does a predicted 0.7 really mean ~70% have disease?
    Uses quantile bins so each point has a similar number of patients."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, proba)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(4.8, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#C44E52", label=f"model (Brier {brier:.3f})")
    ax.set_ylabel("Observed disease rate")
    ax.set_title(f"Calibration (out-of-fold): {title}")
    ax.legend(loc="upper left", fontsize=8)
    axh.hist(proba, bins=20, range=(0, 1), color="#4C72B0")
    axh.set_xlabel("Predicted probability")
    axh.set_ylabel("Count")
    return _save(fig, out_dir / "calibration.png")


def save_subgroup_plot(sub: dict, out_dir: Path, title: str) -> Path:
    metrics = ["recall", "precision", "roc_auc"]
    groups = list(sub)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, g in enumerate(groups):
        vals = [np.nan_to_num(sub[g][m]) for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=f"{g} (n={sub[g]['n']}, prev {sub[g]['prevalence']:.0%})")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(groups) - 1) / 2, metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Performance by sex (out-of-fold): {title}")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_dir / "subgroup_by_sex.png")


def save_permutation_importance(pipeline, X_test, y_test, out_dir: Path, title: str,
                                n_repeats: int = 20) -> Path:
    """Model-agnostic importance on the *raw input* columns, so all three model
    families are compared on the same footing."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=0, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns,
                         "mean": result.importances_mean,
                         "std": result.importances_std})
           .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.barh(imp["feature"], imp["mean"], xerr=imp["std"], color="#4C72B0")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance: {title}")
    return _save(fig, out_dir / "permutation_importance.png")


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
