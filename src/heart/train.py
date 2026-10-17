"""Train, tune, compare and package the heart-disease classifier.

Usage:
    python -m heart.train            # full ablation: 3 models x 3 feature sets
    python -m heart.train --n-iter 10 --feature-sets ratio
    python -m heart.train --models logistic_regression random_forest

What happens
------------
1. Stratified 80/20 train/test split (the test set is touched only for final reporting).
2. For every (model family, feature set) pair: RandomizedSearchCV with stratified
   5-fold CV, optimising ROC-AUC while also recording accuracy/precision/recall/F1.
   Every candidate configuration is logged as a nested MLflow child run.
3. Out-of-fold (OOF) probabilities for the best configuration are used to
   (a) choose a recall-weighted decision threshold (F2),
   (b) assess calibration (reliability diagram + Brier score), and
   (c) evaluate performance separately for female and male patients.
   None of these choices look at the test set.
4. Selection uses a parsimony rule: among all runs within ``--tolerance`` of
   the best mean CV ROC-AUC, prefer the simplest feature set, then the
   higher AUC. Extra features must *earn* their place.
5. The selected pipeline is saved to models/model.joblib with metadata.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from heart import config  # noqa: E402
from heart.data import file_sha256, load_clean, split_xy, validate  # noqa: E402
from heart.evaluate import (  # noqa: E402
    CV_SCORING,
    choose_threshold,
    classification_metrics,
    flatten_subgroups,
    save_calibration_plot,
    save_evaluation_plots,
    save_subgroup_plot,
    subgroup_metrics,
)
from heart.models import build_pipeline, get_candidates  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure training logic (no MLflow) - unit-testable
# ---------------------------------------------------------------------------
def search_model(estimator, search_space, X, y, *, feature_set=config.DEFAULT_FEATURE_SET,
                 n_iter=30, cv_folds=config.CV_FOLDS, random_state=config.RANDOM_STATE):
    """Randomised hyper-parameter search; returns the fitted search object."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        build_pipeline(estimator, feature_set=feature_set),
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=CV_SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def best_cv_metrics(search) -> dict[str, float]:
    """Mean and std of every CV metric for the best configuration."""
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for m in CV_SCORING:
        out[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
        out[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
    out["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][i])
    # Gap between train and validation AUC: a quick overfitting indicator.
    out["cv_overfit_gap"] = out["cv_train_roc_auc_mean"] - out["cv_roc_auc_mean"]
    return out


def oof_probabilities(search, X, y, cv_folds=config.CV_FOLDS,
                      random_state=config.RANDOM_STATE) -> np.ndarray:
    """Out-of-fold P(disease) for every training row, from the best config.

    A different fold seed from the search avoids re-using the exact splits
    the hyper-parameters were selected on.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state + 1)
    return cross_val_predict(clone(search.best_estimator_), X, y, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]


def select_best(results: list[dict], tolerance: float = 0.005) -> dict:
    """Parsimony rule.

    Among runs within `tolerance` of the best mean CV ROC-AUC (differences
    that small are well inside fold-to-fold noise on ~240 training rows):
      1. prefer the more interpretable / cheaper model family,
      2. then the simpler feature set,
      3. then the higher CV ROC-AUC.
    """
    top = max(r["cv"]["cv_roc_auc_mean"] for r in results)
    eligible = [r for r in results if r["cv"]["cv_roc_auc_mean"] >= top - tolerance]
    return min(eligible, key=lambda r: (_rank(config.MODEL_COMPLEXITY_ORDER, r["name"]),
                                        config.FEATURE_SET_ORDER.index(r["feature_set"]),
                                        -r["cv"]["cv_roc_auc_mean"]))


def _rank(order: list[str], item: str) -> int:
    return order.index(item) if item in order else len(order)


def inference_latency_ms(pipeline, X, repeats: int = 20) -> float:
    """Median single-request latency (ms) of the full pipeline: what the API pays."""
    row = X.head(1)
    pipeline.predict_proba(row)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def _clean_param(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _strip_prefix(params: dict) -> dict:
    return {k.replace("model__", ""): _clean_param(v) for k, v in params.items()}


# ---------------------------------------------------------------------------
# MLflow orchestration
# ---------------------------------------------------------------------------
def _log_child_runs(search, run_name: str) -> None:
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{run_name}-cand-{i:02d}", nested=True):
            mlflow.set_tag("candidate_rank", int(res["rank_test_roc_auc"][i]))
            mlflow.log_params(_strip_prefix(params))
            mlflow.log_metrics(
                {f"cv_{m}_mean": float(res[f"mean_test_{m}"][i]) for m in CV_SCORING}
            )


def run_model(name, feature_set, candidate, X_train, X_test, y_train, y_test, *,
              n_iter, data_hash) -> dict:
    run_name = f"{name}__{feature_set}"
    log.info("=== %s: searching %d configurations ===", run_name, n_iter)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model_family": name,
            "feature_set": feature_set,
            "data_sha256": data_hash[:12],
        })
        search = search_model(candidate.estimator, candidate.search_space, X_train, y_train,
                              feature_set=feature_set, n_iter=n_iter)
        _log_child_runs(search, run_name)
        pipe = search.best_estimator_
        cv_metrics = best_cv_metrics(search)

        # --- decisions made on out-of-fold training predictions only ---
        oof = oof_probabilities(search, X_train, y_train)
        threshold, f2 = choose_threshold(y_train, oof)
        oof_metrics = classification_metrics(y_train, oof, threshold)
        groups_train = X_train[config.SUBGROUP_COLUMN].to_numpy()
        oof_sub = subgroup_metrics(y_train, oof, groups_train, threshold, config.SUBGROUP_NAMES)

        # --- final reporting on the untouched test split ---
        proba = pipe.predict_proba(X_test)[:, 1]
        test_default = classification_metrics(y_test, proba, 0.5)
        test_tuned = classification_metrics(y_test, proba, threshold)
        test_sub = subgroup_metrics(y_test, proba, X_test[config.SUBGROUP_COLUMN].to_numpy(),
                                    threshold, config.SUBGROUP_NAMES)

        latency_ms = inference_latency_ms(pipe, X_test)

        mlflow.log_params({"n_iter": n_iter, "cv_folds": config.CV_FOLDS,
                           "cv_strategy": "StratifiedKFold(shuffle=True)",
                           "search": "RandomizedSearchCV", "refit_metric": "roc_auc",
                           "random_state": config.RANDOM_STATE,
                           "test_size": config.TEST_SIZE,
                           "numeric_imputation": "median", "categorical_imputation": "mode",
                           "scaling": "StandardScaler", "encoding": "OneHotEncoder",
                           "threshold_rule": "max F2 on out-of-fold predictions",
                           "feature_set": feature_set,
                           "n_engineered": len(config.FEATURE_SETS[feature_set]),
                           "train_rows": len(X_train), "test_rows": len(X_test),
                           **_strip_prefix(search.best_params_)})
        mlflow.log_metrics({
            **cv_metrics,
            "oof_threshold": threshold,
            "oof_f2": f2,
            "inference_latency_ms": latency_ms,
            **{f"oof_{k}": v for k, v in oof_metrics.items()},
            **flatten_subgroups(oof_sub, "oof"),
            **{f"test_{k}": v for k, v in test_default.items()},
            **{f"test_tuned_{k}": v for k, v in test_tuned.items()},
            **flatten_subgroups(test_sub, "test"),
        })

        fig_dir = config.FIGURES_DIR / run_name
        plots = save_evaluation_plots(pipe, X_test, y_test, threshold, fig_dir, run_name)
        plots.append(save_calibration_plot(y_train, oof, fig_dir, run_name))
        plots.append(save_subgroup_plot(oof_sub, fig_dir, run_name))
        for p in plots:
            mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_dict({"oof": oof_sub, "test": test_sub}, "subgroup_metrics.json")

        signature = infer_signature(X_train, pipe.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature,
                                 input_example=X_train.head(3))

        log.info("%s  CV AUC %.3f +/- %.3f | OOF Brier %.3f | recall F/M %.2f/%.2f | "
                 "test AUC %.3f", run_name, cv_metrics["cv_roc_auc_mean"],
                 cv_metrics["cv_roc_auc_std"], oof_metrics["brier"],
                 oof_sub.get("female", {}).get("recall", float("nan")),
                 oof_sub.get("male", {}).get("recall", float("nan")), test_default["roc_auc"])

        return {
            "name": name,
            "feature_set": feature_set,
            "run_name": run_name,
            "run_id": run.info.run_id,
            "pipeline": pipe,
            "best_params": _strip_prefix(search.best_params_),
            "threshold": threshold,
            "latency_ms": latency_ms,
            "cv": cv_metrics,
            "oof": oof_metrics,
            "oof_subgroups": oof_sub,
            "oof_proba": oof,
            "test": test_default,
            "test_tuned": test_tuned,
            "test_subgroups": test_sub,
        }


# ---------------------------------------------------------------------------
# Cross-run reports
# ---------------------------------------------------------------------------
def save_comparison(results: list[dict]) -> Path:
    rows = []
    for r in results:
        row = {"model": r["name"], "feature_set": r["feature_set"], **r["cv"],
               "oof_brier": r["oof"]["brier"], "threshold": r["threshold"],
               "latency_ms": r["latency_ms"],
               **{f"test_{k}": v for k, v in r["test"].items()},
               "test_tuned_recall": r["test_tuned"]["recall"]}
        for g, d in r["oof_subgroups"].items():
            row[f"oof_recall_{g}"] = d["recall"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "model_comparison.csv"
    df.to_csv(path, index=False)
    return path


def save_ablation_plot(results: list[dict]) -> Path:
    """CV ROC-AUC for every model x feature set, with fold std as error bars."""
    models = list(dict.fromkeys(r["name"] for r in results))
    fsets = [f for f in config.FEATURE_SET_ORDER if any(r["feature_set"] == f for r in results)]
    lookup = {(r["name"], r["feature_set"]): r["cv"] for r in results}
    x = np.arange(len(models))
    width = 0.8 / len(fsets)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, fs in enumerate(fsets):
        means = [lookup.get((m, fs), {}).get("cv_roc_auc_mean", np.nan) for m in models]
        stds = [lookup.get((m, fs), {}).get("cv_roc_auc_std", 0) for m in models]
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=fs)
        for b, v in zip(bars, means):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7, xytext=(0, 4),
                            textcoords="offset points")
    ax.set_xticks(x + width * (len(fsets) - 1) / 2, models)
    lo = min(r["cv"]["cv_roc_auc_mean"] - r["cv"]["cv_roc_auc_std"] for r in results)
    ax.set_ylim(max(0.5, lo - 0.05), 1.01)
    ax.set_ylabel("5-fold CV ROC-AUC")
    ax.set_title("Feature-set ablation")
    ax.legend(title="feature set", fontsize=8)
    return _save(fig, config.FIGURES_DIR / "feature_set_ablation.png")


def save_model_comparison_plot(results: list[dict], feature_set: str) -> Path:
    """All CV metrics per model family, for the selected feature set."""
    sel = [r for r in results if r["feature_set"] == feature_set]
    metrics = list(CV_SCORING)
    x = np.arange(len(metrics))
    width = 0.8 / len(sel)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(sel):
        ax.bar(x + i * width, [r["cv"][f"cv_{m}_mean"] for m in metrics], width,
               yerr=[r["cv"][f"cv_{m}_std"] for m in metrics], capsize=3, label=r["name"])
    ax.set_xticks(x + width * (len(sel) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score (mean +/- std)")
    ax.set_title(f"Model comparison (feature set: {feature_set})")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, config.FIGURES_DIR / "model_comparison.png")


def save_calibration_comparison(results: list[dict], y_train, feature_set: str) -> Path:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for r in results:
        if r["feature_set"] != feature_set:
            continue
        fp, mp = calibration_curve(y_train, r["oof_proba"], n_bins=8, strategy="quantile")
        ax.plot(mp, fp, "o-", ms=4, label=f"{r['name']} (Brier {r['oof']['brier']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disease rate")
    ax.set_title(f"Calibration, out-of-fold (feature set: {feature_set})")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, config.FIGURES_DIR / "calibration_comparison.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def package_best(best: dict, data_hash: str, tolerance: float) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], config.MODEL_PATH)
    metadata = {
        "model_name": best["name"],
        "feature_set": best["feature_set"],
        "engineered_features": config.FEATURE_SETS[best["feature_set"]],
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "mlflow_run_id": best["run_id"],
        "selection_rule": (f"within {tolerance} of best CV ROC-AUC: simplest model family, "
                           "then simplest feature set"),
        "inference_latency_ms": best["latency_ms"],
        "decision_threshold": best["threshold"],
        "best_params": best["best_params"],
        "cv_metrics": best["cv"],
        "oof_metrics": best["oof"],
        "oof_subgroup_metrics": best["oof_subgroups"],
        "test_metrics": best["test"],
        "test_metrics_tuned_threshold": best["test_tuned"],
        "test_subgroup_metrics": best["test_subgroups"],
        "input_features": config.INPUT_FEATURES,
        "data_sha256": data_hash,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    config.METADATA_PATH.write_text(json.dumps(_sanitize(metadata), indent=2, default=str))
    log.info("Saved %s and %s", config.MODEL_PATH, config.METADATA_PATH)


def _sanitize(obj):
    """NaN/inf -> None recursively, so metadata is strict JSON (the API serves it)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main(argv: list[str] | None = None) -> dict:
    candidates = get_candidates()
    parser = argparse.ArgumentParser(description="Train and track heart-disease models")
    parser.add_argument("--models", nargs="+", default=list(candidates), choices=list(candidates))
    parser.add_argument("--feature-sets", nargs="+", default=config.FEATURE_SET_ORDER,
                        choices=config.FEATURE_SET_ORDER)
    parser.add_argument("--n-iter", type=int, default=30, help="Configurations per run")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="CV AUC margin within which the simpler feature set wins")
    parser.add_argument("--data", type=Path, default=config.CLEAN_CSV)
    parser.add_argument("--register", action="store_true",
                        help="Register the selected model in the MLflow model registry")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_clean(args.data)
    validate(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    data_hash = file_sha256(args.data)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    results = [
        run_model(name, fs, candidates[name], X_train, X_test, y_train, y_test,
                  n_iter=args.n_iter, data_hash=data_hash)
        for fs in args.feature_sets
        for name in args.models
    ]

    best = select_best(results, args.tolerance)
    top = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    log.info("Highest CV AUC: %s (%.4f); selected under parsimony rule: %s (%.4f)",
             top["run_name"], top["cv"]["cv_roc_auc_mean"],
             best["run_name"], best["cv"]["cv_roc_auc_mean"])

    artefacts = [
        save_comparison(results),
        save_ablation_plot(results),
        save_model_comparison_plot(results, best["feature_set"]),
        save_calibration_comparison(results, y_train, best["feature_set"]),
    ]
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected", "true")
        for p in artefacts:
            mlflow.log_artifact(str(p), artifact_path="comparison")

    if args.register:
        mlflow.register_model(f"runs:/{best['run_id']}/model", "heart-disease-classifier")

    package_best(best, data_hash, args.tolerance)
    summary = {r["run_name"]: {"cv_roc_auc": r["cv"]["cv_roc_auc_mean"],
                               "test_roc_auc": r["test"]["roc_auc"],
                               "oof_brier": r["oof"]["brier"]} for r in results}
    summary["selected"] = best["run_name"]
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(_sanitize(summary), indent=2))
    return summary


if __name__ == "__main__":
    main()
