"""Tests for model building, tuning, evaluation and inference."""

import numpy as np
import pytest

from heart import config
from heart.evaluate import (
    choose_threshold,
    classification_metrics,
    flatten_subgroups,
    subgroup_metrics,
)
from heart.models import build_pipeline, get_candidates
from heart.predict import HeartModel


@pytest.mark.parametrize("name", list(get_candidates()))
def test_every_candidate_fits_and_predicts_probabilities(xy, name):
    X, y = xy
    pipe = build_pipeline(get_candidates()[name].estimator).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_model_beats_chance(fitted_pipeline, xy):
    X, y = xy
    m = classification_metrics(y, fitted_pipeline.predict_proba(X)[:, 1])
    assert m["roc_auc"] > 0.7


def test_metrics_keys_and_ranges():
    m = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc", "brier"}
    assert all(0 <= v <= 1 for v in m.values())
    assert m["accuracy"] == 1.0


def test_threshold_favours_recall():
    y = np.array([1, 1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.45, 0.3, 0.35, 0.2, 0.1])
    t, _ = choose_threshold(y, proba)
    assert t <= 0.3  # catching all positives is worth a false alarm under F2


def test_brier_rewards_confident_correct_predictions():
    y = [0, 1, 1, 0]
    sharp = classification_metrics(y, [0.05, 0.95, 0.9, 0.1])["brier"]
    vague = classification_metrics(y, [0.4, 0.6, 0.6, 0.4])["brier"]
    assert sharp < vague


def test_subgroup_metrics_split_by_group():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.2, 0.1, 0.3, 0.8, 0.2, 0.7, 0.6])
    sex = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sub = subgroup_metrics(y, proba, sex, 0.5, config.SUBGROUP_NAMES)
    assert set(sub) == {"female", "male"}
    assert sub["female"]["recall"] == 0.5   # one of two female positives caught
    assert sub["male"]["recall"] == 1.0
    assert sub["female"]["n"] == 4
    flat = flatten_subgroups(sub, "oof")
    assert flat["oof_recall_gap"] == 0.5
    assert "oof_n_female" not in flat


def test_subgroup_auc_nan_for_single_class_group():
    sub = subgroup_metrics([1, 1, 0, 1], [0.9, 0.8, 0.2, 0.7], [0, 0, 1, 1], 0.5)
    assert np.isnan(sub["0"]["roc_auc"])
    assert "oof_roc_auc_0" not in flatten_subgroups(sub, "oof")


def _result(fs, auc, name="random_forest"):
    return {"name": name, "feature_set": fs, "cv": {"cv_roc_auc_mean": auc}}


def test_selection_prefers_simpler_set_within_tolerance():
    from heart.train import select_best

    results = [_result("clinical", 0.903), _result("ratio", 0.900), _result("raw", 0.880)]
    assert select_best(results, tolerance=0.005)["feature_set"] == "ratio"


def test_selection_takes_richer_set_when_clearly_better():
    from heart.train import select_best

    results = [_result("clinical", 0.92), _result("ratio", 0.90), _result("raw", 0.89)]
    assert select_best(results, tolerance=0.005)["feature_set"] == "clinical"


def test_selection_prefers_interpretable_model_when_close():
    from heart.train import select_best

    results = [_result("ratio", 0.904, "hist_gradient_boosting"),
               _result("ratio", 0.901, "logistic_regression"),
               _result("raw", 0.902, "random_forest")]
    best = select_best(results, tolerance=0.005)
    assert best["name"] == "logistic_regression"


def test_inference_latency_is_positive(fitted_pipeline, xy):
    from heart.train import inference_latency_ms

    X, _ = xy
    assert inference_latency_ms(fitted_pipeline, X, repeats=3) > 0


def test_search_model_runs(xy):
    from heart.train import best_cv_metrics, search_model

    X, y = xy
    cand = get_candidates()["logistic_regression"]
    search = search_model(cand.estimator, cand.search_space, X, y, feature_set="clinical",
                          n_iter=2, cv_folds=3)
    metrics = best_cv_metrics(search)
    assert 0 <= metrics["cv_roc_auc_mean"] <= 1
    assert "cv_recall_std" in metrics


def test_reloaded_pipeline_predicts_identically(fitted_pipeline, model_files, xy):
    """Training-serving consistency: the serialized artefact must reproduce
    the in-memory pipeline's probabilities exactly."""
    X, _ = xy
    reloaded = HeartModel.load(*model_files).pipeline
    np.testing.assert_array_equal(fitted_pipeline.predict_proba(X),
                                  reloaded.predict_proba(X))


def test_heart_model_roundtrip(model_files, sample_patient):
    model = HeartModel.load(*model_files)
    out = model.predict([sample_patient, {**sample_patient, "ca": None, "thal": None}])
    assert len(out) == 2
    for p in out:
        assert p["prediction"] in (0, 1)
        assert 0.5 <= p["confidence"] <= 1
        assert p["prediction"] == int(p["probability_disease"] >= model.threshold)
