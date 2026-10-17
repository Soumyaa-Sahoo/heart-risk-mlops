"""End-to-end: train.main() with MLflow writing to a temp directory."""

import json

import pytest

from heart import config


@pytest.mark.slow
def test_train_main_end_to_end(clean_csv, tmp_path, monkeypatch):
    import mlflow

    from heart import train

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "models" / "metadata.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")

    summary = train.main(["--data", str(clean_csv), "--n-iter", "2",
                          "--models", "logistic_regression", "random_forest",
                          "--feature-sets", "raw", "clinical"])

    assert summary["selected"].split("__")[0] in {"logistic_regression", "random_forest"}
    assert (tmp_path / "models" / "model.joblib").exists()
    meta = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert 0.2 <= meta["decision_threshold"] <= 0.7
    assert meta["feature_set"] in {"raw", "clinical"}
    assert set(meta["oof_subgroup_metrics"]) == {"female", "male"}
    json.dumps(meta, allow_nan=False)  # strict JSON: safe for the API to serve
    figs = tmp_path / "reports" / "figures"
    for name in ["model_comparison.png", "feature_set_ablation.png",
                 "calibration_comparison.png"]:
        assert (figs / name).exists()
    assert (figs / summary["selected"] / "subgroup_by_sex.png").exists()

    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])
    assert len(runs) >= 4 + 4 * 2  # 4 parent runs + 2 child candidates each
