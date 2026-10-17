"""Shared fixtures.

Tests never hit the network: they use a small synthetic dataset written in
the exact raw UCI format (comma separated, '?' for missing, target 0-4).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from heart import config
from heart.data import clean, load_raw, split_xy
from heart.models import build_pipeline, get_candidates


def _synthetic_rows(n: int = 160, seed: int = 0) -> list[list]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sick = rng.random() < 0.45
        num = int(rng.integers(1, 5)) if sick else 0
        rows.append([
            int(rng.normal(58 if sick else 52, 8)),          # age
            int(rng.random() < (0.8 if sick else 0.55)),     # sex
            4 if sick and rng.random() < 0.7 else int(rng.integers(1, 4)),  # cp
            int(rng.normal(134, 17)),                        # trestbps
            int(rng.normal(245, 45)),                        # chol
            int(rng.random() < 0.15),                        # fbs
            int(rng.integers(0, 3)),                         # restecg
            int(rng.normal(139 if sick else 158, 20)),       # thalach
            int(rng.random() < (0.55 if sick else 0.15)),    # exang
            round(abs(rng.normal(1.6 if sick else 0.6, 1)), 1),  # oldpeak
            int(rng.integers(1, 4)),                         # slope
            "?" if i % 40 == 0 else int(rng.integers(0, 4) if sick else rng.integers(0, 2)),  # ca
            "?" if i % 55 == 0 else (7 if sick and rng.random() < 0.6 else 3),  # thal
            num,
        ])
    return rows


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / config.SITE_FILES["cleveland"]
    path.write_text("\n".join(",".join(str(v) for v in r) for r in _synthetic_rows()) + "\n")
    return path


@pytest.fixture(scope="session")
def clean_df(raw_file) -> pd.DataFrame:
    return clean(load_raw([raw_file]))


@pytest.fixture(scope="session")
def clean_csv(clean_df, tmp_path_factory):
    path = tmp_path_factory.mktemp("processed") / "heart_clean.csv"
    clean_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def xy(clean_df):
    return split_xy(clean_df)


@pytest.fixture(scope="session")
def fitted_pipeline(xy):
    X, y = xy
    est = get_candidates()["logistic_regression"].estimator
    return build_pipeline(est).fit(X, y)


@pytest.fixture(scope="session")
def model_files(fitted_pipeline, tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    model_path, meta_path = d / "model.joblib", d / "metadata.json"
    joblib.dump(fitted_pipeline, model_path)
    meta_path.write_text(json.dumps({
        "model_name": "logistic_regression", "model_version": "test",
        "decision_threshold": 0.4, "input_features": config.INPUT_FEATURES,
    }))
    return model_path, meta_path


@pytest.fixture
def sample_patient() -> dict:
    return {"age": 61, "sex": 1, "cp": 4, "trestbps": 140, "chol": 207, "fbs": 0,
            "restecg": 2, "thalach": 138, "exang": 1, "oldpeak": 1.9, "slope": 1,
            "ca": 1, "thal": 7}
