"""Candidate models and their hyper-parameter search spaces.

Three model families with deliberately different inductive biases:

* logistic_regression - linear, well-calibrated, interpretable baseline.
* random_forest       - bagged trees; captures interactions, robust to scale.
* hist_gradient_boosting - boosted trees; usually strongest on tabular data
                           but most prone to overfitting a 300-row dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart import config
from heart.features import build_preprocessor


@dataclass
class Candidate:
    name: str
    estimator: Any
    search_space: dict[str, Any] = field(default_factory=dict)


def get_candidates(random_state: int = config.RANDOM_STATE) -> dict[str, Candidate]:
    return {
        "logistic_regression": Candidate(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            search_space={
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": Candidate(
            name="random_forest",
            # n_jobs=1: the hyper-parameter search already parallelises over folds
            estimator=RandomForestClassifier(n_jobs=1, random_state=random_state),
            search_space={
                "model__n_estimators": randint(100, 600),
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "hist_gradient_boosting": Candidate(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(random_state=random_state),
            search_space={
                "model__learning_rate": loguniform(1e-2, 3e-1),
                "model__max_depth": [2, 3, 4, None],
                "model__max_iter": randint(50, 400),
                "model__min_samples_leaf": randint(5, 40),
                "model__l2_regularization": uniform(0.0, 1.0),
            },
        ),
    }


def build_pipeline(estimator, feature_set: str = config.DEFAULT_FEATURE_SET) -> Pipeline:
    """Full model: raw patient fields in, probability out."""
    return Pipeline([
        ("preprocess", build_preprocessor(feature_set=feature_set)),
        ("model", estimator),
    ])
