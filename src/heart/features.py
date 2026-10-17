"""Feature engineering and the preprocessing transformer.

The whole preprocessing chain lives inside one sklearn object so that the
saved model artefact takes *raw* patient fields and does everything itself.
The API therefore never re-implements any preprocessing logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


def band(values: pd.Series, edges: list[float]) -> pd.Series:
    """Ordinal band index for each value; NaN stays NaN (imputed later)."""
    out = pd.Series(np.digitize(values, edges, right=False), index=values.index, dtype=float)
    out[values.isna()] = np.nan
    return out


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Add clinically motivated derived features for the chosen feature set.

    hr_reserve_ratio  ("ratio" and "clinical")
        Peak exercise heart rate / age-predicted maximum (220 - age).
        Failing to reach ~85% of predicted max (chronotropic incompetence)
        is an established marker of coronary disease. Raw ``thalach`` mixes
        this signal with age; the ratio separates it out.

    age_band, chol_band, bp_band  ("clinical" only)
        Ordinal risk categories using published guideline cut-offs (see
        config.BAND_EDGES). They encode clinical prior knowledge as step
        functions; whether that helps beyond the raw values is tested in the
        feature-set ablation rather than assumed.
    """

    def __init__(self, feature_set: str = config.DEFAULT_FEATURE_SET):
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_set not in config.FEATURE_SETS:
            raise ValueError(f"Unknown feature_set '{self.feature_set}'")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        added = config.FEATURE_SETS[self.feature_set]
        if "hr_reserve_ratio" in added:
            X["hr_reserve_ratio"] = X["thalach"] / (220 - X["age"]).clip(lower=1)
        for name, (source, edges) in config.BAND_EDGES.items():
            if name in added:
                X[name] = band(X[source], edges)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_) + config.FEATURE_SETS[self.feature_set]
        return np.asarray(names, dtype=object)


def build_preprocessor(feature_set: str = config.DEFAULT_FEATURE_SET) -> Pipeline:
    """Return the (unfitted) preprocessing pipeline.

    numeric (+ engineered) -> median impute -> standard scale
    binary                 -> most-frequent impute (already 0/1)
    categorical            -> most-frequent impute -> one-hot (unknown codes ignored)

    Risk bands are ordinal, so they go through the numeric branch: a linear
    model then sees "one band higher = one step more risk".
    """
    numeric = list(config.NUMERIC_FEATURES) + config.FEATURE_SETS[feature_set]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[config.CATEGORY_LEVELS[c] for c in config.CATEGORICAL_FEATURES],
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    columns = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("bin", binary_pipe, config.BINARY_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("clinical", ClinicalFeatures(feature_set=feature_set)),
        ("columns", columns),
    ])
