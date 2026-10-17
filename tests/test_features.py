"""Tests for feature engineering and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from heart import config
from heart.features import ClinicalFeatures, band, build_preprocessor


def test_hr_reserve_ratio_value():
    X = pd.DataFrame({"age": [40], "thalach": [180]})
    out = ClinicalFeatures("ratio").fit_transform(X)
    assert out.loc[0, "hr_reserve_ratio"] == pytest.approx(1.0)


def test_raw_feature_set_adds_nothing():
    X = pd.DataFrame({"age": [40], "thalach": [180]})
    out = ClinicalFeatures("raw").fit_transform(X)
    assert list(out.columns) == ["age", "thalach"]


def test_unknown_feature_set_rejected():
    with pytest.raises(ValueError):
        ClinicalFeatures("everything").fit(pd.DataFrame({"age": [40]}))


@pytest.mark.parametrize("value,expected", [
    (44, 0), (45, 1), (54.9, 1), (55, 2), (64, 2), (65, 3), (90, 3),
])
def test_age_band_edges_are_left_closed(value, expected):
    edges = config.BAND_EDGES["age_band"][1]
    assert band(pd.Series([value]), edges).iloc[0] == expected


@pytest.mark.parametrize("chol,expected", [(199, 0), (200, 1), (239, 1), (240, 2)])
def test_cholesterol_bands_follow_ncep(chol, expected):
    assert band(pd.Series([chol]), config.BAND_EDGES["chol_band"][1]).iloc[0] == expected


@pytest.mark.parametrize("bp,expected", [(119, 0), (120, 1), (130, 2), (139, 2), (140, 3)])
def test_bp_bands_follow_acc_aha(bp, expected):
    assert band(pd.Series([bp]), config.BAND_EDGES["bp_band"][1]).iloc[0] == expected


def test_band_keeps_missing_as_nan():
    out = band(pd.Series([np.nan, 210.0]), config.BAND_EDGES["chol_band"][1])
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1


def test_clinical_set_adds_all_bands(xy):
    X, _ = xy
    out = ClinicalFeatures("clinical").fit_transform(X)
    for col in config.FEATURE_SETS["clinical"]:
        assert col in out.columns


@pytest.mark.parametrize("feature_set", config.FEATURE_SET_ORDER)
def test_preprocessor_output_has_no_nans(xy, feature_set):
    X, _ = xy
    Xt = build_preprocessor(feature_set).fit_transform(X)
    assert not np.isnan(Xt).any()


@pytest.mark.parametrize("feature_set", config.FEATURE_SET_ORDER)
def test_preprocessor_output_width(xy, feature_set):
    X, _ = xy
    pre = build_preprocessor(feature_set).fit(X)
    n_onehot = sum(len(v) for v in config.CATEGORY_LEVELS.values())
    expected = (len(config.NUMERIC_FEATURES) + len(config.FEATURE_SETS[feature_set])
                + len(config.BINARY_FEATURES) + n_onehot)
    assert pre.transform(X).shape[1] == expected
    assert len(pre.get_feature_names_out()) == expected


def test_numeric_features_are_standardised(xy):
    X, _ = xy
    Xt = build_preprocessor("raw").fit_transform(X)
    num = Xt[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    odd = X.head(1).copy()
    odd["cp"] = 99
    assert pre.transform(odd).shape[0] == 1


def test_all_missing_optional_fields_are_imputed(xy):
    X, _ = xy
    pre = build_preprocessor("clinical").fit(X)
    row = X.head(1).copy()
    row[["ca", "thal", "chol"]] = np.nan
    assert not np.isnan(pre.transform(row)).any()
