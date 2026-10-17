"""Tests for data loading and cleaning."""

import numpy as np
import pandas as pd

from heart import config
import pytest

from heart.data import DataValidationError, clean, load_raw, missing_report, split_xy, validate


def test_load_raw_parses_question_marks_as_nan(raw_file):
    df = load_raw([raw_file])
    assert list(df.columns) == config.RAW_COLUMNS + ["site"]
    assert df["ca"].isna().sum() > 0
    assert (df["site"] == "cleveland").all()


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}
    assert "num" not in clean_df.columns


def test_target_binarisation_rule():
    raw = pd.DataFrame([[50 + n, 1, 4, 130, 250, 0, 0, 150, 0, 1.0, 2, 0, 3, n] for n in range(5)],
                       columns=config.RAW_COLUMNS)
    out = clean(raw)
    assert out[config.TARGET].tolist() == [0, 1, 1, 1, 1]


def test_zero_cholesterol_and_bp_become_missing():
    row = [50, 1, 4, 0, 0, 0, 0, 150, 0, 1.0, 2, 0, 3, 1]
    raw = pd.DataFrame([row], columns=config.RAW_COLUMNS)
    out = clean(raw)
    assert np.isnan(out.loc[0, "chol"]) and np.isnan(out.loc[0, "trestbps"])


def test_invalid_category_codes_are_nulled():
    row = [50, 1, 9, 130, 250, 0, 0, 150, 0, 1.0, 2, 0, 5, 0]  # cp=9, thal=5 are invalid
    raw = pd.DataFrame([row], columns=config.RAW_COLUMNS)
    out = clean(raw)
    assert np.isnan(out.loc[0, "cp"]) and np.isnan(out.loc[0, "thal"])


def test_duplicates_dropped():
    row = [50, 1, 4, 130, 250, 0, 0, 150, 0, 1.0, 2, 0, 3, 0]
    out = clean(pd.DataFrame([row, row], columns=config.RAW_COLUMNS))
    assert len(out) == 1


def test_clean_does_not_mutate_input(raw_file):
    raw = load_raw([raw_file])
    snapshot = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_missing_report_lists_only_missing_columns(clean_df):
    rep = missing_report(clean_df)
    assert (rep["missing"] > 0).all()
    assert "age" not in rep.index


def test_split_xy_returns_expected_columns(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.INPUT_FEATURES
    assert len(X) == len(y)


def test_validate_accepts_clean_data(clean_df):
    assert isinstance(validate(clean_df), list)


def test_validate_rejects_missing_column(clean_df):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate(clean_df.drop(columns=["thalach"]))


def test_validate_rejects_too_few_rows(clean_df):
    with pytest.raises(DataValidationError, match="rows"):
        validate(clean_df.head(10))


def test_validate_rejects_single_class(clean_df):
    df = clean_df.copy()
    df[config.TARGET] = 0
    with pytest.raises(DataValidationError, match="single class"):
        validate(df)


def test_validate_rejects_out_of_range_values(clean_df):
    df = clean_df.copy()
    df.loc[0, "age"] = 250  # e.g. a unit / parsing error
    with pytest.raises(DataValidationError, match="age"):
        validate(df)


def test_validate_warns_on_heavy_missingness(clean_df):
    df = clean_df.copy()
    df.loc[: int(len(df) * 0.6), "ca"] = np.nan
    assert any("ca" in w for w in validate(df))
