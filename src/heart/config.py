"""Central configuration: paths, column definitions and defaults.

Everything that more than one module needs to agree on lives here, so the
training code, the API and the tests can never drift apart on column names.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("HEART_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "heart_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-risk")

# --- UCI source -----------------------------------------------------------
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
DEFAULT_SITES = ("cleveland",)

# --- Schema ---------------------------------------------------------------
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous / ordinal features: median-imputed and standardised.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Binary flags: mode-imputed, passed through unscaled.
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes with no natural order: one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# --- Engineered feature sets (added by heart.features.ClinicalFeatures) -----
# Compared against each other in an ablation study (python -m heart.train).
#   raw      : the 13 original attributes only
#   ratio    : + hr_reserve_ratio = thalach / (220 - age)
#   clinical : + guideline-based risk bands for age, cholesterol and blood pressure
FEATURE_SETS = {
    "raw": [],
    "ratio": ["hr_reserve_ratio"],
    "clinical": ["hr_reserve_ratio", "age_band", "chol_band", "bp_band"],
}
# Order = increasing complexity; used by the parsimony rule in model selection.
FEATURE_SET_ORDER = ["raw", "ratio", "clinical"]
# Model families from most to least interpretable / cheapest to serve.
MODEL_COMPLEXITY_ORDER = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
DEFAULT_FEATURE_SET = "ratio"

# Band edges (left-closed): value < edge[0] -> 0, edge[0] <= value < edge[1] -> 1, ...
BAND_EDGES = {
    # age groups commonly used in cardiovascular epidemiology
    "age_band": ("age", [45, 55, 65]),            # <45, 45-54, 55-64, 65+
    # NCEP ATP III total cholesterol: desirable / borderline high / high
    "chol_band": ("chol", [200, 240]),            # <200, 200-239, 240+
    # ACC/AHA 2017 systolic BP: normal / elevated / stage 1 / stage 2
    "bp_band": ("trestbps", [120, 130, 140]),     # <120, 120-129, 130-139, 140+
}
BAND_LABELS = {
    "age_band": ["<45", "45-54", "55-64", "65+"],
    "chol_band": ["desirable <200", "borderline 200-239", "high 240+"],
    "bp_band": ["normal <120", "elevated 120-129", "stage 1 130-139", "stage 2 140+"],
}

# Subgroup used for fairness evaluation (0 = female, 1 = male)
SUBGROUP_COLUMN = "sex"
SUBGROUP_NAMES = {0: "female", 1: "male"}

# Valid code sets, used for cleaning and for API validation.
CATEGORY_LEVELS = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

# Plausible ranges used by data validation (heart.data.validate). Values
# outside these after cleaning indicate a unit or parsing error, so the
# pipeline stops rather than silently training on them.
VALID_RANGES = {
    "age": (18, 110),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 230),
    "oldpeak": (-3, 10),
    "ca": (0, 3),
}
MIN_ROWS = 50
MAX_MISSING_FRACTION = 0.5   # per feature; above this, warn (multi-site data)

# Physiologically impossible values that the raw files use to mean "missing".
ZERO_MEANS_MISSING = ["trestbps", "chol"]

API_VERSION = "v1"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
