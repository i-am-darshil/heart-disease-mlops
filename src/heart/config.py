"""Central configuration: paths, feature groups and seeds shared by every stage."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
# Pinned so a silently changed upstream file can't alter our results
RAW_DATA_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
RAW_DATA_PATH = RAW_DIR / "processed.cleveland.data"
PROCESSED_DATA_PATH = PROCESSED_DIR / "heart.csv"

# Column order of the raw UCI file (it has no header row)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Valid value ranges, used to reject corrupt rows during cleaning (and later by the API).
# Numeric bounds are clinically plausible limits; categorical sets are the UCI codes.
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
}
CATEGORICAL_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

# Human-readable labels for the coded columns (EDA plots and docs only)
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}

RANDOM_STATE = 42
