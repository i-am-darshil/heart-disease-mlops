"""Preprocessing: one ColumnTransformer that turns a raw patient record into model features.

It lives inside the model Pipeline, so imputation and scaling statistics are learned
from training data only and the identical transformation runs at prediction time.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import BINARY_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_VALUES, NUMERIC_FEATURES


def _onehot_name(feature: str, level) -> str:
    """'cp' + 4.0 -> 'cp_4' (the imputer hands the encoder floats)."""
    if isinstance(level, float) and level.is_integer():
        level = int(level)
    return f"{feature}_{level}"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Categories are fixed from the UCI codebook rather than learned, so a level that is
    # rare (restecg=1 has 4 patients) still gets its column even if a CV fold lacks it.
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[sorted(CATEGORICAL_VALUES[c]) for c in CATEGORICAL_FEATURES],
                    feature_name_combiner=_onehot_name,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
