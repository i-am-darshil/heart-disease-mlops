import numpy as np

from heart.config import BINARY_FEATURES, CATEGORICAL_VALUES, FEATURES, NUMERIC_FEATURES
from heart.data import clean
from heart.features import build_preprocessor

N_ONEHOT = sum(len(v) for k, v in CATEGORICAL_VALUES.items() if k not in BINARY_FEATURES)
N_OUT = len(NUMERIC_FEATURES) + N_ONEHOT + len(BINARY_FEATURES)


def test_output_has_expected_width_and_no_missing(raw_df):
    X = clean(raw_df)[FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert out.shape == (len(X), N_OUT)
    assert not np.isnan(out).any()


def test_numeric_features_are_standardized(raw_df):
    X = clean(raw_df)[FEATURES]
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)[:, : len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-9)


def test_unseen_category_level_does_not_crash(raw_df):
    X = clean(raw_df)[FEATURES]
    pre = build_preprocessor().fit(X)
    unseen = X.iloc[[0]].copy()
    unseen["restecg"] = 1  # not present in the fixture's training rows
    out = pre.transform(unseen)
    names = list(pre.get_feature_names_out())
    assert out[0, names.index("restecg_1")] == 1


def test_imputation_uses_training_statistics_only(raw_df):
    X = clean(raw_df)[FEATURES]
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["chol"] = np.nan
    imputed = pre.named_transformers_["num"].named_steps["impute"].statistics_
    assert imputed[NUMERIC_FEATURES.index("chol")] == X["chol"].median()
    assert not np.isnan(pre.transform(row)).any()
