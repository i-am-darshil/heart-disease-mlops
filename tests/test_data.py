import pandas as pd

from heart.config import CATEGORICAL_VALUES, FEATURES, TARGET
from heart.data import clean, invalid_rows, load_processed, load_raw


def test_load_raw_treats_question_marks_as_missing(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(path)
    assert df.shape == (1, 14)
    assert df["ca"].isna().all()


def test_invalid_rows_flags_out_of_range_and_unknown_codes(raw_df):
    flagged = raw_df[invalid_rows(raw_df)]
    assert set(flagged.index) == {6, 7}


def test_invalid_rows_does_not_flag_missing_values(raw_df):
    assert not invalid_rows(raw_df).loc[[4, 5]].any()


def test_clean_drops_duplicates_and_invalid_rows(raw_df):
    assert len(clean(raw_df)) == 5


def test_clean_keeps_missing_values_for_pipeline_imputation(raw_df):
    df = clean(raw_df)
    assert df["ca"].isna().sum() == 1
    assert df["thal"].isna().sum() == 1


def test_clean_binarizes_target(raw_df):
    df = clean(raw_df)
    assert set(df[TARGET].unique()) <= {0, 1}
    # num values 2, 1 and 4 all mean "disease present"
    assert df[TARGET].tolist() == [0, 1, 1, 0, 1]


def test_clean_output_schema(raw_df):
    df = clean(raw_df)
    assert list(df.columns) == FEATURES + [TARGET]
    assert "num" not in df.columns
    for col in CATEGORICAL_VALUES:
        assert isinstance(df[col].dtype, pd.Int64Dtype)


def test_committed_processed_dataset_matches_schema():
    """The committed CSV is what CI trains on, so guard its shape and contents."""
    df = load_processed()
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 303
    assert set(df[TARGET].unique()) == {0, 1}
    assert not invalid_rows(df).any()
