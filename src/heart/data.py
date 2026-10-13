"""Load the raw UCI file, clean it, and write the processed dataset.

Missing values are kept as NaN on purpose: imputation happens inside the model
pipeline (Phase 2), so imputation statistics are learned from training folds only
and the exact same logic runs at prediction time.
"""

from pathlib import Path

import pandas as pd

from heart.config import (
    CATEGORICAL_VALUES,
    COLUMNS,
    FEATURES,
    NUMERIC_RANGES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET,
)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks a missing value."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def invalid_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows with an out-of-range or unknown value (NaN is allowed)."""
    bad = pd.Series(False, index=df.index)
    for col, (low, high) in NUMERIC_RANGES.items():
        bad |= df[col].notna() & ~df[col].between(low, high)
    for col, allowed in CATEGORICAL_VALUES.items():
        bad |= df[col].notna() & ~df[col].isin(allowed)
    return bad


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and invalid rows, binarize the target, and fix dtypes."""
    df = raw.drop_duplicates()
    df = df[~invalid_rows(df)].copy()

    # num is 0 (no disease) to 4 (severe). The task is presence vs absence.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # Nullable integers keep the coded columns as ints while still allowing NaN
    for col in CATEGORICAL_VALUES:
        df[col] = df[col].astype("Int64")
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def load_processed(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in CATEGORICAL_VALUES:
        df[col] = df[col].astype("Int64")
    return df


def main() -> None:
    raw = load_raw()
    df = clean(raw)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Raw rows: {len(raw)}, cleaned rows: {len(df)}")
    missing = df.isna().sum()
    print("Missing values (imputed later in the model pipeline):")
    print(missing[missing > 0].to_string())
    print(f"Class balance: {df[TARGET].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Wrote {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
