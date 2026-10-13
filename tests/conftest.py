import numpy as np
import pandas as pd
import pytest

from heart.config import COLUMNS


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small raw-format frame: valid rows, a duplicate, missing values and a bad row."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],  # duplicate of the row above
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, np.nan, 3, 0],  # missing ca
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, np.nan, 4],  # missing thal
        [56, 1, 9, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 0],  # cp=9 is not a valid code
        [200, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0, 3, 0],  # age 200 is out of range
    ]
    return pd.DataFrame(rows, columns=COLUMNS)
