from heart.data import clean
from heart.eda import make_all_figures


def test_make_all_figures_writes_pngs(raw_df, tmp_path):
    paths = make_all_figures(clean(raw_df), out_dir=tmp_path)
    assert len(paths) == 4
    for path in paths:
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
