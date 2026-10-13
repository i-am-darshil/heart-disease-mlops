from heart import config


def test_feature_groups_cover_all_raw_inputs():
    raw_inputs = [c for c in config.COLUMNS if c != "num"]
    assert sorted(config.FEATURES) == sorted(raw_inputs)


def test_feature_groups_do_not_overlap():
    assert len(config.FEATURES) == len(set(config.FEATURES))
