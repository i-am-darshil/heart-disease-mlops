import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from heart import evaluation, train
from heart.config import FEATURES, MIN_CV_ROC_AUC, TARGET_RECALL
from heart.data import load_processed
from heart.models import Candidate, build_pipeline, candidates

FAST_MODELS = [
    Candidate("dummy", DummyClassifier(strategy="stratified", random_state=0)),
    Candidate(
        "logistic_regression",
        LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.0, max_iter=5000),
        {"C": [0.1, 1.0]},
    ),
]


@pytest.fixture(scope="module")
def result():
    """One small training run on the real dataset, shared by the tests in this module."""
    return train.run(models=FAST_MODELS, df=load_processed())


def test_split_is_stratified_and_disjoint():
    df = load_processed()
    X_train, X_test, y_train, y_test = train.split(df)
    assert len(X_train) + len(X_test) == len(df)
    assert X_train.index.intersection(X_test.index).empty
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_model_beats_dummy_and_passes_quality_gate(result):
    dummy = next(t for t in result.tuned if t.name == "dummy")
    assert result.selected.cv_mean["roc_auc"] > dummy.cv_mean["roc_auc"] + 0.2
    assert result.selected.cv_mean["roc_auc"] >= MIN_CV_ROC_AUC
    assert train.quality_gate(result.selected, result.tuned) == []


def test_grid_results_record_every_grid_point(result):
    lr = next(t for t in result.tuned if t.name == "logistic_regression")
    assert len(lr.grid_results) == 2  # C in [0.1, 1.0]
    assert {"C", "mean_test_roc_auc", "std_test_recall"} <= set(lr.grid_results.columns)


def test_threshold_meets_recall_target_on_cv(result):
    proba = result.selected.oof_proba
    recall = evaluation.compute_metrics(result.y_train, proba, result.threshold)["recall"]
    assert recall >= TARGET_RECALL
    assert 0 < result.threshold < 1


def test_shipped_model_applies_threshold_and_returns_probabilities(result):
    proba = result.model.predict_proba(result.X_test)
    assert proba.shape == (len(result.X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    expected = (proba[:, 1] >= result.threshold).astype(int)
    np.testing.assert_array_equal(result.model.predict(result.X_test), expected)


def test_model_accepts_raw_records_with_missing_values(result):
    row = result.X_test.iloc[[0]].copy()
    row[["ca", "thal"]] = np.nan
    assert result.model.predict(row[FEATURES]).shape == (1,)


def test_quality_gate_rejects_weak_model(result):
    weak = train.TunedModel("weak", {}, {"roc_auc": 0.6}, {"roc_auc": 0.0}, None, np.array([]))
    failures = train.quality_gate(weak, result.tuned)
    assert any("below the minimum" in f for f in failures)


def test_select_prefers_simpler_model_within_one_standard_error():
    def tuned(name, auc, std):
        return train.TunedModel(name, {}, {"roc_auc": auc}, {"roc_auc": std}, None, np.array([]))

    simple, complex_ = tuned("simple", 0.90, 0.02), tuned("complex", 0.91, 0.05)
    assert train.select([simple, complex_]).name == "simple"  # 0.90 >= 0.91 - 0.05/sqrt(5)
    assert train.select([tuned("simple", 0.80, 0.0), complex_]).name == "complex"


def test_choose_threshold_picks_highest_threshold_meeting_recall():
    y = np.array([0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.6, 0.3, 0.7, 0.8, 0.9])
    assert evaluation.choose_threshold(y, proba, 0.75) == 0.7
    assert evaluation.choose_threshold(y, proba, 1.0) == 0.3


def test_save_outputs_writes_model_and_metrics(result, tmp_path):
    train.save_outputs(result, models_dir=tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert (tmp_path / "model.joblib").stat().st_size > 0
    assert metrics["selected_model"] == result.selected.name
    assert set(metrics["test"]) == set(evaluation.METRICS)


def test_save_figures_writes_pngs(result, tmp_path):
    paths = train.save_figures(result, out_dir=tmp_path)
    assert len(paths) == 4
    assert all(p.stat().st_size > 0 for p in paths)


def test_every_candidate_grid_matches_its_estimator():
    for c in candidates():
        assert set(c.grid) <= set(c.estimator.get_params())


def test_logistic_regression_grid_really_switches_between_l2_and_l1():
    """Guards against l1_ratio being silently ignored (scikit-learn < 1.8 needs elasticnet)."""
    lr = next(c for c in candidates() if c.name == "logistic_regression")
    df = load_processed()
    zero_weights = {}
    for l1_ratio in (0.0, 1.0):
        model = build_pipeline(clone(lr.estimator).set_params(C=0.1, l1_ratio=l1_ratio))
        model.fit(df[FEATURES], df["target"])
        zero_weights[l1_ratio] = int((np.abs(model.named_steps["model"].coef_) < 1e-8).sum())
    assert zero_weights[0.0] == 0  # L2 shrinks but keeps every weight
    assert zero_weights[1.0] > 0  # L1 sets some weights to exactly zero
