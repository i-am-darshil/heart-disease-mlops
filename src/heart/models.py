"""Candidate models and the hyperparameter grids searched for each."""

from dataclasses import dataclass, field

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart.config import RANDOM_STATE
from heart.features import build_preprocessor


@dataclass
class Candidate:
    name: str
    estimator: ClassifierMixin
    # Keys are bare estimator params; build_pipeline prefixes them with "model__"
    grid: dict[str, list] = field(default_factory=dict)


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def candidates() -> list[Candidate]:
    return [
        # Baseline: predicts at random in the training class ratio. Every real model must beat it.
        Candidate("dummy", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
        Candidate(
            "logistic_regression",
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE),
            {
                # Inverse regularization strength: smaller C = simpler model
                "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                # 0 = L2 (shrinks all coefficients), 1 = L1 (can zero some out)
                "l1_ratio": [0.0, 1.0],
            },
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            {
                "max_depth": [None, 4, 8],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
        Candidate(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0],
            },
        ),
    ]
