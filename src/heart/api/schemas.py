"""Request/response models. Validation rules come from heart.config, the same ranges
and codes that clean the training data, so the API accepts exactly what the model saw.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from heart.config import (
    CATEGORICAL_VALUES,
    CATEGORY_LABELS,
    FEATURE_NAMES,
    FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_RANGES,
)

EXAMPLE_PATIENT = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}  # fmt: skip


def _field(name: str):
    """Optional float field: numeric features get min/max, coded ones a list of allowed codes."""
    if name in NUMERIC_FEATURES:
        low, high = NUMERIC_RANGES[name]
        return (
            Annotated[float | None, Field(ge=low, le=high, description=FEATURE_NAMES[name])],
            None,
        )
    codes = sorted(CATEGORICAL_VALUES[name])
    labels = CATEGORY_LABELS.get(name, {})
    meaning = ", ".join(f"{c} = {labels[c]}" for c in codes if c in labels)
    description = FEATURE_NAMES[name] + (f" ({meaning})" if meaning else "")
    # Literal of ints and floats so both 4 and 4.0 are accepted, nothing else
    allowed = Literal[tuple(codes) + tuple(float(c) for c in codes)]  # type: ignore[valid-type]
    return Annotated[allowed | None, Field(description=description)], None


# Every feature is optional: a missing or null value is imputed by the model pipeline
Patient = create_model(
    "Patient",
    __config__=ConfigDict(
        extra="forbid",  # a typo like "cholesterol" is an error, not silently ignored
        json_schema_extra={"examples": [EXAMPLE_PATIENT]},
    ),
    **{name: _field(name) for name in FEATURES},
)
Patient.__doc__ = "One patient's measurements. Any field may be omitted or null."


class Factor(BaseModel):
    feature: str = Field(description="Model feature, e.g. cp_4 (one-hot level) or thalach")
    description: str = Field(description="Readable form, e.g. 'Chest pain type = asymptomatic'")
    effect: float = Field(description="Change in log-odds of disease: > 0 raises risk")


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(description="1 = heart disease, 0 = no heart disease")
    label: str
    probability: float = Field(ge=0, le=1, description="Model probability of heart disease")
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class")
    threshold: float = Field(description="Decision threshold, tuned for >= 90% recall")
    top_factors: list[Factor] = Field(description="Features that most influenced this result")
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[Patient] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not ready"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str
    estimator: str
    params: dict
    decision_threshold: float
    created_at: str
    git_commit: str
    metrics: dict
    features: list[str]
    environment: dict
