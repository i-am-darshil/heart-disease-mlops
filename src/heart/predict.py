"""Inference: turn raw patient records into predictions with a confidence score.

    python -m heart.predict examples/sample_request.json
    python -m heart.predict examples/sample_batch.json
    echo '{"age": 63, ...}' | python -m heart.predict -

Input is one JSON object (or a list of them) keyed by the 13 UCI feature names.
Any feature may be null/missing; the pipeline imputes it. The serving API (Phase 5)
uses the same `Predictor`, so batch scripts and the API can't disagree.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from heart.config import FEATURES, MODELS_DIR
from heart.packaging import ModelPackageError, load_model


@dataclass
class Prediction:
    prediction: int  # 1 = heart disease, 0 = no heart disease (using the tuned threshold)
    label: str
    probability: float  # model's probability of heart disease
    confidence: float  # probability of the predicted class
    threshold: float


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR, verify: bool = True):
        self.model, self.metadata = load_model(models_dir, verify=verify)
        self.threshold = float(self.metadata["decision_threshold"])
        self.labels = self.metadata["classes"]

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]

    def to_frame(self, records: list[dict]) -> pd.DataFrame:
        unknown = sorted({k for r in records for k in r} - set(FEATURES))
        if unknown:
            raise ValueError(f"Unknown feature(s): {unknown}. Expected: {FEATURES}")
        # Missing keys and nulls both become NaN; float64 matches the training contract
        return pd.DataFrame.from_records(records, columns=FEATURES).astype("float64")

    def predict(self, records: list[dict]) -> list[Prediction]:
        X = self.to_frame(records)
        proba = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)  # applies the tuned threshold stored in the model
        return [
            Prediction(
                prediction=int(y),
                label=self.labels[str(int(y))],
                probability=round(float(p), 4),
                confidence=round(float(p if y == 1 else 1 - p), 4),
                threshold=round(self.threshold, 4),
            )
            for y, p in zip(preds, proba, strict=True)
        ]

    def predict_one(self, record: dict) -> Prediction:
        return self.predict([record])[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict heart disease risk from JSON input")
    parser.add_argument("input", help="JSON file (one record or a list), or - for stdin")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    data = json.loads(raw)
    records = data if isinstance(data, list) else [data]
    try:
        predictor = Predictor(args.models_dir)
    except ModelPackageError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    try:
        results = predictor.predict(records)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    output = [asdict(r) for r in results]
    print(json.dumps(output if isinstance(data, list) else output[0], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
