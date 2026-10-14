"""Reproducibility check: train twice from scratch and prove the results are identical.

    python -m heart.reproduce          (./run.sh reproduce / make reproduce)

Each run trains in its own temporary folder with nothing shared but the code, the
committed dataset and the environment. The check passes only if both runs give
byte-identical model files, identical predictions on every row and identical metrics.
Run it inside the Docker image: that's the reference environment, and results from
other platforms (e.g. native macOS) can differ in near-tied model choices.
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from heart import train
from heart.config import FEATURES
from heart.data import load_processed
from heart.packaging import load_model, sha256_file


def train_once(out_dir: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train.run()
    train.save_outputs(result, models_dir=out_dir)
    model, metadata = load_model(out_dir)
    X = load_processed()[FEATURES].astype("float64")
    return {
        "sha256": sha256_file(out_dir / "model.joblib"),
        "proba": model.predict_proba(X)[:, 1],
        "metrics": json.loads((out_dir / "metrics.json").read_text()),
        "metadata": metadata,
    }


def compare(a: dict, b: dict) -> list[tuple[str, bool, str]]:
    same_proba = np.array_equal(a["proba"], b["proba"])
    max_diff = float(np.abs(a["proba"] - b["proba"]).max())
    return [
        ("model file (SHA-256)", a["sha256"] == b["sha256"], a["sha256"][:16]),
        ("predictions, all 303 rows", same_proba, f"max |diff| = {max_diff:.1e}"),
        ("CV + test metrics", a["metrics"] == b["metrics"], ""),
        ("selected model + params", a["metadata"]["params"] == b["metadata"]["params"],
         f"{a['metadata']['estimator']} {a['metadata']['params']}"),
        ("decision threshold", a["metadata"]["decision_threshold"]
         == b["metadata"]["decision_threshold"], f"{a['metadata']['decision_threshold']:.6f}"),
    ]  # fmt: skip


def main() -> int:
    runs = []
    for i in (1, 2):
        print(f"Training run {i}/2 ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            runs.append(train_once(Path(tmp)))

    checks = compare(*runs)
    env = runs[0]["metadata"]["environment"]
    libs = ", ".join(f"{lib} {env[lib]}" for lib in ("scikit-learn", "numpy", "pandas"))
    print(f"\nEnvironment: Python {env['python']} on {env['platform']}, {libs}")
    print("\nReproducibility check (run 1 vs run 2):")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}")

    if all(ok for _, ok, _ in checks):
        print("\nREPRODUCIBLE: two independent training runs produced identical models.")
        return 0
    print("\nNOT REPRODUCIBLE: see FAIL lines above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
