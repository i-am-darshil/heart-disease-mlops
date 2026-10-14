"""Render CI results as Markdown for the GitHub Actions run summary page.

    python scripts/ci_summary.py tests  reports/junit.xml coverage.xml
    python scripts/ci_summary.py train  models/metrics.json models/model_metadata.json

Writes to stdout; the workflow appends it to $GITHUB_STEP_SUMMARY. Missing files
(e.g. a step failed before producing them) are reported, not fatal.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def tests_summary(junit: Path, coverage: Path) -> str:
    if not junit.exists():
        return "### Tests\n\nNo test report was produced (the step failed before pytest ran).\n"
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total, failures, errors, skipped = (
        int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")
    )
    passed = total - failures - errors - skipped
    status = "✅ all passed" if failures + errors == 0 else f"❌ {failures + errors} failed"
    lines = [
        "### Tests",
        "",
        "| Result | Passed | Failed | Errors | Skipped | Time |",
        "|---|---|---|---|---|---|",
        f"| {status} | {passed} | {failures} | {errors} | {skipped} | "
        f"{float(suite.get('time', 0)):.1f}s |",
    ]
    failed = [
        f"{case.get('classname')}::{case.get('name')}"
        for case in suite.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    if failed:
        lines += ["", "**Failing tests**", ""] + [f"- `{name}`" for name in failed[:20]]
    if coverage.exists():
        rate = float(ET.parse(coverage).getroot().get("line-rate", 0))
        lines += ["", f"**Line coverage:** {rate:.0%}"]
    return "\n".join(lines) + "\n"


def train_summary(metrics_path: Path, metadata_path: Path) -> str:
    if not metrics_path.exists():
        return (
            "### Training\n\nNo model was saved: training failed or the quality gate "
            "rejected the model. See the `train` step log.\n"
        )
    metrics = json.loads(metrics_path.read_text())
    lines = [
        "### Training",
        "",
        "| Model | CV ROC-AUC | CV recall | CV precision |",
        "|---|---|---|---|",
    ]
    for name, result in metrics["cv"].items():
        mean, std = result["mean"], result["std"]
        marker = " **(selected)**" if name == metrics["selected_model"] else ""
        lines.append(
            f"| {name}{marker} | {mean['roc_auc']:.3f} ± {std['roc_auc']:.3f} | "
            f"{mean['recall']:.3f} | {mean['precision']:.3f} |"
        )
    test = metrics["test"]
    lines += [
        "",
        f"**Selected:** `{metrics['selected_model']}` `{metrics['params']}`, "
        f"decision threshold **{metrics['threshold']:.3f}** "
        f"(for ≥{metrics['target_recall']:.0%} CV recall)",
        "",
        "| Held-out test | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|",
        "| tuned threshold | " + " | ".join(f"{test[m]:.3f}" for m in
                                            ("accuracy", "precision", "recall", "f1", "roc_auc"))
        + " |",
    ]  # fmt: skip
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        env = meta["environment"]
        lines += [
            "",
            f"**Model version** `{meta['model_version']}` · "
            f"sha256 `{meta['model_sha256'][:16]}…` · "
            f"Python {env['python']} / scikit-learn {env['scikit-learn']} on {env['platform']}",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    kind, *paths = argv
    if kind == "tests":
        print(tests_summary(*map(Path, paths)))
    elif kind == "train":
        print(train_summary(*map(Path, paths)))
    else:
        print(f"unknown summary kind: {kind}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
