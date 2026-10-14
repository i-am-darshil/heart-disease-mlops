"""The serving image must install exactly the versions the model was trained with."""

from pathlib import Path


def pins(path: str) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            out[name.split("[")[0].lower().replace("_", "-")] = version
    return out


def test_serve_pins_match_training_pins():
    serve, train = pins("requirements-serve.txt"), pins("requirements.txt")
    mismatched = {k: (v, train.get(k)) for k, v in serve.items() if train.get(k) != v}
    assert not mismatched, f"requirements-serve.txt differs from requirements.txt: {mismatched}"


def test_serve_excludes_training_only_libraries():
    assert not {"mlflow", "matplotlib", "seaborn"} & set(pins("requirements-serve.txt"))
