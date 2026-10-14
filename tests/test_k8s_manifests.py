"""k8s/heart-api.yaml is generated from the Helm chart; fail if someone forgot to regenerate."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_plain_manifests_match_the_helm_chart():
    committed = Path("k8s/heart-api.yaml").read_text()
    header = "".join(Path("k8s/HEADER.txt").read_text().splitlines(keepends=True)[:7])
    rendered = subprocess.run(
        ["helm", "template", "heart-api", "helm/heart-api", "--namespace", "heart",
         "--set", "image.tag=latest", "--skip-tests"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    lines = rendered.splitlines(keepends=True)
    rendered = "".join(line for line in lines if "helm.sh/chart" not in line)
    assert committed == header + rendered, "k8s/heart-api.yaml is stale: run `make k8s-manifests`"


def test_plain_manifests_use_the_public_registry_image():
    text = Path("k8s/heart-api.yaml").read_text()
    assert 'image: "ghcr.io/i-am-darshil/heart-disease-api:latest"' in text
    assert "kind: Deployment" in text and "kind: Ingress" in text
