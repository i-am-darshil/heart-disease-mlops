"""The monitoring chart's dashboard and rules must stay in step with the API's metrics."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from heart import train
from heart.api.main import create_app
from heart.api.schemas import EXAMPLE_PATIENT
from heart.data import load_processed
from scripts import make_dashboard
from tests.test_train import FAST_MODELS

CHART = Path("helm/monitoring")
DASHBOARD = CHART / "files/dashboards/heart-api.json"
ALERTS = CHART / "files/alerts.yml"
# Metrics Prometheus itself creates, not the API
BUILTIN = {"up", "ALERTS"}
# prometheus_client reads these from /proc, so they only exist on Linux (Docker, CI, the VM)
LINUX_ONLY = {"process_cpu_seconds_total", "process_resident_memory_bytes"}


def test_dashboard_json_is_generated_from_the_script():
    assert DASHBOARD.read_text() == make_dashboard.render(), (
        "dashboard is stale: run `make dashboard`"
    )


def test_dashboard_panels_fit_the_grid_and_have_unique_ids():
    panels = json.loads(DASHBOARD.read_text())["panels"]
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids))
    for p in panels:
        assert p["gridPos"]["x"] + p["gridPos"]["w"] <= 24, p["title"]


def _rules() -> list[dict]:
    return [r for g in yaml.safe_load(ALERTS.read_text())["groups"] for r in g["rules"]]


def _metric_names(expr: str) -> set[str]:
    # Identifiers directly followed by { or [ or standing alone, minus PromQL keywords
    expr = re.sub(r'"[^"]*"', "", expr)  # drop label values
    expr = re.sub(r"\{[^}]*\}", "{}", expr)  # drop label matchers
    names = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr))
    keywords = {"sum", "by", "rate", "increase", "histogram_quantile", "max", "min", "count",
                "abs", "scalar", "vector", "on", "and", "or", "absent", "round", "le", "label",
                "feature", "handler", "status", "endpoint", "pod", "s", "m", "e"}  # fmt: skip
    return {n for n in names if n not in keywords and not re.fullmatch(r"\d+[smh]?", n)}


@pytest.fixture(scope="module")
def exported_metrics(tmp_path_factory) -> set[str]:
    """Every metric name the API exposes after serving one prediction and one bad request."""
    models_dir = tmp_path_factory.mktemp("models")
    train.save_outputs(train.run(models=FAST_MODELS, df=load_processed()), models_dir=models_dir)
    with TestClient(create_app(models_dir)) as c:
        c.post("/predict", json=EXAMPLE_PATIENT | {"thal": None})
        c.post("/predict", json={"age": 999})
        text = c.get("/metrics").text
    names = set()
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name = re.split(r"[{ ]", line, maxsplit=1)[0]
            names.add(name)
            names.add(re.sub(r"_(bucket|sum|count|total|created)$", "", name))
    return names


def test_every_metric_the_rules_and_dashboard_use_is_exported(exported_metrics):
    recorded = {r["record"] for r in _rules() if "record" in r}
    exprs = [r["expr"] for r in _rules()]
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        exprs += [t["expr"] for t in panel.get("targets", [])]
    used = set().union(*map(_metric_names, exprs))
    unknown = used - exported_metrics - recorded - BUILTIN
    if not sys.platform.startswith("linux"):
        unknown -= LINUX_ONLY
    assert not unknown, f"queries reference metrics the API doesn't export: {sorted(unknown)}"


def test_every_alert_has_severity_and_summary():
    alerts = [r for r in _rules() if "alert" in r]
    assert len(alerts) >= 6
    for a in alerts:
        assert a["labels"]["severity"] in ("warning", "critical"), a["alert"]
        assert a["annotations"]["summary"], a["alert"]


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_chart_embeds_the_rules_and_dashboard_files():
    rendered = subprocess.run(
        ["helm", "template", "mon", str(CHART), "--namespace", "monitoring"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    docs = {(d["kind"], d["metadata"]["name"]): d for d in yaml.safe_load_all(rendered) if d}
    config = docs[("ConfigMap", "mon-prometheus")]["data"]
    assert yaml.safe_load(config["alerts.yml"]) == yaml.safe_load(ALERTS.read_text())
    scrape = yaml.safe_load(config["prometheus.yml"])["scrape_configs"]
    assert any(job["job_name"] == "kubernetes-pods" for job in scrape)
    dashboards = docs[("ConfigMap", "mon-grafana-dashboards")]["data"]
    assert json.loads(dashboards["heart-api.json"]) == json.loads(DASHBOARD.read_text())
    # Prometheus may only list pods in the scraped namespace
    assert docs[("Role", "mon-prometheus-discovery")]["metadata"]["namespace"] == "heart"
