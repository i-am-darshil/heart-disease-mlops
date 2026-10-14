"""Generate the Grafana dashboard JSON (helm/monitoring/files/dashboards/heart-api.json).

    python scripts/make_dashboard.py          (or: make dashboard)

Writing panels as code keeps them consistent (units, colours, layout) and reviewable;
the generated JSON is committed and Grafana loads it at startup (provisioning).
tests/test_monitoring.py fails if the JSON is out of date with this script.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "helm/monitoring/files/dashboards/heart-api.json"
DS = {"type": "prometheus", "uid": "prometheus"}
JOB = 'job="heart-api"'

# Status colours, used only for good/warning/critical states
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
# Categorical series colours, assigned in this fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


class Layout:
    """Places panels left to right on Grafana's 24-column grid, wrapping to new rows."""

    def __init__(self):
        self.x = self.y = self.row_h = 0
        self.next_id = 1

    def place(self, panel: dict, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        panel |= {"id": self.next_id, "gridPos": {"x": self.x, "y": self.y, "w": w, "h": h}}
        self.next_id += 1
        self.x += w
        self.row_h = max(self.row_h, h)
        return panel

    def row(self, title: str) -> dict:
        self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        panel = self.place({"type": "row", "title": title, "collapsed": False, "panels": []}, 24, 1)
        self.x, self.y, self.row_h = 0, self.y + 1, 0
        return panel


def target(expr: str, legend: str = "", ref: str = "A", instant: bool = False) -> dict:
    t = {"datasource": DS, "expr": expr, "legendFormat": legend or "__auto", "refId": ref}
    if instant:
        t |= {"instant": True, "range": False}
    return t


def thresholds(*steps) -> dict:
    """steps: (value, colour) pairs; the first value is the base (None)."""
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


def stat(title, expr, unit="short", steps=((None, BLUE),), description="", decimals=None,
         instant=True, mappings=None, color_mode="value", no_value=None):  # fmt: skip
    defaults = {"unit": unit, "thresholds": thresholds(*steps), "color": {"mode": "thresholds"}}
    if no_value:
        defaults["noValue"] = no_value
    if decimals is not None:
        defaults["decimals"] = decimals
    if mappings:
        defaults["mappings"] = mappings
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": [target(expr, instant=instant)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": color_mode,
            "graphMode": "none",
            "textMode": "value",
            "justifyMode": "center",
        },
    }


def timeseries(title, targets, unit="short", description="", colors=None, stack=False,
               min_=None, max_=None, fill=10, steps=None, threshold_style="off",
               soft_max=None, no_value=None):  # fmt: skip
    custom = {
        "lineWidth": 2,
        "fillOpacity": fill,
        "showPoints": "never",
        "spanNulls": False,  # no traffic = a gap, not a line drawn across it
        "axisSoftMin": 0,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        "thresholdsStyle": {"mode": threshold_style},
    }
    defaults = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}}
    if min_ is not None:
        defaults["min"] = min_
    if max_ is not None:
        defaults["max"] = max_
    if steps:
        defaults["thresholds"] = thresholds(*steps)
    if soft_max is not None:  # axis goes at least this high, more if the data does
        custom["axisSoftMax"] = soft_max
    if no_value:
        defaults["noValue"] = no_value
    overrides = [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colour}}],
        }
        for name, colour in (colors or {}).items()
    ]
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": overrides},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def build() -> dict:
    lay = Layout()
    panels = []
    add = panels.append

    # --- Overview: the one-glance health check ------------------------------------------
    add(lay.row("Overview"))
    add(lay.place(stat(
        "Pods serving", f"sum(up{{{JOB}}})",
        steps=((None, CRIT), (1, WARN), (2, GOOD)),
        description="heart-api pods Prometheus can scrape (the chart runs 2).",
    ), 4, 4))  # fmt: skip
    add(lay.place(stat(
        "Model loaded", f"min(heart_model_loaded{{{JOB}}})",
        steps=((None, CRIT), (1, GOOD)),
        mappings=[{"type": "value", "options": {"0": {"text": "NO"}, "1": {"text": "yes"}}}],
        description="Every pod has its model loaded (0 on any pod = NO).",
    ), 4, 4))  # fmt: skip
    add(lay.place(stat(
        "Requests / s", "heart:requests:rate5m", unit="reqps", decimals=2,
        description="All API requests, averaged over 5 minutes.",
    ), 4, 4))  # fmt: skip
    add(lay.place(stat(
        "Server errors (5xx)", "heart:server_error_ratio:rate5m", unit="percentunit",
        decimals=1, steps=((None, GOOD), (0.01, WARN), (0.05, CRIT)),
        description="Share of requests that failed on the server, last 5 minutes. Alert above 5%.",
        no_value="no traffic",
    ), 4, 4))  # fmt: skip
    add(lay.place(stat(
        "p95 latency /predict", "heart:predict_latency_seconds:p95_5m", unit="s",
        decimals=1, steps=((None, GOOD), (0.25, WARN), (0.5, CRIT)),
        description="95% of /predict requests finish within this time. Alert above 500 ms.",
        no_value="no traffic",
    ), 4, 4))  # fmt: skip
    add(lay.place(stat(
        "Firing alerts", 'count(ALERTS{alertstate="firing"}) or vector(0)',
        steps=((None, GOOD), (1, CRIT)),
        description="Alert rules currently firing. Details: /prometheus/alerts",
    ), 4, 4))  # fmt: skip

    # --- Traffic and errors ----------------------------------------------------------
    add(lay.row("Traffic and errors"))
    add(lay.place(timeseries(
        "Requests by endpoint and status",
        [target(
            f"sum by (handler, status) (rate(http_requests_total{{{JOB}}}[1m]))",
            "{{handler}} {{status}}",
        )],  # fmt: skip
        unit="reqps", stack=True, fill=60,
        description="Per second, 1-minute average. 422 = rejected by input validation.",
    ), 12, 8))  # fmt: skip
    add(lay.place(timeseries(
        "Latency /predict",
        [
            target(f'histogram_quantile(0.50, sum by (le) (rate(http_request_duration_seconds_bucket{{{JOB}, handler="/predict"}}[1m])))', "p50", "A"),  # noqa: E501
            target(f'histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{{JOB}, handler="/predict"}}[1m])))', "p95", "B"),  # noqa: E501
            target(f'histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{{{JOB}, handler="/predict"}}[1m])))', "p99", "C"),  # noqa: E501
            target(f"histogram_quantile(0.95, sum by (le) (rate(heart_inference_seconds_bucket{{{JOB}}}[1m])))", "p95 model only", "D"),  # noqa: E501
        ],
        unit="s", colors={"p50": BLUE, "p95": ORANGE, "p99": MAGENTA, "p95 model only": AQUA},
        fill=0,
        description="Whole request (HTTP + validation + model) vs the model's own share.",
    ), 12, 8))  # fmt: skip
    add(lay.place(timeseries(
        "Error rates",
        [
            target("heart:server_error_ratio:rate5m", "server errors (5xx)", "A"),
            target("heart:validation_error_ratio:rate5m", "invalid input (422)", "B"),
        ],
        unit="percentunit", min_=0, colors={"server errors (5xx)": ORANGE, "invalid input (422)": BLUE},  # noqa: E501
        steps=((None, "transparent"), (0.2, CRIT)), threshold_style="dashed", soft_max=0.3,
        description="5-minute ratios. Dashed line: the invalid-input alert threshold (20%).",
    ), 12, 7))  # fmt: skip
    add(lay.place(timeseries(
        "Invalid input by endpoint",
        [target(
            f"sum by (endpoint) (rate(heart_validation_errors_total{{{JOB}}}[1m]))",
            "{{endpoint}}",
        )],  # fmt: skip
        unit="reqps",
        description="Requests rejected by validation per second, by endpoint.",
        no_value="No invalid requests",
    ), 12, 7))  # fmt: skip

    # --- Model behaviour (drift) -----------------------------------------------------
    add(lay.row("Model behaviour (drift monitoring)"))
    add(lay.place(timeseries(
        "Share flagged as heart disease: live vs training",
        [
            target("heart:positive_rate:rate15m", "live (15 min)", "A"),
            target(f"max(heart_reference_positive_rate{{{JOB}}})", "training baseline", "B"),
        ],
        unit="percentunit", min_=0, max_=1, fill=0,
        colors={"live (15 min)": BLUE, "training baseline": "#898781"},
        description=(
            "If the live share moves far from the training baseline, the patients being "
            "scored differ from the training data. Alert when the gap exceeds 20 points."
        ),
    ), 12, 8))  # fmt: skip
    add(lay.place(timeseries(
        "Input drift: feature mean vs training (in training SDs)",
        [target("heart:feature_mean_shift_sd:rate10m", "{{feature}}")],
        unit="none", fill=0, min_=-2, max_=2,
        colors={"age": BLUE, "trestbps": ORANGE, "chol": AQUA, "thalach": YELLOW, "oldpeak": MAGENTA},  # noqa: E501
        steps=((None, "transparent"), (0.5, WARN)), threshold_style="dashed",
        description=(
            "0 = live inputs average the same as training. ±0.5 SD (dashed) is the alert "
            "threshold. Try: ./run.sh load drift"
        ),
    ), 12, 8))  # fmt: skip
    add(lay.place({
        "type": "bargauge",
        "title": "Predicted probability distribution (last 15 min)",
        "description": "How many predictions fell in each probability band. "
                       "The decision threshold is about 0.25.",
        "datasource": DS,
        "targets": [target(
            f"sum by (le) (increase(heart_prediction_probability_bucket{{{JOB}}}[15m]))",
            "{{le}}", instant=True,
        ) | {"format": "heatmap"}],  # fmt: skip
        "fieldConfig": {"defaults": {
            "unit": "short", "decimals": 0, "color": {"mode": "fixed", "fixedColor": BLUE},
        }, "overrides": []},  # fmt: skip
        "options": {
            "orientation": "vertical",
            "displayMode": "basic",
            "showUnfilled": False,
            "valueMode": "text",
            "namePlacement": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }, 8, 8))  # fmt: skip
    add(lay.place(timeseries(
        "Predictions by class",
        [target(f"sum by (label) (rate(heart_predictions_total{{{JOB}}}[1m]))", "{{label}}")],
        unit="short", stack=True, fill=60,
        colors={"heart disease": ORANGE, "no heart disease": BLUE},
        description="Predictions per second (batch requests count every patient).",
    ), 8, 8))  # fmt: skip
    add(lay.place(timeseries(
        "Missing inputs (imputed)",
        [target(f"sum by (feature) (rate(heart_missing_features_total{{{JOB}}}[5m]))", "{{feature}}")],  # noqa: E501
        unit="short",
        description="Fields sent as null or left out, per second. The model imputes them.",
    ), 8, 8))  # fmt: skip

    # --- Pods and model version ------------------------------------------------------
    add(lay.row("Pods"))
    add(lay.place(timeseries(
        "Memory per pod",
        [target(f"process_resident_memory_bytes{{{JOB}}}", "{{pod}}")],
        unit="bytes", fill=0, description="Resident memory. The chart's limit is 512 MiB.",
    ), 6, 7))  # fmt: skip
    add(lay.place(timeseries(
        "CPU per pod",
        [target(f"rate(process_cpu_seconds_total{{{JOB}}}[1m])", "{{pod}}")],
        unit="percentunit", fill=0, description="CPU cores in use (1 = one full core).",
    ), 6, 7))  # fmt: skip
    add(lay.place({
        "type": "table",
        "title": "Model being served",
        "description": "One row per pod. During a rollout, two versions appear side by side.",
        "datasource": DS,
        "targets": [target(f"heart_model_info{{{JOB}}}", instant=True) | {"format": "table"}],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "transformations": [{
            "id": "organize",
            "options": {
                "excludeByName": {"Time": True, "Value": True, "__name__": True, "job": True,
                                  "instance": True, "namespace": True},
                "renameByName": {"model_version": "model version", "version": "image tag"},
            },
        }],  # fmt: skip
        "options": {"showHeader": True, "cellHeight": "sm"},
    }, 12, 7))  # fmt: skip

    return {
        "uid": "heart-api",
        "title": "Heart disease API",
        "description": "Traffic, errors, latency and model drift for the heart-disease API",
        "tags": ["heart-disease", "mlops"],
        "timezone": "browser",
        "editable": False,
        "graphTooltip": 1,  # shared crosshair across panels
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "schemaVersion": 41,
        "version": 1,
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
        "links": [
            {
                "title": "Prometheus alerts",
                "type": "link",
                "url": "/prometheus/alerts",
                "targetBlank": True,
            },
            {"title": "API docs", "type": "link", "url": "/docs", "targetBlank": True},
        ],  # fmt: skip
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT}")
