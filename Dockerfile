# syntax=docker/dockerfile:1
#
# Three targets, all driven by ./run.sh (don't call docker directly):
#   pipeline   lint, test, data, EDA, train, MLflow UI. Project mounted at /app at run
#              time, so outputs (models/, mlflow.db, figures) land on the host.
#   notebooks  pipeline + Jupyter, for ./run.sh jupyter / ./run.sh notebook
#   serve      the production API image: slim, non-root, trained model baked in,
#              nothing mounted. Built by ./run.sh serve-build; deployed in Phase 7.
#
# Python 3.10 is pinned to the exact patch release so every build starts from the
# same interpreter; requirements.txt pins every library.

FROM python:3.10.22-slim-bookworm AS pipeline

# make: every task goes through the Makefile. git: training records the commit SHA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Dependencies first, in their own layer: rebuilt only when requirements change.
# The cache mount keeps downloaded wheels between builds, so an interrupted or
# repeated build doesn't re-download everything.
RUN python -m venv /opt/venv
COPY requirements.txt requirements-dev.txt requirements-notebooks.txt requirements-serve.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install -r requirements-dev.txt

# The package itself, installed editable from /app so the mounted source is what runs
WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps -e .

# The mounted folder belongs to the host user; let git inside the container read it
RUN git config --system --add safe.directory /app

# Makefile targets call .venv/bin/...; point them at the image's venv instead
ENV BIN=/opt/venv/bin \
    MLFLOW_HOST=0.0.0.0

CMD ["make", "help"]


FROM pipeline AS notebooks
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install -r /requirements-notebooks.txt


# --- Serving image ---------------------------------------------------------------
# Built in two stages: dependencies are installed in serve-builder, and only the
# resulting venv is copied into a clean runtime, so no pip cache or build tooling ships.

FROM python:3.10.22-slim-bookworm AS serve-builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_INDEX_URL=https://pypi.org/simple
RUN python -m venv /opt/venv
COPY requirements-serve.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install -r /tmp/requirements-serve.txt
COPY pyproject.toml /src/
COPY src/ /src/src/
RUN /opt/venv/bin/pip install --no-deps --no-compile /src


FROM python:3.10.22-slim-bookworm AS serve

LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI)" \
      org.opencontainers.image.source="https://github.com/i-am-darshil/heart-disease-mlops"

# Unprivileged user: a compromised process can't modify the image or bind low ports
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MODELS_DIR=/app/models \
    PORT=8000 \
    LOG_LEVEL=info

COPY --from=serve-builder /opt/venv /opt/venv

# The trained model is part of the image: the image tag identifies exactly which model
# is served, and a container needs nothing mounted to start. Train before building.
COPY --chown=app:app models/model.joblib models/model_metadata.json /app/models/

WORKDIR /app
USER app
EXPOSE 8000

# Container-level health check (Docker / compose). Kubernetes uses its own probes.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/ready', timeout=2)"]

# One uvicorn worker per container: Prometheus counters live in process memory, so
# several workers would each report only their own share. Scale out with more
# containers/pods instead (Phase 7). `exec` makes uvicorn PID 1, so it receives
# SIGTERM directly and shuts down gracefully on `docker stop` / pod deletion.
CMD ["sh", "-c", "exec uvicorn heart.api.main:app --host 0.0.0.0 --port \"$PORT\" --no-access-log --proxy-headers --forwarded-allow-ips='*'"]
