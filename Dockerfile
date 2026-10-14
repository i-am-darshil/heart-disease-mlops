# syntax=docker/dockerfile:1
#
# Two targets, both driven by ./run.sh (don't call docker directly):
#   pipeline   lint, test, data, EDA, train, MLflow UI
#   notebooks  pipeline + Jupyter, for ./run.sh jupyter / ./run.sh notebook
# The project folder is mounted at /app at run time, so outputs (models/, mlflow.db,
# figures) land on the host.
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
COPY requirements.txt requirements-dev.txt requirements-notebooks.txt ./
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
