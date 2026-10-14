# Single entry point for every task. Run `make help` to list targets.
# Each target is exactly what CI runs, so "works on my machine" == "works in CI".

PYTHON  ?= python3.10
VENV    := .venv
# The Docker image sets BIN=/opt/venv/bin (its venv lives outside the mounted project)
BIN     ?= $(VENV)/bin
MLFLOW_PORT ?= 5050
# The Docker image sets 0.0.0.0 so the UI is reachable through the published port;
# ./run.sh still publishes that port on the host's localhost only
MLFLOW_HOST ?= 127.0.0.1

export MLFLOW_DISABLE_AGENT_HINT := 1

.DEFAULT_GOAL := help

.PHONY: help setup data eda notebook train reproduce predict serve mlflow-ui lint format test clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install all dependencies + this package (editable)
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 10), f"need Python 3.10, got {sys.version.split()[0]} (set PYTHON=/path/to/python3.10)"'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-notebooks.txt
	$(BIN)/pip install -e .

data:  ## Download the raw UCI file (checksum-verified) and write the cleaned CSV
	$(BIN)/python scripts/download_data.py
	$(BIN)/python -m heart.data

eda:  ## Regenerate EDA figures into reports/figures/
	$(BIN)/python -m heart.eda

notebook:  ## Re-execute the notebooks in place (refreshes their outputs)
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelling.ipynb

train:  ## Tune, select, quality-gate, save models/ + figures, log to MLflow and register
	$(BIN)/python -m heart.train

reproduce:  ## Train twice from scratch and verify identical models (byte-for-byte)
	$(BIN)/python -m heart.reproduce

predict:  ## Predict for examples/sample_request.json (INPUT=path to use another file)
	$(BIN)/python -m heart.predict $(or $(INPUT),examples/sample_request.json)

serve:  ## Run the API locally with auto-reload (http://127.0.0.1:8000/docs)
	$(BIN)/uvicorn heart.api.main:app --host 127.0.0.1 --port 8000 --reload

mlflow-ui:  ## Open the MLflow UI for the local tracking store (http://127.0.0.1:5050)
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

lint:  ## Check code style and common bugs (fails on any issue)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-fix style issues
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## Run unit tests with coverage
	$(BIN)/pytest --cov=heart --cov-report=term-missing

clean:  ## Remove caches and build outputs (keeps venv and data)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
