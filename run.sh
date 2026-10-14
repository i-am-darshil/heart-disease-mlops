#!/usr/bin/env bash
# Run every project task inside Docker. The only requirement is Docker itself.
#
#   ./run.sh help        list commands
#   ./run.sh all         build the image, then lint + test + train (the full pipeline)
#   ./run.sh mlflow      browse experiments at http://127.0.0.1:5050
#
# The project folder is mounted into the container, so everything it produces
# (models/, mlflow.db, mlartifacts/, reports/figures/, notebooks) appears right here.

set -euo pipefail

IMAGE="${IMAGE:-heart-pipeline:latest}"
NOTEBOOK_IMAGE="${NOTEBOOK_IMAGE:-heart-notebooks:latest}"
SERVE_IMAGE="${SERVE_IMAGE:-heart-api:latest}"
API_PORT="${API_PORT:-8000}"
API_CONTAINER="${API_CONTAINER:-heart-api}"
MLFLOW_PORT="${MLFLOW_PORT:-5050}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-heart}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -t 1 ]; then
    BOLD=$'\033[1m' GREEN=$'\033[32m' RED=$'\033[31m' YELLOW=$'\033[33m' RESET=$'\033[0m'
else
    BOLD="" GREEN="" RED="" YELLOW="" RESET=""
fi

step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok() { printf '%s✔ %s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '%s! %s%s\n' "$YELLOW" "$*" "$RESET"; }
die() {
    printf '%s✘ %s%s\n' "$RED" "$*" "$RESET" >&2
    exit 1
}

# `docker` or `sudo docker`, whichever works for this user
DOCKER=(docker)
resolve_docker() {
    command -v docker >/dev/null 2>&1 || die "Docker is not installed. See docs/SETUP.md, step 1."
    if docker info >/dev/null 2>&1; then
        return
    fi
    if sudo -n docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
        warn "Using 'sudo docker'. To drop sudo: sudo usermod -aG docker \$USER, then log out and back in."
        return
    fi
    die "Docker is installed but not running. Start it with: sudo systemctl enable --now docker"
}

image_exists() { "${DOCKER[@]}" image inspect "${1:-$IMAGE}" >/dev/null 2>&1; }

# Run a command in the pipeline image with the project mounted at /app.
# Runs as the host user so files created in the project folder aren't owned by root.
# :z relabels the mount for SELinux (enforcing by default on RHEL-family hosts);
# it's ignored where SELinux isn't in use.
in_container() {
    local image="$IMAGE"
    if [ "${USE_NOTEBOOK_IMAGE:-0}" = 1 ]; then
        image="$NOTEBOOK_IMAGE"
        image_exists "$image" || build_notebooks
    else
        image_exists || build
    fi
    local tty=()
    [ -t 0 ] && [ -t 1 ] && tty=(-it)
    # ${arr[@]+"${arr[@]}"} expands an empty array safely under `set -u` on bash 3.2 (macOS)
    "${DOCKER[@]}" run --rm ${tty[@]+"${tty[@]}"} \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
        -v "$PROJECT_DIR:/app:z" \
        ${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"} \
        "$image" "$@"
}
EXTRA_DOCKER_ARGS=()

build() {
    step "Building image $IMAGE (first build downloads about 250 MB of packages; later builds are cached)"
    DOCKER_BUILDKIT=1 "${DOCKER[@]}" build --target pipeline -t "$IMAGE" "$PROJECT_DIR"
    ok "Image built: $IMAGE"
}

build_notebooks() {
    step "Building image $NOTEBOOK_IMAGE (pipeline image + Jupyter)"
    DOCKER_BUILDKIT=1 "${DOCKER[@]}" build --target notebooks -t "$NOTEBOOK_IMAGE" "$PROJECT_DIR"
    ok "Image built: $NOTEBOOK_IMAGE"
}

check() {
    step "Checking prerequisites"
    printf '  OS:      %s\n' "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
    printf '  Arch:    %s\n' "$(uname -m)"
    printf '  Docker:  %s\n' "$("${DOCKER[@]}" version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"
    printf '  Project: %s\n' "$PROJECT_DIR"
    if command -v getenforce >/dev/null 2>&1; then
        printf '  SELinux: %s\n' "$(getenforce)"
    fi
    local free_gb
    free_gb=$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')
    printf '  Free disk: %s GB\n' "$free_gb"
    [ "$free_gb" -ge 3 ] || warn "Less than 3 GB free; the image needs about 2 GB."
    ok "Docker is ready"
}

run_make() {
    step "make $*"
    in_container make "$@"
}

all() {
    check
    build
    run_make lint
    run_make test
    run_make train
    step "Done"
    ok "Lint, tests and training passed."
    printf '  Model:    %s\n' "models/model.joblib, models/metrics.json"
    printf '  Plots:    %s\n' "reports/figures/"
    printf '  MLflow:   run %s./run.sh mlflow%s and open http://127.0.0.1:%s\n' "$BOLD" "$RESET" "$MLFLOW_PORT"
}

mlflow_ui() {
    [ -f "$PROJECT_DIR/mlflow.db" ] || warn "No mlflow.db yet; run ./run.sh train first to have something to browse."
    step "MLflow UI on http://127.0.0.1:$MLFLOW_PORT (Ctrl+C to stop)"
    printf '  From your laptop: ssh -N -L %s:127.0.0.1:%s <user>@<vm-host>, then open the URL above\n' \
        "$MLFLOW_PORT" "$MLFLOW_PORT"
    # Published on the host's localhost only: the MLflow UI has no authentication
    EXTRA_DOCKER_ARGS=(-p "127.0.0.1:$MLFLOW_PORT:$MLFLOW_PORT" -e "MLFLOW_PORT=$MLFLOW_PORT")
    in_container make mlflow-ui
}

jupyter_lab() {
    # A short token you can type by hand. The port is published on localhost only
    # (reach it through an SSH tunnel), so it doesn't need to be a long random secret.
    local url="http://127.0.0.1:$JUPYTER_PORT/lab?token=$JUPYTER_TOKEN"
    step "Jupyter Lab (Ctrl+C to stop)"
    printf '  Open:              %s%s%s\n' "$BOLD" "$url" "$RESET"
    printf '  From your laptop:  ssh -N -L %s:127.0.0.1:%s <user>@<vm-host>, then open the link above\n' \
        "$JUPYTER_PORT" "$JUPYTER_PORT"
    EXTRA_DOCKER_ARGS=(-p "127.0.0.1:$JUPYTER_PORT:$JUPYTER_PORT")
    USE_NOTEBOOK_IMAGE=1 in_container jupyter lab --ip 0.0.0.0 --port "$JUPYTER_PORT" --no-browser \
        --ServerApp.root_dir=/app/notebooks --IdentityProvider.token="$JUPYTER_TOKEN"
}

serve_build() {
    [ -f "$PROJECT_DIR/models/model.joblib" ] || {
        warn "No trained model yet; training one first (the API image bakes it in)."
        run_make train
    }
    step "Building API image $SERVE_IMAGE (slim: FastAPI + scikit-learn, no MLflow/plots)"
    DOCKER_BUILDKIT=1 "${DOCKER[@]}" build --target serve -t "$SERVE_IMAGE" "$PROJECT_DIR"
    local size
    size=$("${DOCKER[@]}" image inspect "$SERVE_IMAGE" --format '{{.Size}}' | awk '{printf "%.0f MB", $1/1e6}')
    ok "API image built: $SERVE_IMAGE ($size)"
}

serve() {
    image_exists "$SERVE_IMAGE" || serve_build
    "${DOCKER[@]}" rm -f "$API_CONTAINER" >/dev/null 2>&1 || true
    local holder
    holder=$("${DOCKER[@]}" ps --format '{{.Names}} {{.Ports}}' | grep -E ":$API_PORT->" | cut -d' ' -f1 || true)
    if [ -n "$holder" ]; then
        die "Port $API_PORT is already used by container '$holder'. Pick another: API_PORT=8001 ./run.sh serve"
    fi
    step "Starting API container '$API_CONTAINER' on http://127.0.0.1:$API_PORT"
    "${DOCKER[@]}" run -d --name "$API_CONTAINER" \
        -p "127.0.0.1:$API_PORT:8000" \
        --restart unless-stopped \
        "$SERVE_IMAGE" >/dev/null
    printf '  Waiting for the model to load'
    local i
    for i in $(seq 1 40); do
        if curl -sf "http://127.0.0.1:$API_PORT/ready" >/dev/null 2>&1; then
            printf '\n'
            ok "API is ready"
            printf '  Docs (Swagger UI): http://127.0.0.1:%s/docs\n' "$API_PORT"
            printf '  Try it:            ./run.sh api-test\n'
            printf '  Logs:              ./run.sh api-logs\n'
            printf '  Stop:              ./run.sh api-stop\n'
            printf '  From your laptop:  ssh -N -L %s:127.0.0.1:%s <user>@<vm-host>\n' "$API_PORT" "$API_PORT"
            return
        fi
        printf '.'
        sleep 1
    done
    printf '\n'
    "${DOCKER[@]}" logs --tail 20 "$API_CONTAINER" >&2 || true
    die "API did not become ready in 40 s; container logs are above."
}

api_test() {
    local base="http://127.0.0.1:$API_PORT"
    command -v curl >/dev/null 2>&1 || die "curl is needed for api-test (sudo dnf -y install curl)."
    curl -sf "$base/health" >/dev/null 2>&1 || die "API isn't running on $base. Start it: ./run.sh serve"
    step "GET /health"
    curl -s "$base/health"
    printf '\n'
    step "GET /ready"
    curl -s "$base/ready"
    printf '\n'
    step "POST /predict  (examples/sample_request.json)"
    curl -s -X POST "$base/predict" -H 'Content-Type: application/json' \
        -d @"$PROJECT_DIR/examples/sample_request.json"
    printf '\n'
    step "POST /predict/batch  (examples/sample_batch.json)"
    printf '{"patients": %s}' "$(cat "$PROJECT_DIR/examples/sample_batch.json")" \
        | curl -s -X POST "$base/predict/batch" -H 'Content-Type: application/json' -d @-
    printf '\n'
    step "POST /predict with invalid input (expect HTTP 422)"
    curl -s -X POST "$base/predict" -H 'Content-Type: application/json' \
        -d '{"age": 500, "cp": 9}' -w '\n  -> HTTP %{http_code}\n'
    step "GET /metrics  (model metrics only)"
    curl -s "$base/metrics" | grep -E '^heart_(predictions_total|validation_errors_total)'
    ok "API answered every request"
}

api_logs() { "${DOCKER[@]}" logs -f --tail 50 "$API_CONTAINER"; }

api_stop() {
    if "${DOCKER[@]}" rm -f "$API_CONTAINER" >/dev/null 2>&1; then
        ok "API container stopped"
    else
        warn "No API container running"
    fi
}

reset_mlflow() {
    step "Deleting the local MLflow store (mlflow.db, mlartifacts/)"
    rm -rf "$PROJECT_DIR/mlflow.db" "$PROJECT_DIR/mlartifacts"
    ok "MLflow history cleared. The next ./run.sh train starts fresh at v1."
}

clean() {
    api_stop >/dev/null 2>&1 || true
    step "Removing images $IMAGE, $NOTEBOOK_IMAGE and $SERVE_IMAGE"
    for image in "$SERVE_IMAGE" "$NOTEBOOK_IMAGE" "$IMAGE"; do
        if image_exists "$image"; then
            "${DOCKER[@]}" image rm "$image"
        fi
    done
    ok "Removed. Outputs in the project folder are kept."
}

usage() {
    cat <<EOF
${BOLD}Usage:${RESET} ./run.sh <command>

${BOLD}Most common${RESET}
  all          Build the image, then lint + test + train  (start here)
  mlflow       MLflow UI on http://127.0.0.1:$MLFLOW_PORT

${BOLD}Pipeline steps${RESET}
  check        Check Docker is installed and running
  build        Build (or rebuild) the Docker image
  data         Download the UCI dataset (checksum-verified) and clean it
  eda          Regenerate the EDA figures in reports/figures/
  train        Train, select, quality-gate, save models/, log to MLflow
  predict      Predict for examples/sample_request.json (or: ./run.sh predict <file.json>)
  reproduce    Train twice from scratch and verify both models are identical
  test         Run the unit tests with coverage
  lint         Check code style
  notebook     Re-execute both notebooks (refreshes their outputs)

${BOLD}API (Phase 5)${RESET}
  serve-build  Build the slim API image with the trained model baked in
  serve        Start the API container on http://127.0.0.1:$API_PORT (docs at /docs)
  api-test     Send example requests to the running API
  api-logs     Follow the API's JSON request logs (Ctrl+C to stop following)
  api-stop     Stop the API container

${BOLD}Interactive${RESET}
  jupyter      Jupyter Lab on http://127.0.0.1:$JUPYTER_PORT  (builds a second image with Jupyter)
  shell        Open a bash shell inside the container

${BOLD}Housekeeping${RESET}
  reset-mlflow Delete local MLflow history (mlflow.db, mlartifacts/)
  clean        Remove the Docker images (keeps all outputs)

Ports can be changed: MLFLOW_PORT=5051 ./run.sh mlflow, API_PORT=8001 ./run.sh serve
Jupyter's login token is "heart"; change it with JUPYTER_TOKEN=... ./run.sh jupyter
EOF
}

main() {
    local cmd="${1:-help}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        help | -h | --help) usage; return ;;
    esac
    resolve_docker
    case "$cmd" in
        all) all ;;
        check) check ;;
        build) build ;;
        data | eda | train | test | lint | reproduce) run_make "$cmd" ;;
        predict)
            step "Predict${1:+ for $1}"
            in_container make predict ${1:+INPUT="$1"}
            ;;
        notebook)
            step "make notebook"
            USE_NOTEBOOK_IMAGE=1 in_container make notebook
            ;;
        mlflow | mlflow-ui) mlflow_ui ;;
        jupyter) jupyter_lab ;;
        serve-build) serve_build ;;
        serve) serve ;;
        api-test) api_test ;;
        api-logs) api_logs ;;
        api-stop) api_stop ;;
        shell) in_container bash ;;
        reset-mlflow) reset_mlflow ;;
        clean) clean ;;
        *)
            usage
            die "Unknown command: $cmd"
            ;;
    esac
}

main "$@"
