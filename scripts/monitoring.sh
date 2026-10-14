#!/usr/bin/env bash
# Prometheus + Grafana on the local k3s cluster (helm/monitoring chart).
#
#   scripts/monitoring.sh up        install or upgrade, wait until both are ready
#   scripts/monitoring.sh status    pods, scrape targets, firing alerts, URLs, Grafana login
#   scripts/monitoring.sh down      uninstall (keeps the API running)
#
# Used by ./run.sh monitoring / monitoring-status / monitoring-down.

set -euo pipefail

# k3s puts kubectl and helm in /usr/local/bin, which isn't on every user's PATH
export PATH="/usr/local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="${MONITORING_RELEASE:-monitoring}"
NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*"; }
die() {
    printf '\033[31m✘ %s\033[0m\n' "$*" >&2
    exit 1
}

for tool in kubectl helm; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found. Run ./run.sh k8s-setup first."
done
kubectl get nodes >/dev/null 2>&1 || die "Can't reach the cluster. Is k3s running? sudo systemctl status k3s"

node_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }

# Ask Prometheus something through the cluster API (works without the Ingress or a tunnel)
prom_query() {
    kubectl get --raw "/api/v1/namespaces/$NAMESPACE/services/$RELEASE-prometheus:http/proxy/prometheus/api/v1/$1"
}

up() {
    step "Installing Prometheus + Grafana (release '$RELEASE', namespace '$NAMESPACE')"
    printf '  First install downloads about 250 MB of images; later upgrades reuse them.\n'
    if kubectl get namespace "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Terminating; then
        printf '  Waiting for the previous uninstall to finish...\n'
        kubectl wait --for=delete "namespace/$NAMESPACE" --timeout=3m >/dev/null 2>&1 || true
    fi
    # Helm's output is kept quiet on success but shown if it fails
    local out
    if ! out=$(helm upgrade --install "$RELEASE" "$PROJECT_DIR/helm/monitoring" \
        --namespace "$NAMESPACE" --create-namespace \
        --wait --timeout "${MONITORING_TIMEOUT:-10m}" 2>&1); then
        printf '%s\n' "$out" >&2
        kubectl -n "$NAMESPACE" get pods 2>/dev/null || true
        die "Install failed (output above). Pods stuck in ImagePullBackOff mean the images couldn't be downloaded."
    fi
    ok "Prometheus and Grafana are running"
    if kubectl get deploy -n heart heart-api >/dev/null 2>&1; then
        printf '  Waiting for the first scrape of the API pods'
        local _
        for _ in $(seq 1 20); do
            prom_query 'query?query=up%7Bjob%3D%22heart-api%22%7D' 2>/dev/null | grep -q '"value"' && break
            printf '.'
            sleep 3
        done
        printf '\n'
    else
        warn "The API isn't deployed yet, so there is nothing to scrape: ./run.sh deploy"
    fi
    status
}

status() {
    step "Pods in namespace '$NAMESPACE'"
    kubectl -n "$NAMESPACE" get pods,svc,ingress,pvc 2>/dev/null || die "Not installed: ./run.sh monitoring"

    step "What Prometheus scrapes"
    prom_query targets 2>/dev/null | python3 -c '
import json, sys
targets = json.load(sys.stdin)["data"]["activeTargets"]
for t in targets:
    labels = t["labels"]
    print("  {:<8} {:<11} {:<36} {}".format(t["health"], labels.get("job", ""), labels.get("pod", t["scrapeUrl"]), t.get("lastError", "")))
if not any(t["labels"].get("job") == "heart-api" for t in targets):
    print("  (no heart-api pods found yet: ./run.sh deploy, then wait 15 s)")
' || warn "Prometheus isn't answering yet"

    step "Alerts firing"
    prom_query alerts 2>/dev/null | python3 -c '
import json, sys
alerts = [a for a in json.load(sys.stdin)["data"]["alerts"] if a["state"] == "firing"]
for a in alerts:
    print("  [{}] {}: {}".format(a["labels"].get("severity"), a["labels"]["alertname"], a["annotations"].get("summary", "")))
print("  none" if not alerts else "")
' || true

    local ip password
    ip=$(node_ip)
    password=$(kubectl -n "$NAMESPACE" get secret "$RELEASE-grafana" -o jsonpath='{.data.admin-password}' 2>/dev/null | base64 -d || true)
    step "Open"
    printf '  Grafana dashboard:  http://%s/grafana   (view without login; to edit: admin / %s)\n' "${ip:-<vm-ip>}" "${password:-?}"
    printf '  Prometheus:         http://%s/prometheus   (alerts: /prometheus/alerts, targets: /prometheus/targets)\n' "${ip:-<vm-ip>}"
    printf '  Generate traffic:   ./run.sh load            (or: ./run.sh load drift | ./run.sh load errors)\n'
}

down() {
    step "Uninstalling monitoring (the API keeps running)"
    helm uninstall "$RELEASE" -n "$NAMESPACE" --wait 2>/dev/null || warn "Not installed"
    # Deleting the namespace also removes the stored metrics (the PersistentVolumeClaim)
    kubectl delete namespace "$NAMESPACE" --ignore-not-found --timeout=3m >/dev/null
    ok "Monitoring removed"
}

case "${1:-status}" in
    up) up ;;
    status) status ;;
    down) down ;;
    *)
        echo "usage: $0 up|status|down" >&2
        exit 2
        ;;
esac
