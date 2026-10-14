#!/usr/bin/env bash
# Render the Helm charts with several value combinations and validate every manifest
# against the Kubernetes API schema. Fails if any manifest is invalid, or if a
# combination renders nothing (a broken render must not count as "0 errors").
# Then unit-tests the Prometheus alert rules with promtool.
#
#   scripts/validate-chart.sh          (CI job helm-lint; needs helm + docker)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.36.0}"
KUBECONFORM=ghcr.io/yannh/kubeconform:v0.8.0
# Same Prometheus as the monitoring chart deploys, so promtool parses rules the same way
PROMETHEUS=$(awk '/^prometheus:/ {p=1} p && /repository:/ {r=$2} p && /tag:/ {print r ":" $2; exit}' \
    "$PROJECT_DIR/helm/monitoring/values.yaml")

# Each entry: chart|name|extra flags for that combination
combos=(
    "heart-api|defaults|--set image.tag=test"
    "heart-api|autoscaling|--set image.tag=test --set autoscaling.enabled=true"
    "heart-api|cluster-ip-only|--set image.tag=test --set ingress.enabled=false --set service.type=ClusterIP"
    "monitoring|defaults|"
    "monitoring|no-ingress|--set ingress.grafana=false --set ingress.prometheus=false --set prometheus.persistence.enabled=false"
    "monitoring|two-namespaces|--set scrapeNamespaces={heart,staging}"
)

for combo in "${combos[@]}"; do
    IFS='|' read -r chart name extra <<<"$combo"
    read -r -a flags <<<"$extra"
    rendered=$(helm template "$chart" "$PROJECT_DIR/helm/$chart" ${flags[@]+"${flags[@]}"})
    count=$(grep -c '^kind:' <<<"$rendered" || true)
    if [ "$count" -eq 0 ]; then
        echo "FAIL [$chart/$name]: chart rendered no resources" >&2
        exit 1
    fi
    printf '%-28s ' "[$chart/$name]"
    docker run --rm -i "$KUBECONFORM" -strict -summary \
        -kubernetes-version "$KUBERNETES_VERSION" - <<<"$rendered"
done

echo "Alert rules (promtool, $PROMETHEUS):"
docker run --rm -v "$PROJECT_DIR:/w:ro" -w /w --entrypoint promtool "$PROMETHEUS" \
    check rules helm/monitoring/files/alerts.yml
docker run --rm -v "$PROJECT_DIR:/w:ro" -w /w --entrypoint promtool "$PROMETHEUS" \
    test rules tests/monitoring/alerts_test.yml
