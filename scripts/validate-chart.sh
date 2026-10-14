#!/usr/bin/env bash
# Render the Helm chart with several value combinations and validate every manifest
# against the Kubernetes API schema. Fails if any manifest is invalid, or if a
# combination renders nothing (a broken render must not count as "0 errors").
#
#   scripts/validate-chart.sh          (CI job helm-lint; needs helm + docker)

set -euo pipefail

CHART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helm/heart-api"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.36.0}"
KUBECONFORM=ghcr.io/yannh/kubeconform:v0.8.0

# Each entry: a name, then the extra flags for that combination
combos=(
    "defaults|"
    "autoscaling|--set autoscaling.enabled=true"
    "cluster-ip-only|--set ingress.enabled=false --set service.type=ClusterIP"
)

for combo in "${combos[@]}"; do
    name="${combo%%|*}"
    read -r -a flags <<<"${combo#*|}"
    rendered=$(helm template heart-api "$CHART" --set image.tag=test ${flags[@]+"${flags[@]}"})
    count=$(grep -c '^kind:' <<<"$rendered" || true)
    if [ "$count" -eq 0 ]; then
        echo "FAIL [$name]: chart rendered no resources" >&2
        exit 1
    fi
    printf '%-16s ' "[$name]"
    docker run --rm -i "$KUBECONFORM" -strict -summary \
        -kubernetes-version "$KUBERNETES_VERSION" - <<<"$rendered"
done
