#!/usr/bin/env bash
# Deploy the heart-disease API to the local k3s cluster with Helm, verify it, and roll
# back automatically if verification fails.
#
#   scripts/deploy.sh                 newest image CI published (resolves :latest to its commit tag)
#   scripts/deploy.sh 759f3d01aefb    a specific commit tag from GHCR
#   scripts/deploy.sh --local         the image built on this machine (./run.sh serve-build)
#   scripts/deploy.sh --if-changed    only deploy if CI published something new (used by the timer)
#
# Used by ./run.sh deploy and by the auto-deploy timer, so both behave identically.

set -euo pipefail

# k3s puts kubectl and helm in /usr/local/bin, which isn't on every user's PATH
export PATH="/usr/local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY="${REPOSITORY:-ghcr.io/i-am-darshil/heart-disease-api}"
LOCAL_IMAGE="${LOCAL_IMAGE:-heart-api:latest}"
RELEASE="${RELEASE:-heart-api}"
NAMESPACE="${NAMESPACE:-heart}"
CHART="$PROJECT_DIR/helm/heart-api"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

log() { printf '%s [deploy] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() {
    log "ERROR: $*" >&2
    exit 1
}

for tool in kubectl helm; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found. Run ./run.sh k8s-setup first."
done
kubectl get nodes >/dev/null 2>&1 || die "Can't reach the cluster (KUBECONFIG=$KUBECONFIG). Is k3s running? sudo systemctl status k3s"

current_tag() {
    helm get values "$RELEASE" -n "$NAMESPACE" -o json 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('image',{}).get('tag',''))" 2>/dev/null || true
}

# The 12-char commit tag that :latest currently points to, read from the image's own
# metadata (no registry API or token needed: the image is public). `docker pull` only
# downloads layers that changed, so polling this is cheap when nothing is new.
resolve_latest() {
    $DOCKER pull -q "$REPOSITORY:latest" >/dev/null || die "Couldn't pull $REPOSITORY:latest"
    $DOCKER run --rm --entrypoint python "$REPOSITORY:latest" -c \
        "import json; print(json.load(open('/app/models/model_metadata.json'))['model_version'])"
}

# docker or sudo docker
DOCKER=docker
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

MODE=registry
TAG=""
IF_CHANGED=0
for arg in "$@"; do
    case "$arg" in
        --local) MODE=local ;;
        --if-changed) IF_CHANGED=1 ;;
        -*) die "Unknown option: $arg" ;;
        *) TAG="$arg" ;;
    esac
done

if [ "$MODE" = local ]; then
    $DOCKER image inspect "$LOCAL_IMAGE" >/dev/null 2>&1 || die "No local image $LOCAL_IMAGE. Build it: ./run.sh serve-build"
    model_version=$($DOCKER run --rm --entrypoint python "$LOCAL_IMAGE" -c \
        "import json; print(json.load(open('/app/models/model_metadata.json'))['model_version'])")
    TAG="local-$model_version-$(date +%Y%m%d%H%M%S)"
    log "Importing local image $LOCAL_IMAGE into k3s as $REPOSITORY:$TAG"
    $DOCKER tag "$LOCAL_IMAGE" "$REPOSITORY:$TAG"
    # k3s runs its own containerd, separate from Docker: hand it the image directly
    $DOCKER save "$REPOSITORY:$TAG" | sudo /usr/local/bin/k3s ctr -n k8s.io images import - >/dev/null
elif [ -z "$TAG" ]; then
    TAG=$(resolve_latest)
    log "latest -> $TAG"
fi

PREVIOUS=$(current_tag)
if [ "$IF_CHANGED" = 1 ] && [ "$TAG" = "$PREVIOUS" ]; then
    log "Already running $TAG; nothing to do"
    exit 0
fi

log "Deploying $REPOSITORY:$TAG (currently: ${PREVIOUS:-nothing})"
# --wait blocks until every new pod passes its readiness probe (rolling update). Rollback
# on failure is done explicitly (not --rollback-on-failure) so it also covers helm test.
# --description shows up in `helm history`, so every revision says which commit it ran.
helm upgrade --install "$RELEASE" "$CHART" \
    --namespace "$NAMESPACE" --create-namespace \
    --set image.repository="$REPOSITORY" \
    --set image.tag="$TAG" \
    --wait --timeout "${DEPLOY_TIMEOUT:-5m}" \
    --description "deploy $TAG" >/dev/null || {
    log "helm upgrade failed"
    if [ -n "$PREVIOUS" ]; then
        log "Rolling back to the previous revision ($PREVIOUS)"
        helm rollback "$RELEASE" -n "$NAMESPACE" --wait --timeout 5m >/dev/null
    fi
    die "Deploy of $TAG failed$([ -n "$PREVIOUS" ] && echo "; rolled back to $PREVIOUS")"
}

log "Rollout done; running in-cluster checks (helm test)"
if ! helm test "$RELEASE" -n "$NAMESPACE" --timeout 2m --logs 2>&1 | grep -E "^OK " ; then
    log "helm test FAILED"
    if [ -n "$PREVIOUS" ]; then
        log "Rolling back to $PREVIOUS"
        helm rollback "$RELEASE" -n "$NAMESPACE" --wait --timeout 5m >/dev/null
        die "Deploy of $TAG failed verification; rolled back to $PREVIOUS"
    fi
    die "Deploy of $TAG failed verification (no previous version to roll back to)"
fi

revision=$(helm status "$RELEASE" -n "$NAMESPACE" -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])")
log "SUCCESS: $TAG is live (helm revision $revision)"
