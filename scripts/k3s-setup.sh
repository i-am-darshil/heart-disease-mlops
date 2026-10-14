#!/usr/bin/env bash
# One-time Kubernetes setup on a RHEL-family VM: installs k3s (a complete, lightweight
# Kubernetes distribution) and Helm, and configures kubectl for the current user.
#
#   bash scripts/k3s-setup.sh          (or ./run.sh k8s-setup)
#
# Safe to re-run: anything already installed is kept. k3s brings its own container
# runtime (containerd), the Traefik ingress controller and a LoadBalancer (ServiceLB),
# so nothing else is needed. Docker stays installed and is still used by ./run.sh.

set -euo pipefail

# Pinned for reproducibility: the k3s "stable" channel at the time of writing
K3S_VERSION="${K3S_VERSION:-v1.36.5+k3s1}"
HELM_VERSION="${HELM_VERSION:-v4.3.0}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*"; }
die() {
    printf '\033[31m✘ %s\033[0m\n' "$*" >&2
    exit 1
}

ME="$(id -un)"
SUDO=()
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "This script needs root or sudo."
    SUDO=(sudo)
fi
command -v curl >/dev/null 2>&1 || "${SUDO[@]}" dnf -y install curl

step "System"
. /etc/os-release 2>/dev/null && printf '  %s (%s)\n' "${PRETTY_NAME:-unknown}" "$(uname -m)"
mem_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
printf '  Memory: %s MB\n' "$mem_mb"
[ "$mem_mb" -ge 1800 ] || warn "k3s + the API want about 2 GB RAM; this VM has ${mem_mb} MB."
if command -v getenforce >/dev/null 2>&1; then
    printf '  SELinux: %s (the k3s installer adds its SELinux policy package)\n' "$(getenforce)"
fi

step "Firewall"
# k3s's documented RHEL requirement: let pods (10.42/16) and services (10.43/16) talk to
# each other, and open the ports the API is reached on. Only touches an active firewalld.
if "${SUDO[@]}" systemctl is-active --quiet firewalld 2>/dev/null; then
    "${SUDO[@]}" firewall-cmd --permanent --zone=trusted --add-source=10.42.0.0/16 >/dev/null # pods
    "${SUDO[@]}" firewall-cmd --permanent --zone=trusted --add-source=10.43.0.0/16 >/dev/null # services
    for port in 6443/tcp 80/tcp 443/tcp 8080/tcp; do
        "${SUDO[@]}" firewall-cmd --permanent --add-port="$port" >/dev/null
    done
    "${SUDO[@]}" firewall-cmd --reload >/dev/null
    ok "firewalld: pod/service networks trusted; 80 (Ingress), 8080 (LoadBalancer), 6443 (API) open"
else
    ok "firewalld not active; nothing to change"
fi

step "k3s $K3S_VERSION"
if command -v k3s >/dev/null 2>&1 && "${SUDO[@]}" systemctl is-active --quiet k3s; then
    ok "already installed and running: $(k3s --version | head -1)"
else
    # --write-kubeconfig-mode 644: let non-root users read the cluster config.
    # metrics-server stays enabled: `kubectl top` and the autoscaler (HPA) need it.
    curl -sfL https://get.k3s.io | "${SUDO[@]}" env \
        INSTALL_K3S_VERSION="$K3S_VERSION" \
        INSTALL_K3S_EXEC="server --write-kubeconfig-mode=644" \
        sh -
    ok "installed: $(k3s --version | head -1)"
fi

step "Waiting for the node to be Ready"
for _ in $(seq 1 90); do
    if "${SUDO[@]}" k3s kubectl get nodes 2>/dev/null | grep -q " Ready"; then
        break
    fi
    sleep 2
done
"${SUDO[@]}" k3s kubectl get nodes 2>/dev/null | grep -q " Ready" || die "The k3s node didn't become Ready. Check: sudo journalctl -u k3s -n 50"
ok "node is Ready"

step "kubectl for $ME"
mkdir -p "$HOME/.kube"
"${SUDO[@]}" cat /etc/rancher/k3s/k3s.yaml > "$HOME/.kube/config"
chmod 600 "$HOME/.kube/config"
# k3s installs kubectl as a symlink to itself; make sure it's on PATH
if ! command -v kubectl >/dev/null 2>&1; then
    "${SUDO[@]}" ln -sf "$(command -v k3s)" /usr/local/bin/kubectl
fi
ok "kubectl works: $(kubectl version --client 2>/dev/null | head -1)"

step "Helm $HELM_VERSION"
if command -v helm >/dev/null 2>&1; then
    ok "already installed: $(helm version --short)"
else
    arch=$(uname -m)
    case "$arch" in
        x86_64) arch=amd64 ;;
        aarch64) arch=arm64 ;;
    esac
    tmp=$(mktemp -d)
    curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${arch}.tar.gz" | tar -xz -C "$tmp"
    "${SUDO[@]}" install -m 0755 "$tmp/linux-${arch}/helm" /usr/local/bin/helm
    rm -rf "$tmp"
    ok "installed: $(helm version --short)"
fi

step "Waiting for Traefik (the Ingress controller)"
for _ in $(seq 1 120); do
    if [ "$(kubectl -n kube-system get deploy traefik -o jsonpath='{.status.readyReplicas}' 2>/dev/null)" = "1" ]; then
        break
    fi
    sleep 3
done
kubectl -n kube-system get deploy traefik >/dev/null 2>&1 && ok "Traefik is running" \
    || warn "Traefik isn't ready yet; it usually finishes within a minute. Check: kubectl -n kube-system get pods"

step "Next"
cat <<EOF
  ./run.sh deploy            deploy the API image CI published (latest)
  ./run.sh k8s-status        pods, services, ingress
  ./run.sh autodeploy-on     deploy new CI images automatically (every 5 minutes)
EOF
