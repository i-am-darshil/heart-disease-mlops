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

# The installer puts k3s, kubectl and helm in /usr/local/bin. That directory isn't on
# every user's PATH, and sudo on RHEL never searches it (secure_path), so call k3s by
# its full path and add the directory to this script's PATH.
ORIGINAL_PATH="$PATH"
export PATH="/usr/local/bin:$PATH"
K3S=/usr/local/bin/k3s

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
if [ -x "$K3S" ] && "${SUDO[@]}" systemctl is-active --quiet k3s; then
    ok "already installed and running: $("$K3S" --version | head -1)"
else
    # --write-kubeconfig-mode 644: let non-root users read the cluster config.
    # metrics-server stays enabled: `kubectl top` and the autoscaler (HPA) need it.
    curl -sfL https://get.k3s.io | "${SUDO[@]}" env \
        INSTALL_K3S_VERSION="$K3S_VERSION" \
        INSTALL_K3S_EXEC="server --write-kubeconfig-mode=644" \
        sh -
    [ -x "$K3S" ] || die "The k3s installer finished but $K3S is missing. See its output above."
    # On a re-run the installer says "No change detected so skipping service start",
    # which leaves a stopped k3s stopped. Start it explicitly.
    "${SUDO[@]}" systemctl enable --now k3s
    ok "installed: $("$K3S" --version | head -1)"
fi

step "Waiting for the node to be Ready (up to 3 minutes)"
node_ready() { "${SUDO[@]}" "$K3S" kubectl get nodes 2>/dev/null | grep -q " Ready"; }
for _ in $(seq 1 90); do
    node_ready && break
    sleep 2
done
if ! node_ready; then
    warn "The node isn't Ready. Diagnostics:"
    "${SUDO[@]}" systemctl status k3s --no-pager -l 2>&1 | head -15 || true
    printf '\n  Last k3s log lines:\n'
    "${SUDO[@]}" journalctl -u k3s --no-pager -n 25 -o cat 2>&1 | sed 's/^/    /' || true
    printf '\n  Memory: %s MB free of %s MB\n' \
        "$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)" "$mem_mb"
    die "The k3s node didn't become Ready (diagnostics above). Full log: sudo journalctl -u k3s -n 200"
fi
ok "node is Ready"

step "kubectl for $ME"
mkdir -p "$HOME/.kube"
"${SUDO[@]}" cat /etc/rancher/k3s/k3s.yaml > "$HOME/.kube/config"
chmod 600 "$HOME/.kube/config"
# k3s installs kubectl as a symlink to itself; make sure it exists
[ -e /usr/local/bin/kubectl ] || "${SUDO[@]}" ln -sf "$K3S" /usr/local/bin/kubectl
ok "kubectl works: $(kubectl version --client 2>/dev/null | head -1)"

step "Helm $HELM_VERSION"
if [ -x /usr/local/bin/helm ] || command -v helm >/dev/null 2>&1; then
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
# Our own PATH has /usr/local/bin (added at the top); check what the user started with
case ":$ORIGINAL_PATH:" in
    *:/usr/local/bin:*) ;;
    *)
        warn "/usr/local/bin isn't on your PATH, so typing kubectl or helm yourself won't work."
        printf '  Fix it once:  echo '"'"'export PATH=/usr/local/bin:$PATH'"'"' >> ~/.bashrc && source ~/.bashrc\n'
        printf '  (./run.sh commands work either way.)\n'
        ;;
esac
cat <<EOF
  ./run.sh deploy            deploy the API image CI published (latest)
  ./run.sh k8s-status        pods, services, ingress
  ./run.sh autodeploy-on     deploy new CI images automatically (every 5 minutes)
EOF
