#!/usr/bin/env bash
# One-time setup for a RHEL-family VM (RHEL, Rocky, AlmaLinux, CentOS Stream 8/9).
#
# Installs git and Docker if they're missing, starts Docker, lets the current user run
# it, and clones (or updates) the project. Safe to run more than once.
#
#   curl -fsSL https://raw.githubusercontent.com/i-am-darshil/heart-disease-mlops/main/scripts/vm-bootstrap.sh | bash
#
# or, from an existing clone:  bash scripts/vm-bootstrap.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/i-am-darshil/heart-disease-mlops.git}"
TARGET_DIR="${TARGET_DIR:-$HOME/heart-disease-mlops}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok() { printf '\033[32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*"; }
die() {
    printf '\033[31m✘ %s\033[0m\n' "$*" >&2
    exit 1
}

# $USER isn't set in every environment (cron, some sudo/ssh setups); ask the system instead
ME="$(id -un)"

SUDO=()
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "This script needs root or sudo to install packages."
    SUDO=(sudo)
fi

if command -v dnf >/dev/null 2>&1; then
    PKG=dnf
elif command -v yum >/dev/null 2>&1; then
    PKG=yum
else
    die "Neither dnf nor yum found. This script is for RHEL-family systems."
fi

step "System"
. /etc/os-release 2>/dev/null && printf '  %s (%s)\n' "${PRETTY_NAME:-unknown}" "$(uname -m)"

step "git"
if command -v git >/dev/null 2>&1; then
    ok "already installed: $(git --version)"
else
    "${SUDO[@]}" "$PKG" -y install git
    ok "installed: $(git --version)"
fi

step "Docker"
if command -v docker >/dev/null 2>&1; then
    ok "already installed: $(docker --version)"
else
    # Docker CE from Docker's own repo (the RHEL repos ship podman instead).
    # RHEL proper uses the "rhel" repo; Rocky, Alma and CentOS Stream use "centos".
    if [ "${ID:-}" = "rhel" ]; then
        DOCKER_REPO=https://download.docker.com/linux/rhel/docker-ce.repo
    else
        DOCKER_REPO=https://download.docker.com/linux/centos/docker-ce.repo
    fi
    "${SUDO[@]}" "$PKG" -y install dnf-plugins-core 2>/dev/null || "${SUDO[@]}" "$PKG" -y install yum-utils
    "${SUDO[@]}" "$PKG" config-manager --add-repo "$DOCKER_REPO" 2>/dev/null \
        || "${SUDO[@]}" yum-config-manager --add-repo "$DOCKER_REPO"
    # podman-docker / buildah conflict with docker-ce on some images; remove if present
    "${SUDO[@]}" "$PKG" -y remove podman-docker buildah 2>/dev/null || true
    "${SUDO[@]}" "$PKG" -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    ok "installed: $(docker --version)"
fi

if ! "${SUDO[@]}" systemctl is-active --quiet docker; then
    "${SUDO[@]}" systemctl enable --now docker
fi
ok "Docker service is running"

NEEDS_RELOGIN=0
if [ "$(id -u)" -ne 0 ] && ! id -nG "$ME" | tr ' ' '\n' | grep -qx docker; then
    "${SUDO[@]}" usermod -aG docker "$ME"
    NEEDS_RELOGIN=1
    warn "Added $ME to the 'docker' group (takes effect after you log out and back in)."
fi

step "Project"
if [ -d "$TARGET_DIR/.git" ]; then
    git -C "$TARGET_DIR" pull --ff-only
    ok "updated $TARGET_DIR"
else
    git clone "$REPO_URL" "$TARGET_DIR"
    ok "cloned into $TARGET_DIR"
fi
chmod +x "$TARGET_DIR/run.sh"

step "Next steps"
if [ "$NEEDS_RELOGIN" -eq 1 ]; then
    cat <<EOF
  1. Log out and log back in (so you can use Docker without sudo)
  2. cd $TARGET_DIR
  3. ./run.sh all

  (Or skip the re-login: ./run.sh falls back to 'sudo docker' automatically.)
EOF
else
    cat <<EOF
  cd $TARGET_DIR
  ./run.sh all
EOF
fi
