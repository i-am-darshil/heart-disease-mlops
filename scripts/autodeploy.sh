#!/usr/bin/env bash
# Continuous deployment by polling: a systemd timer on the VM checks GHCR every few
# minutes and, when CI has published a new API image, deploys it (helm upgrade + helm
# test, automatic rollback on failure). The VM pulls; GitHub never connects to it, and
# no token is stored anywhere.
#
#   scripts/autodeploy.sh on       install + start the timer
#   scripts/autodeploy.sh off      stop + remove it
#   scripts/autodeploy.sh status   is it on, when did it last run, recent deploy log

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT=heart-autodeploy
INTERVAL="${INTERVAL:-5min}"
ME="$(id -un)"

case "${1:-status}" in
    on)
        sudo tee /etc/systemd/system/$UNIT.service >/dev/null <<EOF
[Unit]
Description=Deploy the newest heart-disease API image from GHCR to k3s
After=network-online.target k3s.service docker.service
Wants=network-online.target

[Service]
Type=oneshot
User=$ME
Environment=HOME=$HOME
Environment=KUBECONFIG=$HOME/.kube/config
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/scripts/deploy.sh --if-changed
EOF
        sudo tee /etc/systemd/system/$UNIT.timer >/dev/null <<EOF
[Unit]
Description=Check for a new heart-disease API image every $INTERVAL

[Timer]
OnBootSec=2min
OnUnitActiveSec=$INTERVAL
# Spread checks a little so they don't all hit the registry at the same second
RandomizedDelaySec=30s
Persistent=true

[Install]
WantedBy=timers.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable --now $UNIT.timer
        printf '\033[32m✔ Auto-deploy is ON: checking GHCR every %s\033[0m\n' "$INTERVAL"
        printf '  Watch it: ./run.sh deploy-status   (or: journalctl -u %s -f)\n' "$UNIT"
        ;;
    off)
        sudo systemctl disable --now $UNIT.timer 2>/dev/null || true
        sudo rm -f /etc/systemd/system/$UNIT.service /etc/systemd/system/$UNIT.timer
        sudo systemctl daemon-reload
        printf '\033[32m✔ Auto-deploy is OFF\033[0m\n'
        ;;
    status)
        if systemctl is-enabled --quiet $UNIT.timer 2>/dev/null; then
            printf 'Auto-deploy: \033[32mON\033[0m\n'
            systemctl list-timers $UNIT.timer --no-pager | head -2
        else
            printf 'Auto-deploy: OFF (turn on: ./run.sh autodeploy-on)\n'
        fi
        printf '\nRecent deploy log:\n'
        journalctl -u $UNIT --no-pager -n 15 -o cat 2>/dev/null | grep "\[deploy\]" \
            || echo "  (no runs yet)"
        ;;
    *)
        echo "usage: $0 on|off|status" >&2
        exit 2
        ;;
esac
