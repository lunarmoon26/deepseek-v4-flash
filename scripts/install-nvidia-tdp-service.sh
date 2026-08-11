#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME=nvidia-tdp.service
UNIT_DEST="/etc/systemd/system/$UNIT_NAME"
HELPER_DEST=/usr/local/libexec/deepseek-v4-flash-set-nvidia-tdp
CONFIG_DIR=/etc/deepseek-v4-flash
CONFIG_DEST="$CONFIG_DIR/nvidia-tdp.conf"

if (( EUID != 0 )); then
  exec sudo -- "$0" "$@"
fi

for source in \
  "$ROOT/systemd/$UNIT_NAME" \
  "$ROOT/scripts/set-nvidia-tdp.sh" \
  "$ROOT/config/nvidia-tdp.conf.example"; do
  [[ -f "$source" ]] || {
    printf 'Missing project file: %s\n' "$source" >&2
    exit 1
  }
done

# Preserve a hand-written unit before replacing it with the project-managed
# one. The old Lambda-style helper is intentionally left untouched; the new
# unit invokes HELPER_DEST instead.
if [[ -f "$UNIT_DEST" ]] && ! cmp -s "$ROOT/systemd/$UNIT_NAME" "$UNIT_DEST"; then
  backup="$UNIT_DEST.pre-deepseek-$(date +%Y%m%d%H%M%S)"
  cp -a "$UNIT_DEST" "$backup"
  printf 'Backed up existing unit to %s\n' "$backup"
fi

install -D -m 0755 "$ROOT/scripts/set-nvidia-tdp.sh" "$HELPER_DEST"
install -D -m 0644 "$ROOT/systemd/$UNIT_NAME" "$UNIT_DEST"
install -d -m 0755 "$CONFIG_DIR"
if [[ ! -e "$CONFIG_DEST" ]]; then
  install -m 0644 "$ROOT/config/nvidia-tdp.conf.example" "$CONFIG_DEST"
  printf 'Created %s (edit GPU_POWER_LIMIT_WATTS there to change the cap).\n' \
    "$CONFIG_DEST"
fi

systemctl daemon-reload
systemctl enable "$UNIT_NAME"
systemctl restart "$UNIT_NAME"
systemctl --no-pager --full status "$UNIT_NAME"
/usr/bin/nvidia-smi --query-gpu=index,name,power.limit \
  --format=csv,noheader
