#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

MOET_STORE_DIR="${MOET_STORE_DIR:-}"
MOET_STORE_RANK0_DIR="${MOET_STORE_RANK0_DIR:-}"

for command in cmp findmnt rsync; do
  command -v "$command" >/dev/null || {
    printf 'Missing required command: %s\n' "$command"
    exit 1
  }
done

[[ -d "$MOET_STORE_DIR" ]] || {
  printf '%s\n' "Shared Moet store is unavailable: $MOET_STORE_DIR"
  exit 1
}
[[ -n "$MOET_STORE_RANK0_DIR" ]] || {
  printf '%s\n' "Set MOET_STORE_RANK0_DIR in the ignored .env file first."
  exit 1
}
mkdir -p "$MOET_STORE_RANK0_DIR"

source_device="$(findmnt -n -o SOURCE -T "$MOET_STORE_DIR")"
destination_device="$(findmnt -n -o SOURCE -T "$MOET_STORE_RANK0_DIR")"
[[ "$source_device" != "$destination_device" ]] || {
  printf '%s\n' "Source and rank-0 destination are on the same filesystem: $source_device"
  exit 1
}

shopt -s nullglob
rank0_packs=("$MOET_STORE_DIR"/*.rank0of2.pack)
shopt -u nullglob
(( ${#rank0_packs[@]} == 1 )) || {
  printf '%s\n' "Expected exactly one completed rank0of2 pack in $MOET_STORE_DIR; found ${#rank0_packs[@]}."
  exit 1
}

rank0_pack="${rank0_packs[0]}"
rank0_stem="$(basename "$rank0_pack" .pack)"
files=(
  "$MOET_STORE_DIR/$rank0_stem.pack"
  "$MOET_STORE_DIR/$rank0_stem.json"
  "$MOET_STORE_DIR/$rank0_stem.lock"
)
for file in "${files[@]}"; do
  [[ -f "$file" ]] || {
    printf '%s\n' "Incomplete rank-0 store: missing $file"
    exit 1
  }
done

printf 'Copying TP rank 0 from %s (%s) to %s (%s)...\n' \
  "$MOET_STORE_DIR" "$source_device" \
  "$MOET_STORE_RANK0_DIR" "$destination_device"
rsync --archive --sparse --info=stats2 \
  "${files[@]}" "$MOET_STORE_RANK0_DIR/"

for source in "${files[@]}"; do
  destination="$MOET_STORE_RANK0_DIR/$(basename "$source")"
  cmp -s "$source" "$destination" || {
    printf '%s\n' "Verification failed: $destination differs from $source"
    exit 1
  }
done

printf '%s\n' "Rank-0 copy verified byte-for-byte. Restart serve-moet-0731.sh to activate it."
printf '%s\n' "Keep the shared-store rank-0 files: they are rollback copies and Docker mount targets."
