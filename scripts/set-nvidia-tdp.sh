#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_FILE=/etc/deepseek-v4-flash/nvidia-tdp.conf
readonly NVIDIA_SMI=/usr/bin/nvidia-smi

[[ -x "$NVIDIA_SMI" ]] || {
  printf 'nvidia-smi is unavailable at %s\n' "$NVIDIA_SMI" >&2
  exit 1
}
[[ -r "$CONFIG_FILE" ]] || {
  printf 'Missing power-limit configuration: %s\n' "$CONFIG_FILE" >&2
  exit 1
}

power_limit_watts="$(awk -F= '
  /^GPU_POWER_LIMIT_WATTS=/ {
    value=$2
    sub(/[[:space:]]*#.*/, "", value)
    gsub(/[[:space:]]/, "", value)
  }
  END { print value }
' "$CONFIG_FILE")"

[[ "$power_limit_watts" =~ ^[1-9][0-9]*$ ]] || {
  printf 'GPU_POWER_LIMIT_WATTS must be a positive integer; got %q\n' \
    "$power_limit_watts" >&2
  exit 1
}

gpus=()
for ((attempt = 1; attempt <= 30; attempt++)); do
  if gpu_output="$($NVIDIA_SMI --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)"; then
    gpus=()
    while IFS= read -r gpu; do
      [[ -n "$gpu" ]] && gpus+=("$gpu")
    done <<< "$gpu_output"
    (( ${#gpus[@]} > 0 )) && break
  fi
  sleep 2
done

(( ${#gpus[@]} > 0 )) || {
  printf 'No NVIDIA GPUs became available within 60 seconds.\n' >&2
  exit 1
}

for gpu in "${gpus[@]}"; do
  bounds="$($NVIDIA_SMI -i "$gpu" \
    --query-gpu=power.min_limit,power.max_limit \
    --format=csv,noheader,nounits)"
  IFS=',' read -r min_limit max_limit <<< "$bounds"
  min_limit="${min_limit//[[:space:]]/}"
  max_limit="${max_limit//[[:space:]]/}"

  awk -v requested="$power_limit_watts" -v min="$min_limit" -v max="$max_limit" '
    BEGIN {
      if (min !~ /^[0-9]+(\.[0-9]+)?$/ || max !~ /^[0-9]+(\.[0-9]+)?$/ ||
          requested < min || requested > max) {
        exit 1
      }
    }
  ' || {
    printf 'GPU %s rejects %s W; supported range is %s–%s W.\n' \
      "$gpu" "$power_limit_watts" "$min_limit" "$max_limit" >&2
    exit 1
  }

  printf 'Setting NVIDIA GPU %s to %s W.\n' "$gpu" "$power_limit_watts"
  "$NVIDIA_SMI" -i "$gpu" --power-limit="$power_limit_watts"

  actual="$($NVIDIA_SMI -i "$gpu" --query-gpu=power.limit \
    --format=csv,noheader,nounits)"
  actual="${actual//[[:space:]]/}"
  awk -v expected="$power_limit_watts" -v actual="$actual" '
    BEGIN { exit !(actual >= expected - 0.01 && actual <= expected + 0.01) }
  ' || {
    printf 'GPU %s reports %s W after applying %s W.\n' \
      "$gpu" "$actual" "$power_limit_watts" >&2
    exit 1
  }
done
