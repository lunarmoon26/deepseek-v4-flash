#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/vendor/vLLM-Moet"

command -v docker >/dev/null || {
  printf '%s\n' "Docker Engine is required. Install Docker Engine and NVIDIA Container Toolkit first."
  exit 1
}

exec docker build \
  --file "$SOURCE/Dockerfile.sm120-v024" \
  --tag vllm-moet-sm120:v024 \
  "$SOURCE"
