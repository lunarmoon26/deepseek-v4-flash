#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export MODEL_PATH="${MODEL_PATH:-$ROOT/models/DeepSeek-V4-Flash-0731}"

mkdir -p "$MODEL_PATH"
exec uv run hf download "$MODEL_ID" --local-dir "$MODEL_PATH" --max-workers 8
