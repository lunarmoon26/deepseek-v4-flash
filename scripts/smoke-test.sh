#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-8000}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash}"

curl --fail --silent --show-error "http://$HOST:$PORT/v1/models"
printf '\n'
curl --fail --silent --show-error "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 19? Return only the integer.\"}],\"temperature\":0}"
printf '\n'
