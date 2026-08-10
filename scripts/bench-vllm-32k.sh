#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
TOKENIZER="${MODEL_PATH:-$ROOT/models/DeepSeek-V4-Flash-0731}"
INPUT_TOKENS="${INPUT_TOKENS:-30720}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_CONTEXT="${MAX_CONTEXT:-32768}"
RESULT_DIR="${RESULT_DIR:-$ROOT/benchmark-results}"
# A fresh default seed prevents a previous invocation's identical random
# prompts from turning this into a prefix-cache benchmark. Set BENCH_SEED to a
# fixed value when intentionally reproducing a run on a freshly started server.
BENCH_SEED="${BENCH_SEED:-$(date +%s)}"

if (( INPUT_TOKENS + OUTPUT_TOKENS > MAX_CONTEXT )); then
  printf '%s\n' \
    "INPUT_TOKENS + OUTPUT_TOKENS must not exceed MAX_CONTEXT ($MAX_CONTEXT)."
  exit 1
fi

[[ -x "$ROOT/.venv/bin/vllm" ]] || {
  printf '%s\n' "Missing $ROOT/.venv/bin/vllm"
  exit 1
}
[[ -f "$TOKENIZER/tokenizer_config.json" ]] || {
  printf '%s\n' "Tokenizer is unavailable: $TOKENIZER"
  exit 1
}

mkdir -p "$RESULT_DIR"

exec "$ROOT/.venv/bin/vllm" bench serve \
  --backend openai \
  --base-url "$BASE_URL" \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-mode deepseek_v4 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len "$INPUT_TOKENS" \
  --random-output-len "$OUTPUT_TOKENS" \
  --random-range-ratio 0 \
  --seed "$BENCH_SEED" \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups "$NUM_WARMUPS" \
  --request-rate inf \
  --max-concurrency "$MAX_CONCURRENCY" \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "vllm-32k-c${MAX_CONCURRENCY}.json" \
  --metadata \
    context_limit="$MAX_CONTEXT" \
    input_tokens="$INPUT_TOKENS" \
    output_tokens="$OUTPUT_TOKENS" \
    concurrency="$MAX_CONCURRENCY" \
    seed="$BENCH_SEED"
