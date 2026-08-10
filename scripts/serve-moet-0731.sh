#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env"
fi

export MODEL_PATH="${MODEL_PATH:-$ROOT/models/DeepSeek-V4-Flash-0731}"
export MOET_PLANES_CACHE="${MOET_PLANES_CACHE:-$ROOT/cache/moet-planes-0731}"
# Keep the FP4 expert recovery store on NVMe. Without this setting vLLM-Moet
# uses a pinned/pageable host-RAM store that is larger than this machine's RAM.
export MOET_STORE_DIR="${MOET_STORE_DIR:-$MOET_PLANES_CACHE/packs-ds4-tp2}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
export VLLM_PORT="${SERVER_PORT:-8000}"
# Keep enough physical RAM for the desktop and GPU driver on this 44 GiB-RAM
# host. Docker's memory-swap limit is total RAM + swap available to the
# container, not additional swap.
export MOET_CONTAINER_MEMORY="${MOET_CONTAINER_MEMORY:-34g}"
export MOET_CONTAINER_MEMORY_RESERVATION="${MOET_CONTAINER_MEMORY_RESERVATION:-30g}"
# Docker's --memory-swap is the total RAM + swap limit. Loading this 155 GiB
# checkpoint needs a generous swap budget even though its resident RAM remains
# capped, so the desktop cannot be starved of physical memory.
export MOET_CONTAINER_MEMORY_SWAP="${MOET_CONTAINER_MEMORY_SWAP:-200g}"
# Keep the CUDA caching allocator from fragmenting VRAM before graph capture.
# This matches the upstream RTX PRO 6000 TP2 recipe.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

command -v docker >/dev/null || {
  printf '%s\n' "Docker Engine is required. Run ./scripts/build-moet.sh after installing Docker Engine and NVIDIA Container Toolkit."
  exit 1
}
[[ -f "$MODEL_PATH/config.json" ]] || {
  printf '%s\n' "Official checkpoint is missing. Run ./scripts/download-official-0731.sh first."
  exit 1
}
[[ -d "$MOET_PLANES_CACHE" && -w "$MOET_PLANES_CACHE" ]] || {
  printf '%s\n' "Moet cache path is unavailable: $MOET_PLANES_CACHE"
  printf '%s\n' "Run sudo env ERASE_DEEPSEEK_NVME=YES ./scripts/setup-nvme.sh first."
  exit 1
}
mkdir -p "$MOET_STORE_DIR"
[[ -d "$MOET_STORE_DIR" && -w "$MOET_STORE_DIR" ]] || {
  printf '%s\n' "Moet NVMe store path is unavailable: $MOET_STORE_DIR"
  exit 1
}

host_virtual_kib=0
while read -r key value _; do
  case "$key" in
    MemTotal:|SwapTotal:) host_virtual_kib=$((host_virtual_kib + value)) ;;
  esac
done < /proc/meminfo
minimum_virtual_kib=$((220 * 1024 * 1024))
if (( host_virtual_kib < minimum_virtual_kib )); then
  printf '%s\n' "At least 220 GiB combined RAM+swap is required for the first Moet conversion."
  printf '%s\n' "This host currently has $((host_virtual_kib / 1024 / 1024)) GiB. Add RAM or NVMe-backed swap before retrying."
  exit 1
fi

docker_remove_arg=(--rm)
if [[ "${MOET_KEEP_CONTAINER:-0}" == "1" ]]; then
  # Retain an exited container for `docker inspect` diagnostics. Default is
  # still --rm so normal launches leave no stopped containers behind.
  docker_remove_arg=()
fi

# Sparse-MLA autotuning creates one decode and one prefill request, so keep at
# least two request-state slots even when serving one user request at a time.
exec docker run "${docker_remove_arg[@]}" \
  --name deepseek-v4-flash-moet \
  --runtime=nvidia \
  --network host \
  --ipc host \
  --shm-size 64g \
  --memory "$MOET_CONTAINER_MEMORY" \
  --memory-reservation "$MOET_CONTAINER_MEMORY_RESERVATION" \
  --memory-swap "$MOET_CONTAINER_MEMORY_SWAP" \
  -v "$MODEL_PATH:/model:ro" \
  -v "$MOET_PLANES_CACHE:/planes" \
  -v "$MOET_STORE_DIR:/packs" \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_PLANES_CACHE=/planes \
  -e VLLM_MOE_W2_STORE_DIR=/packs \
  -e VLLM_MOE_W2_DELTA_GB=24 \
  -e VLLM_MOE_W2_GATE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=0 \
  vllm-moet-sm120:v024 \
  --model /model \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$VLLM_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  --tokenizer-mode deepseek_v4 \
  --no-scheduler-reserve-full-isl \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
