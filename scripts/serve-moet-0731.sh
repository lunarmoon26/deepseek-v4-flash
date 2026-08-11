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
# Optional TP=2 rank-0 override. vLLM-Moet still sees one /packs directory;
# Docker overlays rank 0's individual files from this second filesystem.
export MOET_STORE_RANK0_DIR="${MOET_STORE_RANK0_DIR:-}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
export VLLM_PORT="${SERVER_PORT:-8000}"
# The context limit includes both prompt and generated tokens. With the current
# 24 GiB/rank FP4 recovery split, the final 400K geometry measured 3,083,785 KV
# tokens and completed two exact-boundary requests concurrently.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-400000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
export FP4_RECOVERY_POOL_GB="${FP4_RECOVERY_POOL_GB:-24}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
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

rank0_mount_args=()
if [[ -n "$MOET_STORE_RANK0_DIR" ]]; then
  for command in cmp findmnt stat; do
    command -v "$command" >/dev/null || {
      printf 'Missing required command for the rank-0 override: %s\n' "$command"
      exit 1
    }
  done
  [[ -d "$MOET_STORE_RANK0_DIR" ]] || {
    printf '%s\n' "Rank-0 store path is unavailable: $MOET_STORE_RANK0_DIR"
    printf '%s\n' "Run ./scripts/split-moet-store-tp2.sh after the first healthy startup."
    exit 1
  }
  rank0_device="$(findmnt -n -o SOURCE -T "$MOET_STORE_RANK0_DIR")"
  shared_device="$(findmnt -n -o SOURCE -T "$MOET_STORE_DIR")"
  [[ "$rank0_device" != "$shared_device" ]] || {
    printf '%s\n' "Rank-0 override and shared store are on the same filesystem: $rank0_device"
    exit 1
  }

  shopt -s nullglob
  rank0_packs=("$MOET_STORE_RANK0_DIR"/*.rank0of2.pack)
  shopt -u nullglob
  (( ${#rank0_packs[@]} == 1 )) || {
    printf '%s\n' "Expected exactly one rank0of2 pack in $MOET_STORE_RANK0_DIR; found ${#rank0_packs[@]}."
    exit 1
  }

  rank0_pack="${rank0_packs[0]}"
  rank0_stem="$(basename "$rank0_pack" .pack)"
  [[ -f "$MOET_STORE_RANK0_DIR/$rank0_stem.json" && \
     -f "$MOET_STORE_RANK0_DIR/$rank0_stem.lock" ]] || {
    printf '%s\n' "Rank-0 pack sidecar or lock file is missing for $rank0_stem."
    exit 1
  }

  for extension in pack json lock; do
    rank0_name="$rank0_stem.$extension"
    rank1_name="${rank0_name/.rank0of2./.rank1of2.}"
    rank0_source="$MOET_STORE_RANK0_DIR/$rank0_name"
    rank0_fallback="$MOET_STORE_DIR/$rank0_name"
    rank1_source="$MOET_STORE_DIR/$rank1_name"
    [[ -f "$rank0_fallback" && -f "$rank1_source" ]] || {
      printf '%s\n' "Shared store is missing $rank0_name or $rank1_name."
      exit 1
    }
    if [[ "$extension" == "pack" ]]; then
      rank0_size="$(stat -c %s "$rank0_source")"
      fallback_size="$(stat -c %s "$rank0_fallback")"
      rank1_size="$(stat -c %s "$rank1_source")"
      [[ "$rank0_size" == "$fallback_size" && "$rank0_size" == "$rank1_size" ]] || {
        printf '%s\n' "Rank pack sizes do not match; refusing the per-rank mount."
        exit 1
      }
    elif [[ "$extension" == "json" ]]; then
      cmp -s "$rank0_source" "$rank0_fallback" || {
        printf '%s\n' "Rank-0 sidecar differs from its verified shared-store fallback."
        exit 1
      }
    fi
    # Binding individual files makes an automatic stale-pack os.replace fail
    # visibly (EBUSY) instead of silently rebuilding rank 0 on the wrong SSD.
    rank0_mount_args+=(
      --mount "type=bind,src=$rank0_source,dst=/packs/$rank0_name"
    )
  done
  printf 'Using TP rank 0 pack from %s (%s); rank 1 remains in %s (%s).\n' \
    "$MOET_STORE_RANK0_DIR" "$rank0_device" "$MOET_STORE_DIR" "$shared_device"
fi

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

if (( MAX_NUM_SEQS < 2 )); then
  printf '%s\n' "MAX_NUM_SEQS must be at least 2 for sparse-MLA warm-up."
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
  "${rank0_mount_args[@]}" \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_PLANES_CACHE=/planes \
  -e VLLM_MOE_W2_STORE_DIR=/packs \
  -e VLLM_MOE_W2_DELTA_GB="$FP4_RECOVERY_POOL_GB" \
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
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tokenizer-mode deepseek_v4 \
  --no-scheduler-reserve-full-isl \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
