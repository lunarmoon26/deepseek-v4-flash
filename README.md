# DeepSeek V4 Flash local service

## SM120 deployment

Stock vLLM does not currently support the DeepSeek V4 mHC/DeepGEMM path on RTX PRO 6000 Blackwell (`sm_120`). Use the `vLLM-Moet` SM120 Docker build instead. The 0731 + TP2 + DSpark launcher is experimental because its upstream project does not publish this exact recipe.

```bash
test -e .env || cp .env.example .env
# Edit .env for this machine; it is intentionally ignored by Git.
./scripts/download-official-0731.sh
./scripts/build-moet.sh
./scripts/serve-moet-0731.sh
```

The build requires Docker Engine and NVIDIA Container Toolkit. The launcher uses both GPUs and exposes the API at `http://127.0.0.1:8000/v1` by default.

## Requirements

- NVIDIA 595.71.05 or newer with the **open kernel modules** (`nvidia-driver-595-server-open` on this Ubuntu host)
- `uv` for downloading the official checkpoint
- About 170 GB free under `models/` for the checkpoint
- About 350 GB free on fast local storage for the completed Moet plane cache and NVMe expert pack
- Two visible GPUs

The official 0731 model uses DSpark speculative decoding, not MTP. Its separate download does not modify the existing MJPansa checkpoint cache.

The launcher disables NCCL P2P and vLLM PCIe all-reduce. On this host's PCIe `PHB` topology, NCCL otherwise deadlocks immediately after distributed initialization with both GPUs at 100% and only about 1 GB VRAM allocated.

The first `vLLM-Moet` conversion requires at least 220 GiB combined RAM and swap; 128 GiB or more physical RAM is recommended. On this RAM-constrained host, generated planes are persisted under `MOET_PLANES_CACHE`, while the FP4 expert recovery store is persisted under `MOET_STORE_DIR`. Keeping the expert store on NVMe is essential: without it, vLLM-Moet tries to hold a much larger expert store in host memory.

The two dedicated SN8100 SSDs can be destructively provisioned with 128 GiB swap on each drive and ext4 on their remaining capacity:

First set `DEEPSEEK_NVME1_SERIAL` and `DEEPSEEK_NVME2_SERIAL` in the ignored `.env` file. Obtain the values with `lsblk -d -o NAME,MODEL,SERIAL,SIZE`. The provisioning script resolves devices from those serials rather than relying on unstable `/dev/nvme*` numbering.

```bash
sudo env ERASE_DEEPSEEK_NVME=YES ./scripts/setup-nvme.sh
```

The equal-priority swap partitions are striped by the Linux swap subsystem. This is a slower bootstrap workaround for insufficient physical RAM, not a replacement for a RAM upgrade.

Use the two data filesystems for different Moet caches:

```bash
DEEPSEEK_NVME1_MOUNT=/mnt/deepseek-nvme1
DEEPSEEK_NVME2_MOUNT=/mnt/deepseek-nvme2
MOET_PLANES_CACHE="${DEEPSEEK_NVME1_MOUNT}/moet-planes-0731"
MOET_STORE_DIR="${DEEPSEEK_NVME2_MOUNT}/moet-store-0731"
```

The plane cache is about 203 GiB and is used primarily during conversion and startup. The TP=2 FP4 pack store is about 130 GiB and supplies expert rows during inference, so it initially belongs on the other SSD.

[The pinned vLLM-Moet revision](https://github.com/kacper-daftcode/vLLM-Moet/tree/0a927eafa2e7249099073e4c3bb1169ba4b1c328) creates separate `rank0of2` and `rank1of2` pack files inside its one configured store directory. It does not expose a separate directory setting per rank. After the first successful build, rank 0 can be copied to the plane-cache SSD and overlaid into the container as an individual bind mount:

```bash
# Add this to the ignored .env file:
MOET_STORE_RANK0_DIR="${DEEPSEEK_NVME1_MOUNT}/moet-store-rank0-0731"

./scripts/split-moet-store-tp2.sh
# Restart ./scripts/serve-moet-0731.sh after the verified copy completes.
```

The launcher then reads rank 0 from SSD1 and rank 1 from SSD2 while vLLM-Moet still sees a single `/packs` directory. It validates the two stores before launch and uses file bind mounts so a stale automatic rebuild fails visibly instead of silently putting rank 0 back on SSD2. Keep the original rank-0 files in `MOET_STORE_DIR`: they are rollback copies and provide the underlying Docker mount targets.

The pack files are read-only after their first successful build. A difference in lifetime bytes written immediately after conversion is therefore expected and is not evidence of continuing write amplification; steady inference is predominantly reads. Keep the old pack directory until the server has started and passed `scripts/smoke-test.sh` from the new location.

## Desktop-memory safety

On the 44 GiB-RAM desktop host, avoid allowing the model to consume all RAM or the 265 GiB NVMe swap. Extreme memory pressure can kill the graphical session or leave the system unresponsive.

`setup-nvme.sh` configures `vm.swappiness=100` for the RAM-constrained Moet conversion. When serving interactively, lower it to 10 instead:

```bash
sudoedit /etc/sysctl.d/99-deepseek-vllm-swap.conf
# Change: vm.swappiness=100
# To:     vm.swappiness=10
sudo sysctl --system
```

`serve-moet-0731.sh` applies Docker cgroup limits by default: 34 GiB RAM, a 30 GiB soft reservation, and 200 GiB total RAM+swap. This leaves roughly 11 GiB physical RAM plus about 99 GiB swap for Linux, the desktop, and the GPU driver while allowing the 155 GiB checkpoint to load. Docker kills only the model container if it exceeds a hard limit. The launcher also enables PyTorch's expandable CUDA allocator and uses vLLM-Moet's NVMe pack backend for the FP4 expert store.

The first conversion can be very slow under this cap. It persists completed layers in both `MOET_PLANES_CACHE` and `MOET_STORE_DIR`; do not delete either directory after an interrupted attempt. On startup, confirm the log reports a `PACK-FILE` backend for the `delta` store. This shows that the large FP4 recovery store is on NVMe instead of pinned host RAM.

The normal launch is:

```bash
./scripts/serve-moet-0731.sh
```

Do not add `--safetensors-load-strategy eager`: this checkpoint contains the `F8_E8M0` dtype, which the eager safetensors path cannot decode. If the disk-backed run still reaches the container limit while the desktop remains responsive, increase `MOET_CONTAINER_MEMORY` in 2 GiB steps while preserving a meaningful host reserve. Keep `MOET_CONTAINER_MEMORY_SWAP` at least 200 GiB for a first conversion:

```bash
MOET_CONTAINER_MEMORY=36g ./scripts/serve-moet-0731.sh
```

Keep `--max-num-seqs` at 2 or higher. FlashInfer's sparse-MLA autotuner creates a mixed warm-up batch containing one decode request and one prefill request; setting this to 1 fails with `AssertionError: No free indices`. This is a request-state limit, not a VRAM OOM.

If startup stops immediately after a `moe_w2 delta tier` line, retry once with
36 GiB and preserve the stopped container for diagnosis. This leaves about
9 GiB of physical RAM for the host:

```bash
MOET_CONTAINER_MEMORY=36g MOET_CONTAINER_MEMORY_RESERVATION=32g \
MOET_KEEP_CONTAINER=1 ./scripts/serve-moet-0731.sh
docker inspect --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  deepseek-v4-flash-moet
```

`oom=true` confirms the Docker limit was too low; increase the RAM limit only
in 2 GiB steps. Remove the retained diagnostic container afterwards:

```bash
docker rm deepseek-v4-flash-moet
```

Stop the server with `Ctrl-C`, or from another terminal:

```bash
docker stop deepseek-v4-flash-moet
```

After the server starts, verify it from a second terminal:

```bash
./scripts/smoke-test.sh
```

## Performance benchmark

`llama-bench` benchmarks models loaded by llama.cpp and cannot measure this vLLM OpenAI-compatible server. Use vLLM's serving benchmark instead. The wrapper below defaults to a 30,720-token input plus 512 output tokens, greedy decoding, three measured requests, and concurrency 1:

```bash
./scripts/bench-vllm-32k.sh
```

Results are printed to the terminal and saved under `benchmark-results/`. To test the server's two-request limit:

```bash
MAX_CONCURRENCY=2 NUM_PROMPTS=4 ./scripts/bench-vllm-32k.sh
```

The benchmark reports request throughput, input/output token throughput, time to first token (TTFT), time per output token (TPOT), inter-token latency (ITL), and end-to-end latency. `INPUT_TOKENS + OUTPUT_TOKENS` must remain at or below the served context limit. Each invocation uses a fresh random seed and defaults to zero warm-up requests so TTFT measures uncached prompts. Set `BENCH_SEED` explicitly only when reproducing a run against a freshly started server.

## Reasoning request

Use `reasoning_effort` through vLLM's chat template arguments:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
  "temperature": 1.0,
  "top_p": 0.95,
  "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
}
```
