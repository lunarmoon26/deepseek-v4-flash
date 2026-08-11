# Roadmap

This roadmap records possible follow-up work for the verified DeepSeek V4
Flash 0731 TP=2 service. Items here are experiments, not supported launch
configurations. Keep `scripts/serve-moet-0731.sh` as the stable baseline until
an experiment passes its validation gates.

## Current baseline

As of 2026-08-11, the verified service has:

- two RTX PRO 6000 Blackwell GPUs running one TP=2 model instance;
- a 400,000-token total context window and `--max-num-seqs 2`;
- DSpark-7 speculative decoding, 4,096 maximum batched tokens, and an
  `fp8_ds_mla` KV cache;
- a 24 GiB/rank FP4 expert recovery pool;
- TP rank expert-pack reads split across the two data SSDs;
- a 34 GiB container RAM limit protecting the 48 GB desktop host.

At startup with the final geometry, vLLM reported 13.57 GiB available for its
GPU KV cache, 1,765,220 tokens of KV capacity, and theoretical capacity for
4.41 full 400K contexts. The scheduler is intentionally capped at two. At the
fixed 300 W per-GPU power cap, an exact-boundary run of two independent
399,872-token prompts plus 128 generated tokens completed without failures in
425.6 seconds at 1,880 aggregate tokens/s. During the run both GPUs held the
300 W cap at 100% utilization and container RAM stayed below 7 GiB.

This proves capacity and execution, not retrieval quality. A deterministic
400K needle suite must pass before calling the full window quality-verified.

## Completed tuning: 2026-08-11

Synthetic two-client 32K prompt plus 512-token output measurements selected
the stable launcher configuration above:

- raising `MAX_NUM_BATCHED_TOKENS` from 1,024 to 4,096 improved aggregate
  throughput from 1,103 to 1,739 tokens/s and median TTFT from 38.8 to
  23.0 seconds;
- DSpark-7 outperformed DSpark-5 and DSpark-3 at 4,096 tokens: 1,739 versus
  1,669 and 1,676 aggregate tokens/s respectively;
- a 28 GiB/rank FP4 recovery pool increased observed recovery coverage but
  reduced available KV capacity to 1,245,285 tokens and regressed aggregate
  throughput to 1,510 tokens/s, so the 24 GiB pool remains the baseline;
- a 32 GiB pool was not launched because the observed 24-to-28 GiB KV loss
  extrapolates below the 800,000 KV tokens required for two 400K requests.

The launcher now persists vLLM and FlashInfer autotuning data under the
ignored `cache/vllm` directory, allowing compatible restart geometries to
reuse their tuned kernels.

## Priority 1: measure before adding another cache tier

- Add deterministic needle/retrieval checks at 128K, 256K, and 400K, including
  a two-request 400K stress case.
- Test `--max-num-seqs 4` separately before adding storage offload. Preserve
  the value 2 in the stable launcher until startup, quality, latency, and
  memory use pass.
- Record GPU KV-cache utilization and prefix-cache hit rate under the real
  workload, not only synthetic random prompts.
- Use `iostat -dx 1` during long prompts to measure expert-pack traffic and
  latency on both SSDs.
- Separate cold page-cache, warm page-cache, and repeated-prefix results.

If real prefix-cache reuse remains near zero, an NVMe KV cache will add writes
and complexity without avoiding meaningful prefill work.

## Priority 2: evaluate the KV-versus-expert VRAM tradeoff

The current scheduler limit leaves much of the allocated GPU KV capacity
unreachable. Before offloading KV to disk, test whether some VRAM is more
valuable in a larger FP4 expert recovery pool.

- Sweep the expert recovery pool conservatively from the verified 24 GiB
  baseline.
- For every value, record startup success, remaining KV token capacity,
  expert-cache coverage or replay metrics, TTFT, decode throughput, and
  output quality.
- Preserve several GiB of workspace and CUDA graph headroom.
- Do not accept a larger context-capacity number if it requires disabling FP4
  recovery or otherwise weakens the validated quality path.

This optimization is likely to help the current one- or two-user workload
more than adding cold KV storage.

## Priority 3: LMCache/NVMe proof of concept

The useful experiment is persistent prefix reuse, not placing actively
decoded KV blocks on SSD. Hot KV must remain on the GPU. NVMe should hold cold
chunks that would otherwise be recomputed after GPU eviction or a server
restart.

Build this as a separate image and launcher:

1. Pin an LMCache and NIXL version compatible with the patched vLLM 0.24.0
   image. The present image does not install either dependency.
2. Start from LMCache's validated DeepSeek V4 Flash recipe and its
   `LMCacheMPConnector`. Treat the in-process connector as unvalidated until
   it passes the same sparse-MLA, TP=2, and DSpark tests.
3. Mount a dedicated cache directory into the experimental container. Start
   with a 64–100 GB LRU limit rather than allocating the full 4 TB SSD.
4. Bound all CPU or pinned-memory staging buffers so the existing container
   limit still protects the desktop.
5. Prefer direct I/O in the storage backend if it is supported and verified;
   avoid allowing a new file cache to consume the host's limited RAM.
6. Do not colocate an uncontrolled KV write workload with the rank pack files.
   Both SSDs now serve expert data, even though most of their capacity is
   free.

### Proof-of-concept workload

Use one approximately 30K-token prefix and a short changed suffix:

1. Run the first request with an empty cache.
2. Run a second request sharing the prefix.
3. Restart the model server without deleting the LMCache store.
4. Run a third request sharing the same prefix.

Capture TTFT, prefill throughput, decode throughput, cache-hit metrics,
container RSS, swap use, SSD read/write bandwidth, I/O latency, and total
bytes written.

### Acceptance gates

The experiment is useful only if all of the following are true:

- the repeated prefix is loaded rather than recomputed;
- the persisted prefix remains reusable after a clean server restart;
- warm-prefix TTFT improves materially over the stable baseline;
- steady decode throughput does not regress by more than 5%;
- expert-pack I/O latency remains stable during KV-cache activity;
- the container stays within its memory limit without host instability;
- the DeepSeek sparse-MLA and DSpark paths pass the smoke test and a quality
  evaluation;
- cache capacity, eviction, permissions, cleanup, and SSD-write behavior are
  observable and documented.

If these gates fail, retain the built-in GPU prefix cache and do not add an
NVMe tier.

## Deferred: true prefill/decode disaggregation

Disaggregated serving is distinct from using a disk-backed prefix cache. It
runs separate prefill and decode model instances and transfers KV between
them.

The current model needs both available GPUs as one TP=2 instance. A concurrent
prefill worker and decode worker would normally require two complete TP=2
groups—four suitable GPUs—or a future configuration in which the model fits
and performs acceptably with TP=1. Time-sharing the same two GPUs would not
provide useful disaggregation.

Revisit this only after suitable GPU capacity exists. The future experiment
must also verify:

- sparse-MLA compatibility with the selected KV connector;
- identical speculative-decoding configuration on prefill and decode workers;
- tensor-parallel KV layout and transfer compatibility;
- routing/proxy behavior, failure recovery, and backpressure;
- whether the workload has enough concurrent long prefills to justify the
  additional model replica and operational complexity.

## Non-goals

- Do not use NVMe as a per-token substitute for active GPU KV during decode.
- Do not reserve the entire remaining 4 TB before measuring cache reuse.
- Do not copy version-specific configuration from third-party tutorials into
  the stable launcher without checking current LMCache documentation.
- Do not claim a concurrency multiplier from another model or H100 workload.
- Do not merge experimental dependencies into the stable image before the
  rollback and quality gates pass.

## Risks to document

- Compatibility between LMCache/NIXL and the patched vLLM-Moet SM120 image.
- Additional pinned or pageable host-memory pressure on a 48 GB host.
- SSD contention with the per-rank expert pack stores.
- High write volume and filesystem metadata overhead from KV chunks.
- Sensitive prompt-derived KV data persisting on disk.
- Stale cache invalidation after model, tokenizer, template, or runtime
  configuration changes.

## References

- [Spheron: NVMe KV cache offloading](https://www.spheron.network/blog/nvme-kv-cache-offloading-llm-inference/)
- [vLLM: disaggregated prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
- [vLLM: NIXL connector compatibility](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)
- [LMCache: DeepSeek V4 Flash recipe](https://docs.lmcache.ai/recipes/deepseek_v4_flash.html)
- [LMCache: local storage backend](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html)
