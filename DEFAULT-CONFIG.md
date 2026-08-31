# DEFAULT CONFIG — DS4-Flash + DSpark, verified live 2026-07-04 (2× DGX Spark)

> **2026-07-31 - four things on this page are superseded.**
>
> 1. **Apply [Patch 4](patches/0004-dspark-shared-expert-gate-up-proj.patch).** vLLM's DSpark draft
>    loader silently drops the draft's always-on shared expert (12 tensors). Without it you run at
>    roughly half speed with *perfect output quality*. Measured: 32.7 -> 55.4 tok/s mean, acceptance
>    25.7% -> 60.2%. See [`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md).
> 2. **`k=7` does not merely fail at boot - it crashes on first generation** once the boot guard is
>    patched out (`size of tensor a (7) must match tensor b (5)`). The drafter emits exactly 5 per
>    pass. Also, the divisibility rule below is imprecise: the guard only fires when `k > n_predict`,
>    which is why `k=3`/`k=4` boot. Accurate rule: **`k <= 5`, or a multiple of 5.**
> 3. **`draft_sample_method` is a no-op for DSpark.** The rows below that call `probabilistic`
>    "mandatory" / "do NOT switch to greedy", and the "BEATS greedy: 49 vs 32 tok/s mixed" claim,
>    are **withdrawn**. In this runtime the DSpark proposer only populates draft probabilities when
>    `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` (`vllm/v1/spec_decode/dspark_proposer.py`), so the rejection
>    sampler takes the same `NO_DRAFT_PROBS` path for greedy and probabilistic. Re-measured
>    2026-07-31: no difference. The cold-start garble is fixed by Patch 3, not by this flag — see
>    [`docs/PATCHES.md`](docs/PATCHES.md) ("not needed to fix this garble once the scheduler guard
>    is in place"). The setting is harmless to leave in place.
> 4. **The command blocks below still point at the preview checkpoint**
>    (`/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`). This repo now targets
>    `DeepSeek-V4-Flash-0731`; substitute your 0731 weights path. Everything else in the ⭐ block
>    still matches what is served today except that the live server also passes
>    `--reasoning-config` and `--enable-flashinfer-autotune`.
>
> Throughput and acceptance figures on this page were measured on the **preview** checkpoint with the
> **stock (buggy) loader**, and on the **stock (censored)** weights — there is no
> censored-vs-uncensored A/B behind these numbers.


This is the **canonical, verified-working config** — captured live from the running
deployment before teardown. Reproduce this exactly to get the benchmarked result.

**Verified:** TP=2 on Asusi (rank0/head) + Spark4 (rank1/worker), served `:8888`,
clean output (no garble), honest speed **~49 tok/s mixed / 54–60 structured-agentic**.

---

## ⭐ CURRENT BEST — re-measured 2026-07-29 (use this)

The 2026-07-04 profile below is kept as the historical as-deployed record. **This is the
config to run now.** It was re-measured head to head against the newer vLLM 0.25.2 runtime
on the same two nodes, and it won — see
[`RUNTIME-BAKEOFF-2026-07-29.md`](RUNTIME-BAKEOFF-2026-07-29.md).

**Measured:** decode **84.3 tok/s peak / 67.6 mean** (5 content types, temp 0, warm),
**197.3 tok/s aggregate at c6**, prefill **2,639 tok/s at 100K**, KV pool
**1,548,597 tokens**. Garble gate clean.

```
/opt/env/bin/vllm serve /cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 1048576 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.78 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking":false}' \
  --nnodes 2 --node-rank <0|1> \
  --master-addr <head-fabric-ip> --master-port <port>
```

### What changed from the 2026-07-04 profile, and why

| knob | 2026-07-04 | now | why |
| --- | ---: | ---: | --- |
| `max-model-len` | 350000 | **1048576** | 1M is the model's calibrated YaRN ceiling; the KV pool supports it at this gmu. |
| `max-num-seqs` | 12 | **6** | 6 measured better under real traffic and leaves per-request KV headroom; 12 is a known trigger for the memory-pressure class of crash in issue #8. |
| `gpu-memory-utilization` | 0.80 | **0.78** | Speculative decode allocates buffers on the *first real request*, not at boot — 0.80 boots and passes smoke tests, then dies under traffic. See issue #8. |
| `max-cudagraph-capture-size` | 12 | *unset* | When set it must equal `max_num_seqs x (k+1)`; leaving it unset lets vLLM derive it. A stale literal silently truncates capture under concurrency (fixed in PR #5, credit @Wpnx330). |
| `num_speculative_tokens` | 5 | **5** (unchanged) | `k=5` is correct and worth ~24% over `k=3`. `k=7` is rejected at boot (must be a multiple of `n_predict=5`) and `k=10` boots but crashes every generation. |
| `draft_sample_method` | probabilistic | **probabilistic** (unchanged) | Mandatory. Omitting it gives a **greedy draft**, which is the documented root cause of agent garble. Do not remove. |

### Do not add these

Measured on this hardware and rejected — each was neutral or worse:

- `--max-model-len 200000` (65.2 vs 66.8 tok/s baseline) — smaller context buys no speed.
- `--max-num-seqs 2` (66.3) — fewer sequences buys no single-stream speed.
- `--max-cudagraph-capture-size 36` (65.4) — no gain when already derived correctly.
- `--override-generation-config` — remove it entirely; `repetition_penalty` on the
  spec-decode path is a known crash risk. Use `--generation-config vllm`.
- **NVFP4 vs FP8 KV is a context lever, not a speed lever.** On a sibling deployment the
  two measured identically for throughput (41.4 vs 41.5 peak). Choose it for pool size.

---

## Model + image
- **Model:** `fraserprice/DeepSeek-V4-Flash-DSpark` (in HF cache: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`)
- **Image (as deployed):** `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`
  (drowzeys/Keys overlay w/ Patch 1/2/2b concurrency). vLLM 0.21.1rc1, venv `/opt/env`, ENTRYPOINT=`["bash"]`.
- **Canonical image:** `vllm-dspark-runtime:dspark-nvfp4-stage-c` built via `./build-dspark-vllm-runtime.sh`
  — its overlay ALREADY contains Patch 3 (commit e83606a), so a fresh stage-c build needs NO bind-mount.
- **Patch 3 (roady001, issue #3 — cold-start garble root fix):** in a fresh stage-c build it's baked in.
  As-deployed on probe-c-p2b (which predates Patch 3) it was injected via bind-mount:
  `-v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro`
  (source = repo `recipe/overlay/vllm/v1/core/sched/scheduler.py`).
- **Patch 4 (DSpark draft shared-expert loader fix — REQUIRED on probe-c images):** in a fresh
  stage-c build it's baked in. On the pre-patch probe-c-p2b image it must be injected via bind-mount
  **right after the Patch 3 mount** — omitting it loads the draft's always-on shared expert
  uninitialised and runs at ~half speed, silently (see [`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md)):
  `-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro`
  (source = repo `recipe/overlay/vllm/v1/spec_decode/dspark.py`). Verify with
  `docker exec <container> grep -c shared_experts /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py`
  → expect **6** (stock loader returns 0), or run [`scripts/check-patch4.sh`](scripts/check-patch4.sh).
  The current **vision** deployment (`DeepSeek-V4-Flash-Vision-Exp`) runs this exact probe-c image
  with both bind-mounts **plus** the `ds4v_*.py` vision-model mounts — see
  [`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md).

## Exact vLLM command (byte-for-byte, as running)
```
/opt/env/bin/vllm serve /cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 350000 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 12 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  --nnodes 2 --node-rank <0|1> --master-addr 192.168.192.3 --master-port 25440
```
Key spec-decode choices (verified): **`num_speculative_tokens: 5`** (not 3 — Patch 3 makes the
higher depth garble-safe; +24% over 3) and **`draft_sample_method: probabilistic`** (BEATS greedy
for DSpark's calibrated draft heads: 49 vs 32 tok/s mixed — do NOT switch to greedy). No
`--override-generation-config`. `--max-cudagraph-capture-size` MUST equal `--max-num-seqs`.

## Runtime env (compose defaults — all verified)
```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  VLLM_TRITON_MLA_SPARSE=1  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0  VLLM_SKIP_INIT_MEMORY_CHECK=1  VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_MOE=1  VLLM_USE_B12X_WO_PROJECTION=1  VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0
VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16  B12X_W4A16_TC_DECODE=0
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0  VLLM_DSPARK_CONFIDENCE_SCHEDULER=off  VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1  VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0  VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0  VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
VLLM_DSV4_B12X_COMPRESSED_MLA=0  VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0  VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0
TORCH_CUDA_ARCH_LIST=12.1a  FLASHINFER_CUDA_ARCH_LIST=12.1a  FLASHINFER_DISABLE_VERSION_CHECK=1
TILELANG_CLEANUP_TEMP_FILES=1  DG_JIT_USE_NVRTC=0  DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_NET=IB  NCCL_IB_DISABLE=0  NCCL_IB_HCA=rocep1s0f0  NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_GID_INDEX=3  NCCL_CROSS_NIC=1  NCCL_CUMEM_ENABLE=0  NCCL_IGNORE_CPU_AFFINITY=1
NCCL_DEBUG=WARN  NCCL_NVLS_ENABLE=0
```

> **Dual-HCA note (direct QSFP pairs):** the GB10 QSFP port is TWO virtual NICs
> (2x PCIe x4). With only one configured, NCCL runs at ~half the port: measured
> 98 Gb/s single vs **161 Gb/s** with both HCAs listed and `NCCL_IB_MERGE_NICS=1`
> (+64% busbw, nccl-tests). Both interfaces need an IP (separate subnets) and MTU
> 9000; map names with `ibdev2netdev`; confirm GID 3 is RoCE v2 via
> `/sys/class/infiniband/<hca>/ports/1/gid_attrs/types/3`. Two related traps:
> pre-2026-04 BIOS wires the second controller at Gen5 x2 (half bandwidth -
> update via `fwupdmgr` first), and GB10 has no GPUDirect RDMA (GDR 0), which is
> the remaining decode ceiling. On a switched fabric (like the CRS812 setup
> above) the second NIC may be deliberately unused - this note is for
> back-to-back QSFP pairs chasing interconnect-bound decode.

```
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  HF_HUB_DISABLE_XET=1  HF_HOME=/cache/huggingface  VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
```

## Node / network (2× DGX Spark over CRS812 switched fabric 192.168.192.0/24)
| role | node | user | fabric IP | tailnet | GID | HF cache |
|---|---|---|---|---|---|---|
| head (rank0) | Asusi | tonyspark3 | 192.168.192.3 | 100.90.25.78 | 3 | /home/tonyspark3/.cache/huggingface |
| worker (rank1, --headless) | Spark4 | tonyspark4 | 192.168.192.4 | 100.121.11.91 | 3 | /home/tonyspark4/.cache/huggingface |

MASTER_ADDR=192.168.192.3, MASTER_PORT=25440. Container: `network_mode: host`, `ipc: host`,
`shm_size: 64gb`, `gpus: all`, `-v /dev/infiniband:/dev/infiniband`, memlock unlimited.

## Reproduction (worker-first)
1. Ensure image on both nodes + model in HF cache (or build stage-c via `./build-dspark-vllm-runtime.sh`).
2. If using a pre-Patch-3 image (probe-c-p2b), stage `patch3-scheduler.py` at `/var/tmp/` on both nodes
   (`cp recipe/overlay/vllm/v1/core/sched/scheduler.py /var/tmp/patch3-scheduler.py`) + add the bind-mount.
   A fresh stage-c build skips this (Patch 3 already in the overlay).
3. Worker (Spark4): `cd <repo> && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 HF_CACHE=/home/tonyspark4/.cache/huggingface VLLM_HOST_IP=192.168.192.4 docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d`
4. Head (Asusi): same with `NODE_RANK=0 HEADLESS= VLLM_HOST_IP=192.168.192.3`.
5. ~4-5 min: weight-load + 2-node NCCL + KV profiling + CUDA-graph capture → serves `:8888`.
   Smoke: `curl :8888/v1/chat/completions ... "Reply with exactly: NVFP4 DSPARK OK"`.

## Honest benchmark (temp 0, authoritative completion_tokens/wall, 5 varied prompts)
| category | tok/s |
|---|---|
| Math | 60.1 |
| JSON | 54.0 |
| Code | 53.8 |
| Communication | 42.0 |
| Narrative | 33.7 |
| **mixed avg** | **48.7** |

Structured/agentic (JSON/code/math — the supervisor workload) = 54–60 tok/s. Deterministic
best-case ~75 (not representative). DSpark draft acceptance ~92% on structured, ~40% on creative.

## openclaw wiring (Mac)
A/B proxy `workspace/ds4-lb-proxy/proxy.py` A-backend → `http://100.90.25.78:8888`. Supervisors
flipped via `workspace/spark-swap/route-supervisors.py ds4` (target = `ds4-lb-proxy/deepseek-v4-flash-dspark`),
gateway `ai.openclaw.gateway` kickstarted. Fallback = 3090 27B.
