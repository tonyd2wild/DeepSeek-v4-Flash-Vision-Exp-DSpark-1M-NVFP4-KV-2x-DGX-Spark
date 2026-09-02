# VISION-EXP DEFAULT CONFIG — DeepSeek-V4-Flash-Vision-Exp + DSpark, 2× DGX Spark

> **Recipe summary lives in [`CURRENT.md`](CURRENT.md).** The runnable launcher is
> [`launchers/ds4-vision-tp2.sh`](launchers/ds4-vision-tp2.sh) (TP2) and
> [`launchers/ds4-vision-tp4.sh`](launchers/ds4-vision-tp4.sh) (TP4). This page is the
> long-form explanation of the TP2 flags.

> **Current default deployment (2026-08-31).** This is the experimental **vision** build of
> DeepSeek-V4-Flash (native image input) as we run it today. It is the same two-node DSpark
> recipe as [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) / the text README, pointed at the vision
> checkpoint and with the vision-model files added as read-only bind-mounts. Everything not
> restated here (NCCL/RoCE, JIT-cache split, garble history, tuning) carries over from the text
> recipe unchanged.

**Verified live:** TP=2 on **asusi** (rank0/head) + **bluey** (rank1/worker), served `:8888`,
clean output. Decode (warmed, single-stream, temp 0, with Patch 4): **count 80.1 / code 51.8 /
prose 33.2 tok/s**.

---

## Model + image

- **Model:** `DeepSeek-V4-Flash-Vision-Exp` — checkpoint pinned at commit
  `86f746b36186f0e567729a5c06a8c918caba82a9`.
- **Image (as deployed):** `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`
  (vLLM `0.21.1rc1.dev339+g1967a5627bc3`, B12X MXFP4 MoE). Same probe-c image family the text
  recipe was captured on — it predates the baked-in Patch 3/Patch 4 overlay, so both patches are
  injected here via bind-mount.

## Bind-mounts (patches + vision files)

Stage each source file on **both** nodes at `/var/tmp/` first — `start-*.sh` syncs the compose/env
files to the worker but **not** bind-mounted patch files, so a file missing on the worker runs
unpatched and silently gives half-fixed results.

```bash
# Patch 3 — cold-start garble root fix.
#   source: recipe/overlay/vllm/v1/core/sched/scheduler.py  ->  /var/tmp/patch3-scheduler.py
-v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro \

# Patch 4 — DSpark draft shared-expert loader fix.  *** REQUIRED ***
#   source: recipe/overlay/vllm/v1/spec_decode/dspark.py    ->  /var/tmp/spec-dspark.py
#   Without it the draft's always-on shared expert loads UNINITIALISED (12 tensors dropped),
#   acceptance collapses, and decode runs at ~half speed — failing SILENTLY (logger.debug).
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro \

# Vision-model files (native image support). Staged the same way onto the image's
# DeepSeek-V4 model + registry module paths.
-v /var/tmp/ds4v_model.py:.../ds4v_model.py:ro \
-v /var/tmp/ds4v_vision.py:.../ds4v_vision.py:ro \
-v /var/tmp/ds4v_mm.py:.../ds4v_mm.py:ro \
-v /var/tmp/ds4v_registry.py:.../ds4v_registry.py:ro \
```

> **The Patch 4 mount is the one that got dropped.** When the vision serving port was first stood
> up, its run command carried the Patch 3 and vision mounts but not `spec-dspark.py`. Everything
> booted, smoke-tested and served correct output — at roughly half the decode speed. Add the mount,
> and **verify it landed on every node** (see below). This is not optional on the vision port.

## Exact vLLM command (as running)

The vision port passes the same flags as the text recipe's "CURRENT BEST" profile, at
`max_num_seqs 12` / `gpu-memory-utilization 0.85` (the C12-style lane), pointed at the Vision-Exp
weights. The bind-mounts above are added to the `docker run` / compose service; the served command
is:

> This block is transcribed from [`launchers/ds4-vision-tp2.sh`](launchers/ds4-vision-tp2.sh),
> which is the executable source of truth. If the two ever disagree, the launcher wins.

```
/opt/env/bin/vllm serve <path-to-DeepSeek-V4-Flash-Vision-Exp> \
  --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"]}' \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 1048576 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 12 \
  --gpu-memory-utilization 0.85 \
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
  --nnodes 2 --node-rank <0|1> \
  --master-addr <head-fabric-ip> --master-port <port>
```

Flag notes:

- **`--served-model-name deepseek-v4-flash-dspark`** — the id clients ask for is the *text*
  recipe's id, deliberately: the vision build is a drop-in replacement on the same endpoint, so
  agents wired to `deepseek-v4-flash-dspark` do not need rewiring. Earlier revisions of this doc
  said `deepseek-v4-flash-vision-exp`; that was never what the launcher passed.
- **`--hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"]}'`** — selects the
  multimodal registry alias added by `ds4v_registry.py`. Without it vLLM decides
  `is_multimodal_model` from its static arch-name table and answers "is not a multimodal model".
- **`--max-model-len 1048576`** — 1M, the model's true YaRN ceiling and the standard across both
  topologies. (The launcher previously passed `1500000`.)
- **`--max-cudagraph-capture-size 12`** — pinned to `max_num_seqs` on TP2. The TP4 launcher uses
  `64` with `--max-num-seqs 64`.
- **`--async-scheduling`, `--enable-chunked-prefill`, `--enable-flashinfer-autotune`** — carried
  over from the text recipe's CURRENT BEST profile; they were omitted from earlier revisions of
  this block by mistake.
- **`--reasoning-config`** — `<think>` / `</think>` markers, paired with
  `--default-chat-template-kwargs '{"thinking":false}'` (thinking off by default).

Runtime env (B12X, DSpark, NCCL/RoCE, JIT-cache split) is identical to
[`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) and [`docker-compose.dspark.yml`](docker-compose.dspark.yml)
— reuse it verbatim.

### `k` is 5 on this image — do not use `k=6`

The DSpark drafter requires `num_speculative_tokens` divisible by `n_predict=5`. `k=6` is rejected
at boot with `must be divisible by n_predict=5`. `k=5` is correct and is the verified default. This
is the same runtime property documented for the text recipe (`k <= 5, or a multiple of 5`).

## Verify Patch 4 landed

Run against the head **and** worker container:

```bash
# Expect 6 — the two shared-expert mapping rows plus their explanatory comment.
docker exec <container> grep -c shared_experts \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py     # -> 6

# The stock/unpatched loader carries only the 2 attention rows and returns 0 here.

# Definitive: boot with VLLM_LOGGING_LEVEL=DEBUG, then confirm there are NO
# "Skipping unknown DSpark weight" lines for shared_experts.w1 / shared_experts.w3:
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight.*shared_experts"   # -> (empty)
```

Or use [`scripts/check-patch4.sh <head-container> <worker-container>`](scripts/check-patch4.sh),
which resolves the vLLM package root per image and fails closed.

## Node / network

Same two-node fabric as the text recipe (see [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) for the full
NCCL/RoCE table). Roles for this deployment:

| role | node | fabric | served |
|---|---|---|---|
| head (rank0) | **asusi** | `--master-addr <head-fabric-ip>` | `:8888` |
| worker (rank1, `--headless`) | **bluey** | worker fabric IP | — |

Container: `network_mode: host`, `ipc: host`, `shm_size: 64gb`, `gpus: all`,
`-v /dev/infiniband:/dev/infiniband`, memlock unlimited.

## Benchmark (2× DGX Spark, warmed, single-stream, temp 0)

Server-reported `completion_tokens` over wall time. **Warm the engine first** — this image has a
documented cold-start penalty (~58 → 83 tok/s on the text recipe); send a few long (500+ token)
generations before trusting any number.

| workload | with Patch 4 | without Patch 4 |
|---|---:|---:|
| count to 100 | **80.1** | 50.7 |
| code | **51.8** | 47.2 |
| prose | **33.2** | 30.4 |

KV pool: **2.79M tokens / 19.12 GiB** at `gpu_memory_utilization=0.85`. For reference, MiaAI-Lab's
comparable two-node recipe reports ~2.33M tokens and 62–83 tok/s single-stream.

**Honest reading.** The large gain on *count* is Patch 4 (the always-on shared expert now computes
correctly) **plus** the warm-up effect above — a cold count number understates the warm one.
Acceptance on prose-heavy traffic remains modest (~25%), which is inherent to the vision variant, so
**code and prose gained less than count**. Patch 4 is a real correctness fix regardless of the exact
per-workload speed attribution: the draft's shared expert was loading uninitialised, and now it is
not. Full mechanism, evidence, and per-position acceptance data:
[`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md).
