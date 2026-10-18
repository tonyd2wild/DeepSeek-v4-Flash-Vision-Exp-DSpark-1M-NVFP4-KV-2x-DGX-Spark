# The vLLM 0.31 lane: stock `vllm/vllm-openai:v0.31.0` on 2x DGX Spark

An **alternative lane** beside the current recipe. `CURRENT.md` and
`launchers/ds4-vision-tp2.sh` are unchanged. The lane runs
DeepSeek-V4-Flash-Vision-Exp with DSpark (k=5, probabilistic), TP2 and 1M
context, on the stock vLLM v0.31.0 image. The launcher is
[`launchers/ds4-vision-tp2-vllm031.sh`](../launchers/ds4-vision-tp2-vllm031.sh).

## Run it

1. **Install b12x on each node, once.** vLLM 0.31 lists `b12x==1.3.0` as an
   optional extra, and the stock image leaves it out. The launcher refuses
   without it and prints this command:
   ```bash
   mkdir -p ~/.cache/vllm031-deps
   docker run --rm -v ~/.cache/vllm031-deps:/out --entrypoint pip vllm/vllm-openai:v0.31.0 \
     install -q --no-deps --target /out/b12x-1.3.0 b12x==1.3.0
   ```
   b12x 1.3.0 pins CUTLASS DSL 4.6.2, while the image ships 4.7.1. Both
   1.3.0 and 1.5.0 (which pins 4.7.1) pass 0.31's own b12x device check on
   GB10. If 1.3.0 fails at kernel compile, install 1.5.0 and set
   `B12X_DIR`.
2. **Pull the image:** `docker pull vllm/vllm-openai:v0.31.0` (about 24 GB).
3. **Stop the current engine,** then run `launchers/ds4-vision-tp2-vllm031.sh 1`
   on the worker and then `... 0` on the head, as with `ds4-vision-tp2.sh`.
   - **Overriding the defaults:** host addresses, NICs, model paths, `PORT`
     and `KV_CACHE_MEMORY_BYTES` all come from the environment.
   - **Sandboxed or rented hosts** that refuse `--gpus all`: set
     `GPU_ARGS="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all"`. If the host
     has no `/dev/infiniband`, the launcher skips it and NCCL uses sockets.
   - **Loader:** `LOAD_STRATEGY` defaults to `eager`, which a 64K-page kernel
     requires. On stock DGX OS (4K pages) the default lazy loader also works.
   - **Boot time:** the first boot compiles 0.31's kernels into
     `~/.cache/vllm-dspark031`, a separate cache that never mixes with 0.21's.
     That boot takes about 7.5 min to `/health`; later boots about 4 min.

Going back is the reverse: stop it, and run `ds4-vision-tp2.sh` again. It uses
the same container name.

## What 0.31 brings, and what this lane adds

**Upstream now has these, so they aren't mounted:**
- **Vision-Exp natively:** `DeepseekV4ForConditionalGeneration` (ViT,
  aligner, `bias_vl`). No `ds4v_*` files or registry alias.
- **DSpark on model runner V2** (vllm#46995). It replaces Patches 1/2/2b and
  the Patch 3 scheduler.
  - **Patch 4 (shared-expert loader):** its symptom, silently half-speed
    drafting, is absent. Mean acceptance length is 2.6 on 0.31 against 2.38
    on 0.21. `scripts/check-patch4.sh` is specific to 0.21.
- **The b12x MXFP4 MoE** through `--moe-backend b12x`, with W4A16 via
  `VLLM_B12X_MOE_FP4_FORCE_A16=1`.

**Added, from this repo:**

| piece | why |
|---|---|
| b12x 1.3.0, mounted | the stock image lacks it: `Mxfp4 MoE backend 'B12X_MXFP4_BF16' does not support ... current device` |
| `patches/safetensors_e8m0_plugin` | the eager loader cannot decode `F8_E8M0` (`KeyError: 'F8_E8M0'`) |
| `patches-031/dspark031_plugin` (LOAD-1) | see below; without it the head is OOM-killed at shard 28/48 |
| `patches-031/detokenizer.py` | Patch 5 on 0.31: vllm#52486 plus the output-side guard (#42); `patches-031/pristine/` holds the source |

**Why LOAD-1 is needed:**
- **The cause:** `DeepseekV4ForConditionalGeneration.load_weights` does
  `iter(sorted(mapped, ...))`, so the text backbone gets its weights as one
  contiguous run. `sorted()` materialises the whole 156 GiB checkpoint first.
  - With the lazy loader that's mmap'd page cache.
  - With `--safetensors-load-strategy eager`, which the 64K-page kernel needs
    (the lazy loader hangs at shard 0 there), it's anonymous memory.
- **How it was found:** a probe showed every tensor of every shard still
  alive two shards later.
- **The fix:** stream `language_model.*` in checkpoint order and hold back
  only the vision and aligner weights (about 1.9 GiB). Memory stays flat at
  about 100 GB per node.
- **Safety and switches:**
  - LOAD-1 installs only if upstream's method still has that exact sort line.
  - `DSPARK031_STREAM_LOAD=0` turns it off.
  - The plugin's two other patches are off or inert here:
    - KV-7 (commit-time block dedupe) is off via `DSPARK_BLOCK_DEDUPE=0`,
      which leaves `BlockPool` untouched (the boot log says `KV-7 off`);
    - MNT-10 (a live long-prefill threshold) does nothing unless
      `VLLM_LONG_PREFILL_TOKEN_THRESHOLD` is set.

## Measured (2026-10-16, 2x GB10, same nodes, same tests back to back)

Against the 0.21 Keys/B12X image with the same checkpoint, TP2, DSpark k=5,
and the same `--kv-cache-memory-bytes` (20.29 GiB reserved). No other traffic
was on either engine.

| | 0.31 lane | 0.21 Keys/B12X |
|---|---|---|
| single stream, 400 tokens, temperature 0 | ~37 tok/s | ~21 tok/s |
| DSpark mean acceptance length | 2.6 | 2.38 |
| `bench_concurrent.py` 2/4/6, aggregate | 93 / 146 / 191 tok/s | 46 / 81 / 96 tok/s |
| 44K-token cold prefill | ~20 s | ~33 s |
| boot line, 1M | 2,115,150 tokens (2.02x) | 3,377,409 tokens (3.22x) |
| measured KV capacity (`tools/kv-capacity.sh`) | ~5.2M tokens | ~5-7M (see below) |
| needle, cold, temperature 0: 44K x5, 377K x1 | all pass | all pass |
| needle 44K x5 under 2/4/6 concurrency, 1M | 0/5 failures | (not run) |

**The KV rows:**
- **The boot lines are counted differently,** so they can't be compared.
  Measured directly, both versions hold about the same.
- **0.21's figure is a range,** because its usage counter keeps blocks counted
  after a request finishes.

**The KV format is the same in both:** `--kv-cache-dtype fp8` resolves to
`fp8_ds_mla`, 584 B per token. That's the same envelope as the current
recipe's "nvfp4_ds_mla" probe. A real 4-bit KV for V4 on SM12x isn't wired in
vLLM 0.31; FlashInfer 0.7.0.post1 has the 384 B record (flashinfer#4955).

## Tested from this branch (2026-10-18)

The launcher in this PR, run as committed on the same 2x GB10 pair, stock
`vllm/vllm-openai:v0.31.0`, the HF snapshot `6821d6ad` of
DeepSeek-V4-Flash-Vision-Exp, NCCL over two RoCE rails, warm JIT cache, no
other traffic.

| | result |
|---|---|
| boot to `/health` | 221 s |
| `smoke-deepseek-v4-flash-dspark.sh` | 6/6 |
| single stream, prose, 400 tokens, temperature 0 | 400 tokens in 10.8 s (~37 tok/s) |
| DSpark mean acceptance length | 2.57 |
| needle, cold, 44K x3 | 3/3 pass, ~20 s prefill each |
| `bench_concurrent.py` 2/4/6, aggregate | 94 / 175 / 187 tok/s |
| GPU KV cache, boot line | 2,115,150 tokens (2.02x at 1M) |
| `Traceback`/`ERROR` in either rank's log | 0 / 0 |

Host overrides used: `MASTER_ADDR`, `HOST_IP`, `MODELS_HOST`/`MODEL_DIR`
(the HF cache dir, so the snapshot's `../../blobs` links resolve),
`B12X_DIR`, `NCCL_IB_HCA` (two rails), `NCCL_IFACE`, `NCCL_IB_GID_INDEX=`
(empty: NCCL picks), `PORT=18888`.

## Open

- **Patch 3's cold-start garble isn't proven gone.** There was no real
  capture, so the checks used synthetic requests (`benchmarks/garble_synth.py`):
  - the cold multi-chunk needle with a prompt-echo detector passed;
  - in the Hermes-shaped replay, every reply flagged was either a scorer false
    positive or ambiguous (the model remarking on the synthetic system prompt).

  A real `garble_tap.py` capture through `replay_hermes.py` (and
  `replay_hermes_full.py` to read flagged replies in full) would settle it.
- **`--gpu-memory-utilization` sizing is untested on 0.31.** The lane pins the
  pool in bytes.
- **Measured on one pair of nodes, on a 64K-page kernel with eager loading.**
  Stock DGX OS (4K pages) is not yet measured.

## Tests

The plugin and Patch 5 tests need no GPU. They run inside the v0.31.0 image
(the CPU image works too: `vllm/vllm-openai-cpu:v0.31.0-arm64`):

```bash
docker run --rm -v "$PWD:/w" -e DSPARK031_DIR=/w/patches-031/dspark031_plugin \
  --entrypoint bash vllm/vllm-openai:v0.31.0 -c '
  pip install -q pytest
  cp /w/patches-031/detokenizer.py "$(python3 -c "import vllm.v1.engine.detokenizer as m; print(m.__file__)")"
  cd /w/patches-031/upstream-tests
  python3 -m pytest -q test_dspark031_plugin.py                        # 7 passed
  python3 test_load1_stream.py                                         # 0 failed
  python3 -m pytest -q test_stop_in_reasoning.py test_mnt5_output_side.py   # 8 passed'
```

The Patch 5 tests load the `Qwen/Qwen3-0.6B` tokenizer from the Hugging Face
cache or the network. `test_load1_stream.py` passes against LOAD-1; if
LOAD-1 is changed to sort before streaming, its no-buffering check fails.
