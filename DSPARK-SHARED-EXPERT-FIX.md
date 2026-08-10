# The DSpark shared-expert loader bug — +69% decode on DeepSeek-V4-Flash-0731

**Date:** 2026-07-31
**Applies to:** any vLLM serving `DeepSeek-V4-Flash-0731` (or the `-DSpark` preview) with `method: dspark`
**Impact measured here:** mean decode **32.7 → 55.4 tok/s**, peak **42.0 → 66.1 tok/s**, draft acceptance **25.7% → 60.2%**
**Fix:** two lines — [`patches/0004-dspark-shared-expert-gate-up-proj.patch`](patches/0004-dspark-shared-expert-gate-up-proj.patch)

---

## Symptom

After swapping from the preview checkpoint to the official `DeepSeek-V4-Flash-0731` release, decode
throughput roughly halved. Critically, **output quality was perfect** — no garble, no drift, no
special-token leakage. Only speed changed.

The give-away is in the spec-decode metrics, not the throughput number:

```
Mean acceptance length: 2.28, Accepted throughput: 19.00 tokens/s,
Drafted throughput: 74.49 tokens/s,
Per-position acceptance rate: 0.631, 0.282, 0.181, 0.114, 0.067,
Avg Draft acceptance rate: 25.5%
```

Drafted throughput is **healthy**. Accepted throughput has collapsed. That is an acceptance
failure, not a step-time failure.

## Why it looks like a model regression (and isn't)

`tok/s = steps/s × accepted-tokens-per-step`. Measured across every content type and every config
we tried, `steps/s` never moved off ~14.4. The engine, the fabric, the KV cache and the target model
were all healthy the entire time. The whole deficit was in `accepted-tokens-per-step`.

Because speculative decoding is **verified by the target model**, a bad draft can never corrupt
output — it can only cost speed. So a broken drafter presents as "the new weights are slower",
which is exactly the wrong place to look.

## Root cause

The DSpark draft's FFN is a `DeepseekV4MoE` containing a shared expert built as a `DeepseekV4MLP`,
whose projections are `gate_up_proj` (a `MergedColumnParallelLinear`, fed by checkpoint tensors
`w1` and `w3`) and `down_proj` (fed by `w2`).

The draft weight loader renames only `w2`:

```python
# vllm/models/deepseek_v4/nvidia/dspark.py:1066
name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
```

and the stacked-parameter mapping it consults contains **only the two attention entries**:

```python
# vllm/v1/spec_decode/dspark.py:15-18   (BEFORE)
_STACKED_PARAM_NAME_MAPPING = (
    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),
    ("attn.fused_wqa_wkv", ".attn.wkv", 1),
)
```

So `shared_experts.w1` and `shared_experts.w3` match nothing, fall through to:

```python
# vllm/models/deepseek_v4/nvidia/dspark.py:1122-1125
param = params_dict.get(name)
if param is None:
    logger.debug("Skipping unknown DSpark weight %s", name)
    continue
```

…and are dropped. `logger.debug` is invisible at the default INFO level, so **the load reports
success**.

**12 checkpoint tensors are lost** — `w1` and `w3`, each as `weight` + `weight_scale_inv` (renamed
from `.scale` before the mapping is consulted), across all three draft stages. That leaves these six
parameters uninitialised:

```
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight_scale_inv
```

`n_shared_experts: 1`, and the shared expert is **always-on** — its output is summed into every
token, unconditionally, alongside the routed experts. So each of the three draft stages ran with
its always-active expert uninitialised. The drafter still produces fluent, plausible tokens; they
just disagree with the target far more often.

### The tell: the target loader has the rows the draft loader is missing

```python
# vllm/models/deepseek_v4/nvidia/model.py:1952-1953   (target — correct)
("gate_up_proj", "w1", 0),
("gate_up_proj", "w3", 1),
```

The draft loader lost them when its mapping was narrowed to avoid a name collision — the DSpark
checkpoint also contains `markov_head.markov_w1`, which must not be treated as an FFN `w1` shard.
The narrowing worked, but took the shared-expert shards with it.

## The fix

```python
# vllm/v1/spec_decode/dspark.py:15-18   (AFTER)
_STACKED_PARAM_NAME_MAPPING = (
    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),
    ("attn.fused_wqa_wkv", ".attn.wkv", 1),
    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),
    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),
)
```

**Why this is safe:**

- `map_dspark_stacked_param_name()` returns early on `".experts."` — note the **leading dot** — so
  routed experts (`ffn.experts.0.w1`) are untouched and still go through `expert_mapping`.
- The new patterns are anchored on the full `".shared_experts.wN"` segment. `markov_w1` has no
  `.shared_experts.` prefix and cannot match. Verified explicitly:

  ```
  43.ffn.shared_experts.w1.weight            -> ('...shared_experts.gate_up_proj.weight', 0)
  43.ffn.shared_experts.w1.weight_scale_inv  -> ('...shared_experts.gate_up_proj.weight_scale_inv', 0)
  44.ffn.shared_experts.w3.weight            -> ('...shared_experts.gate_up_proj.weight', 1)
  43.ffn.experts.0.w1.weight                 -> None      (routed expert, unchanged)
  45.markov_head.markov_w1.weight            -> None      (no collision)
  43.attn.wkv.weight                         -> ('...attn.fused_wqa_wkv.weight', 1)
  ```

- `.scale` → `.weight_scale_inv` renaming happens at `dspark.py:1074-1080`, *before* the mapping is
  consulted, and `_EXPERT_SCALE_RE = r"\.experts\.\d+\.w[123]\.scale$"` requires a digit after
  `.experts.`, so `shared_experts` correctly takes the `.weight_scale_inv` branch that
  `MergedColumnParallelLinear` expects for FP8 block quantisation.
- A wrong mapping **fails loudly** — `dspark.py:1086-1087` raises `KeyError` rather than skipping.
  A clean boot is therefore positive evidence the mapping resolved.

## Results

Same hardware (2× DGX Spark, TP=2), same runtime, same checkpoint, same flags. **The loader mount is
the only variable.**

| | accept | tok/step | steps/s | mean tok/s | peak tok/s |
|---|---|---|---|---|---|
| 0731, stock loader | 25.7% | 2.28 | 14.4 | 32.7 | 42.0 |
| **0731, patched** | **60.2%** | **4.01** | 13.8 | **55.4** | **66.1** |
| preview `-DSpark`, stock loader | 57.8% | 3.89 | 14.3 | ~56 | — |

Per-position acceptance:

```
before   0.631 / 0.282 / 0.181 / 0.114 / 0.067
after    0.826 / 0.725 / 0.572 / 0.471 / 0.399
```

By content type, patched:

| content | accept | tok/step | tok/s |
|---|---|---|---|
| structured / repetitive | 78.3% | 4.91 | 66.1 |
| code generation | 68.7% | 4.43 | 62.2 |
| prose reasoning | 33.7% | 2.68 | 37.8 |

Prose reasoning remains the weak case. That is genuine content difficulty — the least predictable
text — and the preview checkpoint shows the same shape. Acceptance is content-driven; a single
number without the content mix behind it is not meaningful.

**The preview carries the same 18 shared-expert tensors**, so the same 12 are dropped there and the
bug was never 0731-specific. It was **not**, however, re-measured with the patch applied, and its
stock 57.8% already sits close to 0731's patched 60.2% — so the effect on the preview is
unquantified, and nothing measured here explains why the same missing tensors cost 0731 so much
more. Treat the preview case as an open question, not as "also roughly half speed".

## How to check whether you are hit

You are affected if you serve **any** DeepSeek-V4-Flash DSpark checkpoint on a vLLM build whose
`_STACKED_PARAM_NAME_MAPPING` lacks the `shared_experts` rows.

```bash
# 1. does your checkpoint carry shared-expert draft tensors? (expect 18: w1/w3 = 12, w2 = 6)
python3 - <<'PY'
import json, glob
wm = json.load(open(glob.glob("<MODEL_DIR>/*.index.json")[0]))["weight_map"]
hits = sorted(k for k in wm if "mtp" in k and "shared_experts" in k)
print(len(hits), "shared-expert mtp tensors")
print("\n".join(hits[:6]))
PY

# 2. does your runtime's mapping include them?
grep -A6 '_STACKED_PARAM_NAME_MAPPING' \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py

# 3. definitive: DEBUG logging will print every dropped tensor at load
#    VLLM_LOGGING_LEVEL=DEBUG ... then:
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight"
```

Apply with a read-only bind mount — no rebuild required:

```bash
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro
```

## What this was NOT

Recorded so nobody repeats the search. Every item below was measured, not assumed:

| tried | result |
|---|---|
| `draft_sample_method` greedy vs probabilistic | no change — it is a **no-op** for DSpark; the proposer never populates draft probs unless `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` |
| `fp8_ds_mla` vs `nvfp4_ds_mla` KV | no change (26% both) — KV dtype is a context lever, not an acceptance lever |
| temperature 0 vs 0.7 | no change |
| B12X kernels off (`VLLM_USE_B12X_MOE=0`, `_WO_PROJECTION=0`) | **worse** — steps/s 14.4 → 10.7, then crashed |
| dedicated node pair, zero competing traffic | no change — contention was never involved |
| runtime / image build differences | identical vLLM version and identical DSpark env across nodes |
| draft tensor names & dtypes vs preview | byte-identical structure; only values differ |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` / `_SCHEDULER` | inert — `0.0`/`off` means *no gating*, and the confidence head is skipped on the hot path |

## Correction to earlier releases of this repo

Two things previously documented here were wrong, and are fixed in this revision:

1. **`k=7` is not merely "rejected at boot".** The boot-time guard
   (`num_speculative_tokens % n_predict != 0`) can be patched out — but the run then **crashes on
   the first generation**:

   ```
   RuntimeError: The size of tensor a (7) must match the size of tensor b (5)
   ```

   The drafter emits exactly `dspark_block_size` (5) tokens per pass and `propose()` calls it once;
   multi-block drafting is not implemented. `k=7` on DeepSeek-V4-Flash needs real proposer work, not
   a flag. This is the same root cause as the previously noted "k=10 boots but crashes".

2. **The divisibility rule was stated imprecisely.** The guard only fires when
   `k > n_predict`, which is why `k=3` and `k=4` boot fine. The accurate rule is: **`k ≤ 5`, or a
   multiple of 5.** Upstream 0.25.2 uses the **same** `>` check, but resolves `n_predict` to
   `num_nextn_predict_layers` (1 for this checkpoint) rather than `dspark_block_size`, so the
   guard never fires there; and its DSpark speculator sizes the draft block from `k`, so no shape
   mismatch follows. See issue #22.

Where k=7 *does* work is on drafters whose draft block is at least 7 — e.g. MiMo-V2.5 DFlash
(`block_size` = 8, run here at `num_speculative_tokens: 7`), GLM-5.2's DSpark speculator
(`block_size` = 8, and its own `speculators_config` asks for 7 speculative tokens), and the Inkling
DSpark preview (`dspark_block_size` = `n_predict` = 7). DeepSeek-V4-Flash is 5 in **both** the
preview and the 0731 release.
