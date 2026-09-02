# Parity Bench — verify your deployment actually matches the model

This folder lets any deployment of this recipe answer, with hard numbers: **"is my local serving faithful to DeepSeek-V4-Flash-0731, or is my
stack silently costing me intelligence?"**

## Why this exists

While debugging issue #18 we measured a 25-point GSM8K gap between this stack and a
hosted deployment of the *identical checkpoint* — caused entirely by serving-layer bugs
(stop-strings evaluated inside the reasoning segment), not by the model or the hardware.
After fixing the stack, our 2× GB10 pair matched the hosted reference **exactly**
(IFEval 0.9375 = 0.9375, first paired benchmark). Deviation from reference = actionable
serving defect. That is the entire premise.

## The frozen reference card

`reference-card_hosted_v3.json` — DeepSeek-V4-Flash-0731 measured on a professional
hosted deployment, protocol v3 (`reasoning_effort: max`, temp 0.6 / top_p 0.95, seeds
1234/2345/3456, lm-eval 0.4.12, fixed item slices):

| Benchmark | Median | Runs |
|---|---|---|
| IFEval (80) | **0.938** | 0.900 / 0.938 / 0.938 |
| MMLU-Pro (1120)¹ | **0.837** | 0.826 / 0.837 / 0.841 |
| GPQA Diamond (60) | **0.733** | 0.667 / 0.733 / 0.767 |
| AIME 2025 (30) | **0.700** | 0.700 / 0.700 / 0.733 |

¹ `--limit 80` is per-subtask; MMLU-Pro is a 14-subtask group, so n=1120. This also
sizes the wall clock: the full 3-seed campaign is a **multi-hour / overnight** affair
at typical local throughput, not a lunch break. GPQA Diamond appears in this table as
reference context but is not run by the script (the HF dataset is gated and needs
authenticated access); the machine-readable `parity_targets` list in the card says
which rows the script reproduces. The card's run-to-run spread reflects server-side
nondeterminism — the hosted runs' request-level seed was constant (see the card's
`seed_note`); local runs now pass real per-run seeds.

You do NOT need to re-buy this card. It is a property of the checkpoint. Your local
runs compare against it directly.

The card also carries reference-only rows the run script does not reproduce (GSM8K,
AA-LCR, HLE, SciCode, LiveCodeBench, Terminal-Bench-hard, τ²-telecom) — each with its
harness, protocol, and caveats spelled out in the card's `notes`. Treat those as
context, not as parity targets.

## Run it

```bash
pip install "lm-eval[api]==0.4.12" langdetect immutabledict   # once (pinned: the JSON gen_kwargs form and log sentinels are version-coupled)
PARITY_API_KEY="" ./run_parity_bench.sh mybox "http://localhost:8888/v1/chat/completions"
# results land in ./parity-results/<run>/, resumable via .done markers
# (the key travels via env, not argv — argv is visible in `ps`)
# The script first fingerprints whether your endpoint renders reasoning_effort
# (1-token probe) and refuses to start if it doesn't — stock vLLM silently
# ignores the field, and a no-effort campaign is not card-comparable.
```

Three seeds per benchmark; compare your medians to the card. Interpretation:

- **Within the card's run-range** → your serving is faithful. Enjoy.
- **A benchmark clearly below range** → serving defect. Check the null-content count
  in the run logs first (`grep -c "null content" *.log`).
- **High nulls** → truncation (raise `max_gen_toks`) or the stop-string bug (see below).

## Traps (learned the hard way) — 1, 2 and 4 are enforced by the harness; 3 and 5 are documented so you can check for them

1. **Stop-strings inside reasoning** — think-in-prompt models restate phrases like
   "Question:" while reasoning; serving stacks that match client stop-strings in the
   reasoning segment decapitate generation (score craters, e.g. 0.04 on MMLU-Pro).
   This harness sends `until: []` so it measures correctly on unpatched stacks. We
   observed this bug on two of three serving stacks tested (issue #18).
2. **Budget truncation ≠ model failure** — budgets here are sized ~1.6× the observed
   legit deliberation ceiling per benchmark. Count nulls before blaming the model.
3. **Runaway-loop tail** — at low temperature this model family can loop verbatim in
   reasoning (issue #18, mechanism B; reproducers there). Criterion: repeated fragment
   ≥3×, not budget exhaustion. At temp 0.6 the tail is ~1-2 items per 50 and hits the
   reference too.
4. **Single-prompt A/Bs are trajectory noise** — only rates across fixed-seed item
   sets discriminate. Three seeds minimum; medians with ranges.
5. **Grouped tasks (MMLU-Pro)** report subtask scores first — read the aggregate key.

## Provenance

Protocol, seeds, and metric keys are embedded in the reference card JSON. Reference
measured 2026-08-05 via a hosted provider serving the official checkpoint; local
validation on 2× DGX Spark (GB10) running this recipe at k=5. Full investigation
narrative: issue #18.
