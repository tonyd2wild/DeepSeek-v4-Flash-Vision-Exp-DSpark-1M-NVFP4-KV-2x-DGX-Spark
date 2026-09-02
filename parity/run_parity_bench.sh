#!/usr/bin/env bash
# Parity bench Tier A: {mmlu_pro, aime25, ifeval} x 3 seeds x 1 endpoint.
# Usage:  PARITY_API_KEY=sk-... run_parity_bench.sh <endpoint-name> <base_url>
# (A third argv is still accepted for the key but the env var is preferred —
#  argv is visible to every user on the box via `ps`.)
# NOTE: mmlu_pro's --limit is per-subtask (14 subtasks) → n=1120, and the
# campaign is a multi-hour/overnight affair at typical c6 throughput.
set -u
EP="$1"; URL="$2"; KEY="${PARITY_API_KEY:-${3:-}}"
OUT="${PARITY_OUT:-./parity-results}"; mkdir -p "$OUT"
LM="${LM_EVAL:-lm_eval}"

# ── Effort-render preflight (review #34, item 4) ────────────────────────────
# Fingerprints whether the endpoint actually renders `reasoning_effort` BEFORE
# the campaign spends anything: a 1-token request bare vs effort=max, read back
# via prompt_tokens. A backend that renders the effort preamble shows a delta
# (pair-measured: none=+0, high=+79, max=+92); stock vLLM silently ignores the
# field entirely (delta 0) — running the campaign there measures a DIFFERENT
# protocol than the card. Abort on delta 0 unless PARITY_ALLOW_NO_EFFORT=1.
# The fingerprint is stored next to the results so the card comparison can
# carry it. (The fingerprinting technique is issue #33's, turned into a guard.)
_pt() {
  curl -s --max-time 60 "$URL" -H "Content-Type: application/json" \
    ${KEY:+-H "Authorization: Bearer $KEY"} \
    -d "{\"model\":\"deepseek-v4-flash\",\"max_tokens\":1,$1\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
  | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["usage"]["prompt_tokens"])
except Exception: print(-1)'
}
PT_BARE=$(_pt "")
PT_MAX=$(_pt "\"reasoning_effort\":\"max\",")
if [ "$PT_BARE" -lt 0 ] || [ "$PT_MAX" -lt 0 ]; then
  echo "PREFLIGHT FAILED: endpoint did not answer the 1-token fingerprint probes ($URL)"; exit 1
fi
DELTA=$((PT_MAX - PT_BARE))
echo "effort fingerprint: bare=$PT_BARE max=$PT_MAX delta=+$DELTA" | tee "$OUT/effort_fingerprint.txt"
if [ "$DELTA" -eq 0 ] && [ "${PARITY_ALLOW_NO_EFFORT:-0}" != "1" ]; then
  echo "PREFLIGHT ABORT: endpoint does not render reasoning_effort (delta 0) —"
  echo "  your runs would measure a different protocol than the reference card."
  echo "  Set PARITY_ALLOW_NO_EFFORT=1 to run anyway (results are NOT card-comparable)."
  exit 1
fi

for SEED in 1234 2345 3456; do
  for SPEC in "mmlu_pro:80:24000" "aime25:30:30000" "ifeval:80:16000"; do
    TASK="${SPEC%%:*}"; REST="${SPEC#*:}"; LIMIT="${REST%%:*}"; TOKS="${REST#*:}"
    TAG="${EP}_${TASK}_s${SEED}"
    [ -f "$OUT/$TAG.done" ] && continue
    echo "=== $TAG @ $(date +%H:%M)"
    # seed=$SEED in --model_args is the one that reaches the request payload;
    # --seed alone only seeds python/numpy/torch/fewshot (review #34, item 1).
    OPENAI_API_KEY="$KEY" $LM --model local-chat-completions \
      --model_args "model=deepseek-v4-flash,base_url=$URL,num_concurrent=6,max_retries=3,timeout=1800,seed=$SEED" \
      --tasks "$TASK" --limit "$LIMIT" --apply_chat_template --log_samples \
      --gen_kwargs "{\"temperature\":0.6,\"top_p\":0.95,\"reasoning_effort\":\"max\",\"max_gen_toks\":$TOKS,\"until\":[]}" \
      --output_path "$OUT/$TAG" --seed "$SEED" > "$OUT/$TAG.log" 2>&1 \
      && touch "$OUT/$TAG.done"
    echo "    exit=$? nulls=$(grep -c 'null content' "$OUT/$TAG.log")"
  done
done
echo "CAMPAIGN $EP COMPLETE $(date +%H:%M)"
