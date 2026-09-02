#!/usr/bin/env bash
# check-patch4.sh — fail-closed preflight for the DSpark draft shared-expert loader fix.
#
# Patch 4 (see DSPARK-SHARED-EXPERT-FIX.md) adds two rows to _STACKED_PARAM_NAME_MAPPING in
# vllm/v1/spec_decode/dspark.py so the DSpark draft's always-on shared expert
# (shared_experts.gate_up_proj, fed by checkpoint tensors w1/w3) actually loads. WITHOUT it the
# stock loader drops 12 tensors via logger.debug("Skipping unknown DSpark weight") — invisible at
# INFO — the draft runs its always-active expert UNINITIALISED, and decode runs at ~half speed with
# perfect output quality. The break is silent; a warm smoke test passes on a broken deployment.
#
# This bit us on the vision serving port (DeepSeek-V4-Flash-Vision-Exp): the run command carried the
# Patch 3 + ds4v_*.py mounts but silently dropped the spec-dspark.py mount. Measured cost: count-to-100
# 50.7 -> 80.1 tok/s once restored.
#
# usage:  ./scripts/check-patch4.sh <container-name> [more containers...]
# exit 0 = Patch 4 present everywhere, exit 1 = missing somewhere.
set -uo pipefail

# The vLLM package root is NOT the same on every image (see check-patch3.sh). Ask Python where vllm
# actually lives, fall back to probing the known layouts. VLLM_ROOT overrides.
VLLM_ROOT_OVERRIDE="${VLLM_ROOT:-}"

resolve_root() {
  local c="$1" root=""
  if [ -n "$VLLM_ROOT_OVERRIDE" ]; then
    printf '%s\n' "$VLLM_ROOT_OVERRIDE"
    return 0
  fi
  for py in /opt/env/bin/python /usr/local/bin/python3 python3 python; do
    root=$(docker exec "$c" "$py" -c \
      'import os,vllm;print(os.path.dirname(vllm.__file__))' 2>/dev/null) || continue
    [ -n "$root" ] && { printf '%s\n' "$root"; return 0; }
  done
  for cand in /opt/env/lib/python3.12/site-packages/vllm \
              /usr/local/lib/python3.12/dist-packages/vllm \
              /usr/lib/python3/dist-packages/vllm; do
    if docker exec "$c" test -d "$cand" 2>/dev/null; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

rc=0

if [ $# -eq 0 ]; then
  echo "usage: $0 <container-name> [container-name...]" >&2
  echo "       VLLM_ROOT=/path/to/site-packages/vllm overrides autodetection" >&2
  exit 2
fi

for c in "$@"; do
  if ! docker inspect "$c" >/dev/null 2>&1; then
    echo "  ?? $c — container not found"
    rc=1
    continue
  fi
  if ! root=$(resolve_root "$c"); then
    echo "  ?? $c — could not locate the vllm package inside the container."
    echo "       Re-run with an explicit path, e.g.:"
    echo "         VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm $0 $c"
    rc=1
    continue
  fi
  DSPARK="$root/v1/spec_decode/dspark.py"
  if ! docker exec "$c" test -f "$DSPARK" 2>/dev/null; then
    echo "  ?? $c — no dspark.py at $DSPARK (vllm root resolved to $root)"
    rc=1
    continue
  fi
  # Patched loader has 6 lines matching shared_experts (2 mapping rows + comment); stock has 0.
  n=$(docker exec "$c" grep -c shared_experts "$DSPARK" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 6 ]; then
    echo "  OK   $c — Patch 4 present ($n shared_experts references) at $root"
  else
    echo "  FAIL $c — PATCH 4 MISSING ($n shared_experts references, expected 6)."
    echo "       The DSpark draft's shared expert loads uninitialised -> ~half decode speed, silently."
    echo "       Fix on every node:"
    echo "         cp recipe/overlay/vllm/v1/spec_decode/dspark.py /var/tmp/spec-dspark.py"
    echo "       then add to the container run (right after the Patch 3 mount):"
    echo "         -v /var/tmp/spec-dspark.py:$DSPARK:ro"
    echo "       (or rebuild as dspark-nvfp4-stage-c, which bakes Patch 4 in)"
    echo "       NOTE: mount to the path printed above, not to a remembered one —"
    echo "       a bind mount to the wrong path fails SILENTLY."
    rc=1
  fi
done

exit $rc
