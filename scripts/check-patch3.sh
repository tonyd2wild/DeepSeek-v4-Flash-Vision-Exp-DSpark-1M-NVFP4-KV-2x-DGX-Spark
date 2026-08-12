#!/usr/bin/env bash
# check-patch3.sh — fail-closed preflight for the cold-start garble root fix.
#
# Patch 3 (credit @roady001, issue #3) lives in the vLLM scheduler. A pre-Patch-3 image
# (notably `probe-c-p2b`) boots clean, passes smoke tests and serves warm requests
# correctly — it only garbles on COLD prefill. That makes it invisible until production.
#
# Measured 2026-07-30 on 2x DGX Spark with a ~20k-token agent prompt, forced cold prefill:
#     without Patch 3 : 44/44 requests garbled  (k=3, k=5, confidence scheduler, all doc settings)
#     with Patch 3    :  0/28 requests garbled  (k=3 and k=5)
# Warm requests never failed in any configuration, which is exactly why a 5-prompt gate
# passes on a broken deployment.
#
# usage:  ./scripts/check-patch3.sh <container-name> [more containers...]
# exit 0 = Patch 3 present everywhere, exit 1 = missing somewhere.
set -uo pipefail

# The vLLM package root is NOT the same on every image, and hardcoding it makes
# this preflight report "No such file" — a FAIL that looks like a missing patch —
# on a perfectly good deployment. Known layouts:
#   /opt/env/lib/python3.12/site-packages          this recipe's Stage-C image
#   /usr/local/lib/python3.12/dist-packages        Debian layout, e.g.
#                                                  ghcr.io/anemll/dspark-vllm-gx10
# Reported by @robotnurse in issue #22. Ask Python where vllm actually is, fall
# back to probing the known paths if no interpreter is on PATH.
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
  SCHED="$root/v1/core/sched/scheduler.py"
  if ! docker exec "$c" test -f "$SCHED" 2>/dev/null; then
    echo "  ?? $c — no scheduler.py at $SCHED (vllm root resolved to $root)"
    rc=1
    continue
  fi
  n=$(docker exec "$c" grep -c is_prefill_chunk "$SCHED" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 1 ]; then
    echo "  OK   $c — Patch 3 present ($n guard references) at $root"
  else
    echo "  FAIL $c — PATCH 3 MISSING. Cold prefills will garble."
    echo "       Fix on every node:"
    echo "         cp recipe/overlay/vllm/v1/core/sched/scheduler.py /var/tmp/patch3-scheduler.py"
    echo "       then add to the container run:"
    echo "         -v /var/tmp/patch3-scheduler.py:$SCHED:ro"
    echo "       (or rebuild as dspark-nvfp4-stage-c, which bakes Patch 3 in)"
    echo "       NOTE: mount to the path printed above, not to a remembered one —"
    echo "       a bind mount to the wrong path fails SILENTLY."
    rc=1
  fi
done

exit $rc
