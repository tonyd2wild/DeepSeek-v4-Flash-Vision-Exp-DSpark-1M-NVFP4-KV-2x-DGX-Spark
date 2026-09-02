#!/usr/bin/env bash
# build-ds4v-files.sh [IMAGE] [DEST]
#
# Produces the four files that launchers/ds4-vision-tp2.sh (and launchers/ds4-vision-tp4.sh)
# bind-mount. Run it on EVERY node.
#
# Two of the four are shipped in this repo and are copied verbatim:
#   vision-exp/port/ds4v_vision.py   ported ViT + Aligner
#   vision-exp/port/ds4v_mm.py       multimodal processing / dummy inputs / processor
#
# The other two are DERIVED from whatever image you are running, by applying the
# patchers in vision-exp/port/ to files extracted from that image:
#   ds4v_model.py     = <image>:.../deepseek_v4/nvidia/model.py + patch_vision.py
#   ds4v_registry.py  = <image>:.../model_executor/models/registry.py + patch_registry.py
#
# They are derived rather than shipped on purpose: both are copies of vLLM files, so a
# checked-in copy would silently pin you to one image build and drift the moment the
# image moves. Generating them here means the port always applies to the image you
# actually run, and the patchers fail loudly (anchor count != 1) if the image changed
# under them. See issue #46.
set -euo pipefail

IMAGE="${1:-${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b}}"
DEST="${2:-/var/tmp}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="$HERE/port"

for f in patch_vision.py patch_registry.py ds4v_vision.py ds4v_mm.py; do
  test -f "$PORT/$f" || { echo "MISSING $PORT/$f -- run this from a clone of the repo" >&2; exit 2; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "MISSING image $IMAGE -- pull or build it first, or pass the tag as \$1" >&2; exit 3; }

mkdir -p "$DEST"
echo "image: $IMAGE"
echo "dest : $DEST"

# 1. copy the two files this repo owns
install -m 0644 "$PORT/ds4v_vision.py" "$DEST/ds4v_vision.py"
install -m 0644 "$PORT/ds4v_mm.py"     "$DEST/ds4v_mm.py"
echo "  ds4v_vision.py    copied from repo"
echo "  ds4v_mm.py        copied from repo"

# 2. extract the two vLLM files from the image we will actually run
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# The DSpark image keeps its interpreter in /opt/env/bin, which is not on the default
# PATH for a plain `bash -lc` -- without this you get "python3: command not found".
docker run --rm -v "$TMP:/out" --entrypoint bash "$IMAGE" -lc '
  set -e
  export PATH="/opt/env/bin:$PATH"
  V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
  cp "$V/models/deepseek_v4/nvidia/model.py" /out/ds4v_model.py
  cp "$V/model_executor/models/registry.py"  /out/ds4v_registry.py
'
# container writes as root; make them ours before patching
if [ ! -w "$TMP/ds4v_model.py" ]; then sudo chown "$(id -u):$(id -g)" "$TMP"/ds4v_*.py; fi

# 3. apply the ports
python3 "$PORT/patch_vision.py"   "$TMP/ds4v_model.py"
python3 "$PORT/patch_registry.py" "$TMP/ds4v_registry.py"

# 4. verify before installing -- a patcher that no-ops leaves you debugging a silent
#    "is not a multimodal model" instead of a clear failure here
python3 - "$TMP/ds4v_model.py" "$TMP/ds4v_registry.py" <<'PY'
import ast, sys
model, registry = sys.argv[1], sys.argv[2]
m = open(model).read(); r = open(registry).read()
ast.parse(m); ast.parse(r)
assert "ds4v_vision" in m, "model.py: vision import missing"
assert "e_score_correction_bias_vl" in m, "model.py: bias_vl gate missing"
assert 'startswith(("vision.", "aligner."))' in m, "model.py: loader guard missing"
assert "DeepseekV4VForConditionalGeneration" in r, "registry.py: multimodal alias missing"
print("  verified: both files parse and carry every port marker")
PY

install -m 0644 "$TMP/ds4v_model.py"    "$DEST/ds4v_model.py"
install -m 0644 "$TMP/ds4v_registry.py" "$DEST/ds4v_registry.py"
echo "  ds4v_model.py     generated from $IMAGE + patch_vision.py"
echo "  ds4v_registry.py  generated from $IMAGE + patch_registry.py"

echo
echo "All four staged in $DEST:"
ls -la "$DEST"/ds4v_*.py | sed 's/^/  /'
echo
echo "NOTE: vLLM caches model inspection on disk keyed by module+class. After changing"
echo "      these files, clear it or the old 'text-only' verdict is reused (issue: the"
echo "      alias resolves to the same class):"
echo "        rm -rf \$VLLM_CACHE_ROOT/modelinfos/"
