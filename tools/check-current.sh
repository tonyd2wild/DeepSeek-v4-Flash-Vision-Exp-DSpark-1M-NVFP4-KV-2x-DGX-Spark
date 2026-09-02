#!/bin/bash
# tools/check-current.sh: verify that every "sha256 <hex>  <path>" line in CURRENT.md matches the
# file in the tree, and that every launcher listed under launchers/ (or the repo's named launcher)
# has such a line. Run locally before a PR; CI runs it on every PR and push to main.
#   Update the hashes after an intentional change:  bash tools/check-current.sh --write
set -u
cd "$(dirname "$0")/.."
[ -f CURRENT.md ] || { echo "no CURRENT.md at the repo root"; exit 1; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }
launchers=$( { ls launchers/*.sh 2>/dev/null; ls launch-*.sh 2>/dev/null; } | sort -u)
if [ "${1:-}" = "--write" ]; then
  tmp=$(mktemp)
  grep -vE '^sha256 [0-9a-f]{64}  ' CURRENT.md > "$tmp"
  { cat "$tmp"; echo; echo "<!-- launcher hashes, maintained by tools/check-current.sh --write -->"; for f in $launchers; do echo "sha256 $(sha "$f")  $f"; done; } > CURRENT.md
  rm -f "$tmp"; echo "CURRENT.md hashes written for: $launchers"; exit 0
fi
rc=0
for f in $launchers; do
  want=$(grep -E "^sha256 [0-9a-f]{64}  $f\$" CURRENT.md | awk '{print $2}' | head -1)
  have=$(sha "$f")
  if [ -z "$want" ]; then echo "MISSING: $f has no sha256 line in CURRENT.md"; rc=1
  elif [ "$want" != "$have" ]; then echo "DRIFT: $f changed but CURRENT.md still lists $want"; rc=1
  else echo "ok: $f"; fi
done
[ $rc -eq 0 ] && echo "CURRENT.md matches the launchers." || echo "Fix: edit CURRENT.md to describe the change, then: bash tools/check-current.sh --write"
exit $rc
