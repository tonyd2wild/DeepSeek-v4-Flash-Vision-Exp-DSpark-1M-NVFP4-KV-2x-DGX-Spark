# Real KV capacity, independent of how a vLLM version counts its boot-line "GPU KV cache size":
# one ~145K-token cold request with a 1,500-token reply, and the engine's own KV usage % while
# it decodes. capacity ~= total tokens / usage. Run it with nothing else on the engine, and
# read the usage BEFORE too: some builds keep finished requests' blocks counted as used.
# usage: tools/kv-capacity.sh <base url> <container name>
URL=$1; C=$2
cd "$(dirname "$0")/../benchmarks"
python3 - "$URL" > /tmp/kv-capacity.out 2>&1 <<'PY' &
import json, random, sys, urllib.request, os
sys.path.insert(0, "."); os.environ.setdefault("TOKENS", "200000")
import garble_synth as G
m = G.conversation(random.Random(99), "KV-MEASURE-1")
m[0] = dict(m[0], content="[kvm %d]\n" % os.getpid() + m[0]["content"])
m[-1] = {"role": "user", "content": "Now write a long, detailed 1200-word essay about log analysis."}
b = {"model": "deepseek-v4-flash-dspark", "messages": m, "max_tokens": 1500, "temperature": 0, "ignore_eos": True}
d = json.load(urllib.request.urlopen(urllib.request.Request(sys.argv[1] + "/v1/chat/completions",
    data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
print("usage", d["usage"])
PY
P=$!
T=$(date -u +%Y-%m-%dT%H:%M:%S)
while kill -0 $P 2>/dev/null; do sleep 10; done
cat /tmp/kv-capacity.out
docker logs --since $T $C 2>&1 | grep -oE "Running: 1 reqs.*GPU KV cache usage: [0-9.]+%" | sed -E 's/.*(GPU KV cache usage: [0-9.]+%)/\1/' | sort | uniq -c | tail -5
