"""replay_hermes.py's cold loop and scorer, keeping each FULL reply (garble triage).

replay_hermes.py prints 90 characters per reply; a flagged reply needs reading in full
to tell real garble from a scorer false positive. Same env as replay_hermes.py, plus
OUT (jsonl of every reply, verdict and the regex match). COLD is always on."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_hermes as R
req = R.load_request()
base = {k: v for k, v in req.items() if k not in ("stream", "stream_options")}
base.update(stream=False, max_tokens=200)
out = open(os.environ.get("OUT", "/var/tmp/garble_synth/hermes-full.jsonl"), "w")
fails = 0
for i in range(R.N):
    b = dict(base); m0 = dict(b["messages"][0])
    m0["content"] = "[session %s-%d]\n" % (R.SALT, i) + m0["content"]
    b["messages"] = [m0] + b["messages"][1:]
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        R.URL + "/chat/completions", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    text = d["choices"][0]["message"].get("content") or ""
    v, why = R.score(text)
    m = R.CATALOG.search(text) or R.REGURG.search(text)
    fails += v == "FAIL"
    out.write(json.dumps({"i": i, "verdict": v, "why": why, "match": m.group(0) if m else None,
                          "text": text}) + "\n"); out.flush()
    print("%-5s %2d %5.1fs %-28s match=%r" % (v, i, time.time() - t0, why, m.group(0) if m else None), flush=True)
print("FAILURES %d/%d" % (fails, R.N))
