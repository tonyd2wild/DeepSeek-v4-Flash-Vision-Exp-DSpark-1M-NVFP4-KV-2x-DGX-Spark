#!/usr/bin/env python3
"""Sustained agent-shaped soak for DS4 issues #6 (soft-empty) and #8 (CUBLAS death).

#6 fingerprint: request returns 200 with completion_tokens > 0 but content parses EMPTY
   (or degenerates: repetition loop, CJK drift, template leakage). Reported at ~13-25/day
   under sustained agent load — so this needs hours and concurrency, not a 5-shot gate.
#8 fingerprint: engine dies (CUBLAS_STATUS_INTERNAL_ERROR); server stops answering.
   Hypothesis under test: it's device-memory pressure, so it should reproduce sooner at
   Dave's aggressive gmu 0.85 / seqs 12 than at our 0.78 / seqs 6.

Emits ONE LINE PER EVENT to stdout so it can drive a Monitor. Quiet otherwise except a
periodic heartbeat.

Env: URL, MODEL, CONC, MINUTES, TAG
"""
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

URL = os.environ.get("URL", "http://192.168.192.2:8889/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
CONC = int(os.environ.get("CONC", "4"))
MINUTES = float(os.environ.get("MINUTES", "45"))
TAG = os.environ.get("TAG", "soak")

# Agent-shaped traffic: tool-ish, code, long-context, structured output, reasoning.
PROMPTS = [
    ("tool", "You have a tool `search_inventory(query: str, limit: int)`. The user asks: "
             "'find me size 10 Jordan 4s under $300'. Emit the tool call, then explain."),
    ("code", "Refactor this into async with proper error handling and type hints:\n"
             "def fetch_all(urls):\n    out = []\n    for u in urls:\n        out.append(requests.get(u).json())\n    return out"),
    ("json", 'Return ONLY a JSON object: {"status":"ok","items":[...12 items, each '
             '{"sku":"...","qty":N,"price":F}...],"total":F}'),
    ("reason", "A warehouse has 3 zones. Zone A ships 40% of orders at 2.1 days, Zone B 35% "
               "at 1.8 days, Zone C the rest at 3.4 days. Compute weighted average delivery "
               "time, then explain which zone to expand and why."),
    ("long", "Summarize the tradeoffs between speculative decoding, tensor parallelism, and "
             "quantization for a 400B MoE served on two 128GB unified-memory nodes. "
             "Be specific about which lever affects step time vs acceptance."),
    ("chat", "Explain to a junior engineer why a burst of 80 tok/s can drop to 35 tok/s "
             "mid-generation on the same server with no config change."),
]

CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
TEMPLATE = re.compile(r"<\|.*?\|>|<｜.*?｜>|</?(?:think|tool_call|assistant)>", re.I)

stop = threading.Event()
lock = threading.Lock()
stats = {"n": 0, "empty": 0, "degen": 0, "err": 0, "tok": 0, "sec": 0.0}


def emit(msg):
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(label, text, ntok):
    """Return a reason string if this looks like issue #6, else None."""
    body = TEMPLATE.sub("", text).strip()
    if ntok > 0 and len(body) == 0:
        return f"SOFT-EMPTY: {ntok} tokens billed, content parses empty"
    if len(body) > 200:
        # repetition loop: same 40-char window many times
        w = body[-240:-200]
        if w and body.count(w) > 4:
            return f"REPETITION: 40-char window repeats {body.count(w)}x"
    cjk = len(CJK.findall(body))
    if cjk > 5 and label not in ("long",):
        return f"CJK-DRIFT: {cjk} CJK chars in an English response"
    if ntok > 0 and len(body) < ntok * 0.4:
        return f"SHORT-BODY: {len(body)} chars for {ntok} tokens (possible partial garble)"
    return None


def worker(wid):
    while not stop.is_set():
        label, prompt = random.choice(PROMPTS)
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": random.choice([300, 500, 800]),
                "temperature": random.choice([0.0, 0.3, 0.7])}
        req = urllib.request.Request(URL + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            dt = time.time() - t0
            msg = r["choices"][0]["message"]
            text = (msg.get("content") or "")
            # This runtime returns reasoning in `reasoning`; `reasoning_content` is
            # deprecated and only accepted on input. Reading only the old name makes
            # every thinking-mode response look empty. Keep the fallback for other runtimes.
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning:
                text += reasoning
            if msg.get("tool_calls"):
                text += json.dumps(msg["tool_calls"])
            ntok = r["usage"]["completion_tokens"]
            with lock:
                stats["n"] += 1; stats["tok"] += ntok; stats["sec"] += dt
            bad = check(label, text, ntok)
            if bad:
                with lock:
                    stats["empty" if "EMPTY" in bad else "degen"] += 1
                emit(f"ISSUE6 [{label}] {bad} | finish={r['choices'][0].get('finish_reason')} "
                     f"| head={text[:80]!r}")
        except Exception as e:
            with lock:
                stats["err"] += 1
            emit(f"ERROR [{label}] {type(e).__name__}: {str(e)[:140]}")
            time.sleep(2)


def heartbeat():
    t0 = time.time()
    while not stop.is_set():
        time.sleep(300)
        with lock:
            n, tk, sc = stats["n"], stats["tok"], stats["sec"]
            e, d, er = stats["empty"], stats["degen"], stats["err"]
        el = (time.time() - t0) / 60
        emit(f"HEARTBEAT {el:.0f}min | reqs={n} tok={tk} avg={tk/sc if sc else 0:.1f}tok/s "
             f"| soft-empty={e} degen={d} errors={er}")


if __name__ == "__main__":
    emit(f"SOAK START [{TAG}] {URL} conc={CONC} for {MINUTES}min")
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONC)]
    hb = threading.Thread(target=heartbeat, daemon=True)
    for t in ths: t.start()
    hb.start()
    try:
        time.sleep(MINUTES * 60)
    except KeyboardInterrupt:
        pass
    stop.set()
    time.sleep(3)
    n, tk, sc = stats["n"], stats["tok"], stats["sec"]
    emit(f"SOAK DONE [{TAG}] reqs={n} tokens={tk} avg={tk/sc if sc else 0:.1f}tok/s "
         f"| soft-empty={stats['empty']} degen={stats['degen']} errors={stats['err']}")
