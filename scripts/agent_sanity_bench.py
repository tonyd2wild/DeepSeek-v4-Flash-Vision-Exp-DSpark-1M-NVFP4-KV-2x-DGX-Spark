#!/usr/bin/env python3
"""Small OpenAI-compatible stability/concurrency check for agent endpoints."""

import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request


BASE_URL = os.environ.get("DSPARK_BASE_URL", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("DSPARK_MODEL", "deepseek-v4-flash-dspark")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
CONCURRENCY_LIST = [
    int(x) for x in os.environ.get("CONCURRENCY", "1,2,4,6").split(",") if x.strip()
]


def make_prompt(i: int) -> str:
    filler = " ".join(f"token{i}_{j}" for j in range(420))
    return (
        "Write a concise implementation note in plain English. "
        "Stay in English. Do not repeat characters. Do not output XML. "
        "Keep writing useful detail until the answer is complete.\n\n"
        f"Context salt {i}: {filler}"
    )


def looks_bad(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    repeated = any(ch * 18 in text for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    leaked = any(marker in text.lower() for marker in ("<available_skills", "<tool", "</tool", "<think>"))
    return cjk > 0 or repeated or leaked


def request_one(i: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": make_prompt(i)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=420) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    content = data["choices"][0]["message"].get("content") or ""
    completion = usage.get("completion_tokens") or 0
    return {
        "id": i,
        "seconds": round(dt, 3),
        "completion_tokens": completion,
        "tok_s": round(completion / dt, 2) if dt else 0,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "bad_output": looks_bad(content),
        "sample": content[:200],
    }


def run(concurrency: int) -> dict:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(request_one, range(concurrency)))
    wall = time.perf_counter() - start
    total = sum(r["completion_tokens"] for r in rows)
    return {
        "concurrency": concurrency,
        "success": f"{sum(not r['bad_output'] for r in rows)}/{len(rows)}",
        "max_tokens": MAX_TOKENS,
        "wall_seconds": round(wall, 3),
        "completion_tokens": total,
        "aggregate_tok_s": round(total / wall, 2) if wall else 0,
        "per_request_tok_s_mean": round(statistics.mean(r["tok_s"] for r in rows), 2),
        "bad_outputs": sum(1 for r in rows if r["bad_output"]),
        "rows": rows,
    }


def main() -> int:
    failed = False
    for concurrency in CONCURRENCY_LIST:
        result = run(concurrency)
        print(json.dumps(result, indent=2))
        failed = failed or result["bad_outputs"] > 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

