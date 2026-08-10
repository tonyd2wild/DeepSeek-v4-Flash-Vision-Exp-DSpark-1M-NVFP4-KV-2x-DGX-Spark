#!/usr/bin/env python3
"""garble_tap — transparent recording proxy for a vLLM OpenAI endpoint.

Point Hermes/OpenClaw at this instead of the lane. It forwards everything unchanged and
records each request + response, flagging the garble signature so the EXACT prompt that
produced bad output is captured for deterministic replay.

Why: five-prompt gates pass while real agent traffic garbles. Synthetic approximations of
Hermes have not reproduced it. The only reliable path is the real prompt.

  listen  0.0.0.0:PORT           (default 8890)
  upstream UPSTREAM              (default http://192.168.192.1:8888)

  /var/tmp/garble_tap/all.jsonl      every exchange (prompt + completion + timing)
  /var/tmp/garble_tap/FLAGGED.jsonl  only exchanges whose output tripped a detector
  stdout                             one line per flagged event (drives a Monitor)

Handles streaming (SSE) as well as non-streaming, because agents usually stream and the
failure must be observable either way.
"""
import json
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://192.168.192.1:8888").rstrip("/")
PORT = int(os.environ.get("PORT", "8890"))
OUTDIR = os.environ.get("OUTDIR", "/var/tmp/garble_tap")
os.makedirs(OUTDIR, exist_ok=True)
ALL = os.path.join(OUTDIR, "all.jsonl")
FLAGGED = os.path.join(OUTDIR, "FLAGGED.jsonl")

# DeepSeek special tokens in both ASCII and full-width forms, plus turn markers.
SPECIALS = re.compile(
    r"<\|?begin[_▁]of[_▁]sentence\|?>|<｜begin▁of▁sentence｜>|"
    r"<\|?end[_▁]of[_▁]sentence\|?>|<｜end▁of▁sentence｜>|"
    r"<\|?User\|?>|<｜User｜>|<\|?Assistant\|?>|<｜Assistant｜>|"
    r"<\|tool[▁_][a-z]+\|>|<｜tool▁[a-z]+｜>"
)
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
lock = threading.Lock()


def detect(text, n_out):
    """Return list of reasons this output looks wrong."""
    bad = []
    hits = set(SPECIALS.findall(text))
    if hits:
        bad.append(f"special-token-leak:{sorted(hits)}")
    if n_out and not text.strip():
        bad.append("empty-with-tokens-billed")
    body = text.strip()
    if len(body) > 240:
        w = body[-240:-200]
        if w and body.count(w) > 4:
            bad.append(f"repetition-loop:x{body.count(w)}")
    cjk = len(CJK.findall(body))
    if cjk > 5 and cjk > len(body) * 0.05:
        bad.append(f"cjk-drift:{cjk}")
    # document-continuation signature: reply opens as a markdown heading
    if re.match(r"^\s*#{1,6}\s", body):
        bad.append("opens-as-markdown-heading")
    return bad


def record(path, obj):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # keep stdout clean; it is the event stream

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=raw if raw else None, method=method)
        for h, v in self.headers.items():
            if h.lower() not in ("host", "content-length", "connection"):
                req.add_header(h, v)

        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        streaming = bool(body.get("stream"))
        t0 = time.time()

        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            payload = json.dumps({"error": f"tap upstream error: {e}"}).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(resp.status)
        for h, v in resp.getheaders():
            if h.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(h, v)
        if streaming:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        collected = ""
        n_out = 0

        if streaming:
            for line in resp:
                # pass through verbatim, chunked
                self.wfile.write(hex(len(line))[2:].encode() + b"\r\n" + line + b"\r\n")
                self.wfile.flush()
                s = line.decode("utf-8", "replace").strip()
                if s.startswith("data:"):
                    d = s[5:].strip()
                    if d and d != "[DONE]":
                        try:
                            j = json.loads(d)
                            ch = j.get("choices", [{}])[0]
                            delta = ch.get("delta") or {}
                            collected += (delta.get("content") or "")
                            collected += (delta.get("reasoning") or delta.get("reasoning_content") or "")
                        except Exception:
                            pass
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            n_out = len(collected.split())
        else:
            data = resp.read()
            self.wfile.write(data)
            try:
                j = json.loads(data)
                ch = j.get("choices", [{}])[0]
                msg = ch.get("message") or {}
                collected = (msg.get("content") or "")
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                if reasoning:
                    collected += reasoning
                if msg.get("tool_calls"):
                    collected += json.dumps(msg["tool_calls"])
                n_out = (j.get("usage") or {}).get("completion_tokens", 0)
            except Exception:
                pass

        if "chat/completions" not in self.path and "completions" not in self.path:
            return

        dt = time.time() - t0
        bad = detect(collected, n_out)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "path": self.path,
            "streaming": streaming,
            "model": body.get("model"),
            "n_messages": len(body.get("messages") or []),
            "has_tools": bool(body.get("tools")),
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "seconds": round(dt, 2),
            "out_tokens": n_out,
            "flags": bad,
            "request": body,          # FULL prompt, for deterministic replay
            "output": collected,
        }
        record(ALL, entry)
        if bad:
            record(FLAGGED, entry)
            print(f"[{entry['ts']}] GARBLE {bad} | msgs={entry['n_messages']} "
                  f"tools={entry['has_tools']} out={n_out} | head={collected[:120]!r}",
                  flush=True)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


if __name__ == "__main__":
    print(f"garble_tap listening on :{PORT} -> {UPSTREAM}", flush=True)
    print(f"  all exchanges  : {ALL}", flush=True)
    print(f"  flagged only   : {FLAGGED}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
