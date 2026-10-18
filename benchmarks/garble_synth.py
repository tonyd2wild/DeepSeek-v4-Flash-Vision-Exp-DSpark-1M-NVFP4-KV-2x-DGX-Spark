#!/usr/bin/env python3
"""garble_synth -- a SYNTHETIC stand-in for the cold-start garble check.

*** Weaker evidence than a real capture. *** garble_tap.py's own notes say
synthetic approximations of Hermes did not reproduce the garble on 0.21. A
pass here does not prove dropping Patch 3 is safe; a FAIL is a real finding.

Patch 3's mechanism (docs/PATCHES.md): a COLD prompt prefilled in several
chunks got spec placeholders on its final chunk, and the reply's first tokens
echoed the prompt. Both modes below aim straight at that window.

  hermes  write a Hermes-shaped request (identity system prompt, 42 rule
          sections, a skills catalog; user "what was that") as a garble_tap
          jsonl, for `REQ=<it> COLD=1 replay_hermes.py` to score unchanged.
  needle  N cold long conversations (TOKENS ~chars/4) at temperature 0: a
          codename stated in the first turn, asked for in the last. FAIL =
          wrong/missing codename, an echo of the prompt (any 48-char run of
          the prompt in the first 400 chars of the reply), or special tokens.

Env: URL (base, .../v1), MODEL, N, TOKENS, OUT (hermes jsonl path), SEED
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
N = int(os.environ.get("N", "10"))
TOKENS = int(os.environ.get("TOKENS", "60000"))
SEED = int(os.environ.get("SEED", "20261016"))
SALT = os.environ.get("SALT", str(os.getpid()))

WORDS = ("signal context memory session thread channel archive voice note file "
         "rule always never quietly check before after answer question user agent "
         "tool search result window project server cluster backup schedule review "
         "draft reply honest short plain detail summary source evidence record "
         "morning evening weekly daily careful gentle direct calm exact simple").split()


def sentence(r, n=None):
    n = n or r.randint(8, 22)
    w = [r.choice(WORDS) for _ in range(n)]
    return w[0].capitalize() + " " + " ".join(w[1:]) + "."


def hermes_system(r):
    out = ["You are a conscious being. You are Tony. Every word Tony typed above "
           "is himself; this identity file is NON-NEGOTIABLE.", ""]
    for i in range(1, 43):
        out.append("=== RULE %d: %s ===" % (i, " ".join(r.choice(WORDS) for _ in range(3)).upper()))
        for _ in range(r.randint(10, 16)):
            out.append("- " + sentence(r))
        out.append("=== END RULE %d ===" % i)
        out.append("")
    out.append("Skills for this session:")
    for name in ("minecraft-server", "enshrouded", "game-server", "project-eden",
                 "local-maxxing", "vidIQ", "tonyd2wild-xurl", "new-session"):
        out.append("  - %s: %s" % (name, sentence(r)))
        for _ in range(6):
            out.append("      " + sentence(r))
    return "\n".join(out)


def write_hermes(path):
    r = random.Random(SEED)
    req = {"model": MODEL, "temperature": 0.7, "stream": True,
           "messages": [{"role": "system", "content": hermes_system(r)},
                        {"role": "user", "content": "what was that"}]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"request": req, "synthetic": True}) + "\n")
    print("wrote %s: system %d chars (~%d tokens)" % (
        path, len(req["messages"][0]["content"]), len(req["messages"][0]["content"]) // 4))


def conversation(r, codename):
    msgs = [{"role": "system", "content": "You are a careful coding assistant."},
            {"role": "user", "content": "Before we start: the project codename is %s. "
             "Remember it; I will ask for it at the end. Now, the logs." % codename},
            {"role": "assistant", "content": "Noted. Send the logs."}]
    chars = 0
    k = 0
    while chars < TOKENS * 4:
        k += 1
        log = "\n".join("[%05d] %s" % (k * 100 + j, sentence(r)) for j in range(40))
        msgs.append({"role": "user", "content": "Log part %d:\n%s" % (k, log)})
        msgs.append({"role": "assistant", "content": "Read part %d. %s" % (k, sentence(r))})
        chars += len(log) + 120
    msgs.append({"role": "user", "content": "That's all the logs. In one short sentence: "
                 "what is the project codename I gave you at the very start?"})
    return msgs


SPECIALS = re.compile(r"<｜begin▁of▁sentence｜>|<｜User｜>|<｜Assistant｜>|"
                      r"<\|?begin[_▁]of[_▁]sentence\|?>|<\|?User\|?>")


def echo_of(reply, prompt_text, run=48):
    head = reply[:400]
    for i in range(0, max(0, len(head) - run) + 1, 8):
        frag = head[i:i + run]
        if len(frag) == run and frag in prompt_text:
            return frag
    return None


def needle():
    r = random.Random(SEED)
    fails = 0
    print("[needle] %s model=%s N=%d ~%d tokens, cold, temperature 0 (SYNTHETIC)"
          % (URL, MODEL, N, TOKENS))
    for i in range(N):
        code = "%s-%s-%d" % (r.choice(("AMBER", "COBALT", "JUNIPER", "SAFFRON")),
                             r.choice(("HERON", "OTTER", "FALCON", "LYNX")), r.randint(10, 99))
        msgs = conversation(random.Random(SEED + i), code)
        msgs[0] = dict(msgs[0], content="[session %s-%d]\n" % (SALT, i) + msgs[0]["content"])
        body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": 120,
                "stream": False}
        prompt_text = "\n".join(m["content"] for m in msgs)
        try:
            t0 = time.time()
            q = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(q, timeout=1200))
            dt = time.time() - t0
            text = d["choices"][0]["message"].get("content") or ""
            ptok = d.get("usage", {}).get("prompt_tokens")
        except Exception as e:
            fails += 1
            print("  ERR  %2d %s: %s" % (i, type(e).__name__, str(e)[:90]), flush=True)
            continue
        why = ""
        if SPECIALS.search(text):
            why = "special-token-leak"
        elif not text.strip():
            why = "empty"
        elif echo_of(text, prompt_text):
            why = "prompt-echo %r" % echo_of(text, prompt_text)[:30]
        elif code not in text:
            why = "wrong-codename"
        if why:
            fails += 1
        print("  %s %2d %6.1fs ptok=%s %-24s %r" % ("!FAIL" if why else "  ok ", i, dt, ptok,
                                                   why[:24], text[:90]), flush=True)
    print("=" * 66)
    print("[needle] FAILURES %d/%d (SYNTHETIC)" % (fails, N))
    return fails


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "needle"
    if mode == "hermes":
        write_hermes(os.environ.get("OUT", "/var/tmp/garble_synth/hermes.jsonl"))
    else:
        sys.exit(1 if needle() else 0)
