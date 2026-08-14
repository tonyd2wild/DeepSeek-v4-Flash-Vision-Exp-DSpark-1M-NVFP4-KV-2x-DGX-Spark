#!/usr/bin/env python3
"""Patch 5 port: suppress client stop strings inside the reasoning segment.

Port of tonyd2wild DSpark 0005-suppress-stops-in-reasoning (commit 3ba6ee21,
PR #21) from vLLM 0.21.1 to this image (0.25.2.dev). Mechanism: if the request
prompt ends with the reasoning start token (<think>, per think-in-prompt
templates), client stop strings stay dormant until the end marker appears in
OUTPUT; on close, stop checking resumes only past the marker (spec decode
delivers k+1 tokens per update, so the closing chunk carries reasoning tail).
EOS/max_tokens unaffected; non-thinking requests untouched.
Opt-out: VLLM_SUPPRESS_STOPS_IN_REASONING=0 (process-wide).

Runs at container start BEFORE vllm serve. Idempotent. Exits 0 ALWAYS (a hard
exit would kill the boot); prints PATCH5-APPLIED / PATCH5-FAILED - the
post-boot verification greps for the marker, which is where loudness lives.
"""
import sys

P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py"

HELPERS = '''
def _p5_reasoning_markers():
    # PATCH(stop-in-reasoning): prefer the deployment reasoning-config markers;
    # fall back to the common defaults (which this fleet also configures).
    start, end = "<think>", "</think>"
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        rc = getattr(cfg, "reasoning_config", None) if cfg else None
        if rc is not None:
            start = getattr(rc, "reasoning_start_str", "") or start
            end = getattr(rc, "reasoning_end_str", "") or end
    except Exception as e:
        logger.debug("stop-in-reasoning: no reasoning config (%s); using %r/%r",
                     e, start, end)
    return start, end


def _p5_enable_reasoning_guard(detok, tokenizer, request):
    # PATCH(stop-in-reasoning): arm the guard when the prompt ends inside the
    # reasoning segment. Failures log at debug, never raise - but they also
    # never silently strand the fix: post-boot verification greps this file.
    if os.environ.get("VLLM_SUPPRESS_STOPS_IN_REASONING", "1") == "0":
        return
    try:
        stop = getattr(detok, "stop", None)
        ptids = getattr(request, "prompt_token_ids", None)
        if not stop or not ptids:
            return
        start_str, end_str = _p5_reasoning_markers()
        think_id = None
        try:
            think_id = tokenizer.convert_tokens_to_ids(start_str)
        except Exception:
            pass
        if think_id is None or (isinstance(think_id, int) and think_id < 0):
            try:
                enc = tokenizer.encode(start_str)
                if isinstance(enc, list) and len(enc) == 1:
                    think_id = enc[0]
            except Exception:
                pass
        if isinstance(think_id, int) and think_id >= 0 and ptids[-1] == think_id:
            detok._reasoning_stop_guard = True
            detok._reasoning_end_str = end_str
    except Exception as e:
        logger.debug("stop-in-reasoning: guard not armed (%s)", e)

'''

FACTORY_OLD = """        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)"""

FACTORY_NEW = """        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            detok = FastIncrementalDetokenizer(tokenizer, request)
        else:
            # Fall back to slow python-based incremental detokenization.
            detok = SlowIncrementalDetokenizer(tokenizer, request)
        _p5_enable_reasoning_guard(detok, tokenizer, request)
        return detok"""

INIT_OLD = "        self._last_output_text_offset: int = 0\n"

INIT_NEW = """        self._last_output_text_offset: int = 0

        # PATCH(stop-in-reasoning): stops stay dormant while reasoning is open.
        self._reasoning_stop_guard: bool = False
        self._reasoning_closed: bool = False
        self._reasoning_end_str: str = "</think>"
"""

STOP_OLD = """        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:"""

STOP_NEW = """        # 2) Evaluate stop strings.
        # PATCH(stop-in-reasoning): keep stops dormant while reasoning is open.
        if self._reasoning_stop_guard and not self._reasoning_closed:
            _p5_marker = self._reasoning_end_str
            _p5_window = max(0, stop_check_offset - (len(_p5_marker) - 1))
            _p5_idx = self.output_text.find(_p5_marker, _p5_window)
            if _p5_idx != -1:
                self._reasoning_closed = True
                # Spec decode: this same update() may carry the reasoning tail
                # that precedes the close; stops must only see what FOLLOWS.
                stop_check_offset = max(stop_check_offset, _p5_idx + len(_p5_marker))
        stop_string = None
        if (
            self.stop
            and self.num_output_tokens() > self.min_tokens
            and (not self._reasoning_stop_guard or self._reasoning_closed)
        ):"""


def main():
    s = open(P).read()
    if "_reasoning_stop_guard" in s:
        print("PATCH5-APPLIED (already)")
        return

    subs = [
        ("from abc import ABC, abstractmethod",
         "import os\nfrom abc import ABC, abstractmethod"),
        (FACTORY_OLD, FACTORY_NEW),
        ("class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):",
         HELPERS + "\nclass BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):"),
        (INIT_OLD, INIT_NEW),
        (STOP_OLD, STOP_NEW),
    ]

    for old, _new in subs:
        if old not in s:
            print("PATCH5-FAILED: anchor missing: " + old[:70].replace("\n", "\\n"))
            return
    for old, new in subs:
        s = s.replace(old, new, 1)
    compile(s, P, "exec")
    open(P, "w").write(s)
    print("PATCH5-APPLIED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("PATCH5-FAILED: %s" % e)
    sys.exit(0)
