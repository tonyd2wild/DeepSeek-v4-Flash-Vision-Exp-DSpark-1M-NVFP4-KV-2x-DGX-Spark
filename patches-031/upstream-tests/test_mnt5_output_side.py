# SPDX-License-Identifier: Apache-2.0
"""MNT-5 output-side: the guard is PENDING when the prompt does not end with
the reasoning start token, and the first visible output decides.

Ours, not part of vllm#52486 (test_stop_in_reasoning.py beside this file is
the PR's own). Runs against the patched patches-031/detokenizer.py.
"""

from test_stop_in_reasoning import _detokenizer, _feed, tokenizer  # noqa: F401


def test_output_opens_with_think_keeps_stops_dormant(tokenizer):
    """The model opens <think> itself: stops wait for </think>, then fire."""
    d = _detokenizer(tokenizer, prompt_ends_in_think=False, stop=["Question:"])
    out = tokenizer.encode("<think>Restating. Question: what is 2+2?",
                           add_special_tokens=False)
    assert _feed(d, out) is None, "stop fired inside model-opened reasoning"
    assert _feed(d, tokenizer.encode("</think>", add_special_tokens=False)) is None
    assert _feed(d, tokenizer.encode("It is 4. Question: next?",
                                     add_special_tokens=False)) == "Question:"


def test_output_opens_with_think_after_whitespace(tokenizer):
    """Leading template whitespace before the tag still counts as opening."""
    d = _detokenizer(tokenizer, prompt_ends_in_think=False, stop=["Question:"])
    out = tokenizer.encode("\n<think>Question: inside.", add_special_tokens=False)
    assert _feed(d, out) is None


def test_output_opens_in_content_evaluates_withheld_span(tokenizer):
    """Generation opens in content: the stop fires, including one in the
    very first chunk that was withheld while the guard was pending."""
    d = _detokenizer(tokenizer, prompt_ends_in_think=False, stop=["Question:"])
    one_chunk = tokenizer.encode("Question: first.", add_special_tokens=False)
    assert d.update(one_chunk, False) == "Question:"


def test_stop_withheld_while_still_pending(tokenizer):
    """While the first output is only whitespace the decision is still
    pending, and a stop must not fire on it: here the model emits a newline
    BEFORE opening <think>, with a newline stop. Fired early, the reasoning
    would never start; held, the guard arms when <think> arrives."""
    d = _detokenizer(tokenizer, prompt_ends_in_think=False, stop=["\n"])
    assert _feed(d, tokenizer.encode("\n", add_special_tokens=False)) is None, (
        "stop fired while the guard was still pending")
    assert _feed(d, tokenizer.encode("<think>a\nb", add_special_tokens=False)) is None
    assert _feed(d, tokenizer.encode("</think>", add_special_tokens=False)) is None
    assert _feed(d, tokenizer.encode("x\n", add_special_tokens=False)) == "\n"
