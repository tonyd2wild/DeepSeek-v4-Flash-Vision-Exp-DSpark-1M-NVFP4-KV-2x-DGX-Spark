"""LOAD-1: lm_first keeps the one property upstream's sort exists for
(language_model.* as ONE contiguous run) and never holds an LM tensor back.
Pure; no vLLM needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dspark031_plugin"))
import dspark031 as D  # noqa: E402

FAIL = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL.append(name)


# checkpoint order as on disk: vision first, LM layers, an aligner and an image_ key
# interleaved, the head last
src = [("vision.blocks.1.w", 1), ("vision.blocks.0.w", 2),
       ("language_model.layers.0.a", 3), ("aligner.w1", 4),
       ("language_model.layers.0.b", 5), ("image_newline", 6),
       ("language_model.layers.1.a", 7), ("language_model.lm_head.weight", 8)]
out = list(D.lm_first(iter(src)))
names = [n for n, _ in out]
lm = [n for n in names if n.startswith("language_model.")]
check("nothing lost or duplicated", sorted(out) == sorted(src))
check("*** language_model.* is ONE contiguous run ***",
      names[:len(lm)] == lm and all(not n.startswith("language_model.") for n in names[len(lm):]))
check("...in arrival (checkpoint) order",
      lm == [n for n, _ in src if n.startswith("language_model.")])
check("the rest after, sorted by name", names[len(lm):] == sorted(n for n, _ in src
                                                             if not n.startswith("language_model.")))

# streaming: an LM item is yielded before the NEXT item is read
pulled = []


def tracked():
    for item in src:
        pulled.append(item[0])
        yield item


g = D.lm_first(tracked())
first = next(g)
check("*** the first LM weight comes out after reading only up to it (no buffering) ***",
      first[0] == "language_model.layers.0.a" and pulled == [n for n, _ in src[:3]])

print("%d failed" % len(FAIL))
sys.exit(1 if FAIL else 0)
