#!/usr/bin/env python3
"""Register a multimodal architecture alias for DeepSeek-V4-Flash-Vision-Exp.

vLLM answers `ModelConfig.is_multimodal_model` from a STATIC table keyed by
architecture name — not by inspecting the class. `DeepseekV4ForCausalLM` lives in
`_TEXT_GENERATION_MODELS`, so even with `SupportsMultiModal` in its MRO and a
processor registered, the API server still refuses images with:

    "<model> is not a multimodal model"

The Vision-Exp checkpoint reports the SAME `architectures` string as text-only
0731, so the name cannot simply be moved — that would break 0731. Instead we add
an ALIAS in `_MULTIMODAL_MODELS` pointing at the same (patched) class, and select
it for the vision checkpoint with:

    --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"]}'

Text-only 0731 keeps resolving through the untouched text-generation entry.
This mirrors the approach taken upstream in vllm-project/vllm#54561 / #54566.

Usage: patch_registry.py <path-to-registry.py>
"""
import sys

ALIAS = "DeepseekV4VForConditionalGeneration"
ANCHOR = "_MULTIMODAL_MODELS = {\n    # [Decoder-only]"
BLOCK = (
    "_MULTIMODAL_MODELS = {\n"
    "    # [Decoder-only]\n"
    "    # DeepSeek-V4-Flash-Vision-Exp. Alias onto the same class as the text-only\n"
    "    # entry: the checkpoint reports architectures=['DeepseekV4ForCausalLM'],\n"
    "    # so it is selected via --hf-overrides. See patch_registry.py docstring.\n"
    f'    "{ALIAS}": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),'
)


def main(path: str) -> int:
    src = open(path).read()
    if ALIAS in src:
        print("already patched")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"ERROR: anchor found {src.count(ANCHOR)} times, expected 1", file=sys.stderr)
        return 2
    open(path, "w").write(src.replace(ANCHOR, BLOCK, 1))
    print(f"patched {path}: registered {ALIAS} as multimodal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
