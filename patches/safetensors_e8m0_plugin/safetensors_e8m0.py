"""Teach safetensors' eager loader the F8_E8M0 dtype string.

WHY THIS EXISTS

The 64 KiB-page kernel that kindling spark-os boots makes a CUDA copy from a
file-backed mmap HANG. vLLM's default (lazy) safetensors loader does exactly
that copy for every tensor, so a model stalls at shard 0 with the GPU idle --
kindling's own AGENTS.md records it. The documented answer is
`--safetensors-load-strategy eager`, which reads each shard whole and copies
from anonymous memory instead.

That flag then fails on THIS checkpoint with `KeyError: 'F8_E8M0'`, because the
two strategies do not share a decoder:

    eager:   safetensors.torch.load(f.read())  -> Python, maps dtype STRINGS
             through safetensors.torch._TYPES
    default: safetensors.safe_open(...)        -> Rust, knows every dtype

`_TYPES` in this safetensors build carries only F8_E4M3 and F8_E5M2. It is a
safetensors-lagging-torch gap, not a configuration mistake: torch 2.11.0 DOES
have float8_e8m0fnu, and safetensors' own `_SIZE` table already lists it (1 byte
per element). Only the string -> dtype entry is missing.

It is not a corner case here. NVFP4 keeps its block scales in E8M0, so the first
60 shards of this checkpoint hold 44,259 F8_E8M0 tensors -- every `.scale`. Eager
can decode none of the model without this.

VERIFIED, AND NOT MERELY "IT LOADS". With the key added, shard 2 decodes 1,565
tensors and the F8_E8M0 ones come back BYTE-IDENTICAL to what the Rust
`safe_open` path returns -- same dtype, same shape, equal when viewed as uint8.
That is the check worth repeating if this is ever revisited: a dtype table fixed
by guessing would produce plausible tensors of the wrong scale, and a model that
loads and generates subtly wrong output is far worse than one that refuses to
start.

DELIVERY. A vLLM general plugin, discovered through the entry-point group
`vllm.general_plugins`, like any vLLM general plugin. That keeps it off the whole-file mount list:
patching site-packages/safetensors/torch.py would pin us to one safetensors
build forever, which is how this repo ended up on an August engine in October.
Plugins load in each worker BEFORE load_model, which is what makes this work --
observed in the logs, where the plugin's line precedes the weight load.

REMOVING IT. Harmless to leave once safetensors ships the key: the update is
conditional on absence and logs when it finds nothing to do. Drop it when
`"F8_E8M0" in safetensors.torch._TYPES` is already true on a fresh image.
"""

import logging

logger = logging.getLogger(__name__)

_DTYPE_STR = "F8_E8M0"
_TORCH_ATTR = "float8_e8m0fnu"


def register() -> None:
    """Add the missing F8_E8M0 entry to safetensors.torch._TYPES.

    Fails LOUDLY rather than degrading. A silent no-op here means the engine
    falls back to `KeyError: 'F8_E8M0'` thousands of tensors into model load,
    where the cause is far from the symptom.
    """
    import torch

    try:
        import safetensors.torch as st
    except Exception as e:  # pragma: no cover - safetensors is a hard dep of vLLM
        raise RuntimeError(
            "safetensors_e8m0 plugin: cannot import safetensors.torch"
        ) from e

    types = getattr(st, "_TYPES", None)
    if types is None:
        raise RuntimeError(
            "safetensors_e8m0 plugin: safetensors.torch has no _TYPES table. "
            "The eager loader has been restructured; re-derive this plugin "
            "against the new code rather than deleting it -- the 64k kernel "
            "still needs eager loading."
        )

    if _DTYPE_STR in types:
        logger.info(
            "safetensors_e8m0: %s already mapped; nothing to do "
            "(safetensors has caught up -- this plugin can be removed)",
            _DTYPE_STR,
        )
        return

    dtype = getattr(torch, _TORCH_ATTR, None)
    if dtype is None:
        raise RuntimeError(
            f"safetensors_e8m0 plugin: torch {torch.__version__} has no "
            f"torch.{_TORCH_ATTR}, so the eager safetensors loader cannot decode "
            f"{_DTYPE_STR} and this checkpoint's NVFP4 scales are unreadable. "
            "Either use a torch that has it, or boot the 4k kernel flavour where "
            "the default lazy loader works."
        )

    types[_DTYPE_STR] = dtype
    logger.info(
        "safetensors_e8m0: mapped %s -> torch.%s for the eager safetensors "
        "loader (needed by NVFP4 block scales on the 64k-page kernel)",
        _DTYPE_STR,
        _TORCH_ATTR,
    )
