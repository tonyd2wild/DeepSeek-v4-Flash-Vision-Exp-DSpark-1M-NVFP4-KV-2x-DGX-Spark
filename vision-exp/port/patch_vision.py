#!/usr/bin/env python3
"""Patch vLLM's vendored deepseek_v4 model.py to accept the Vision-Exp checkpoint.

DeepSeek-V4-Flash-Vision-Exp adds a 32-block ViT, a 2-layer aligner, and four
special embeddings on top of DeepSeek-V4-Flash-0731. vLLM's DeepseekV4ForCausalLM
is the text-only class, so loading the vision checkpoint dies with:

    ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM

This makes three edits, all idempotent:
  1. import the ported vision tower
  2. build `vision` / `aligner` / `image_{start,end,newline,pad}` on DeepseekV4Model
     when config.vision_n_layers > 0
  3. teach the weights mapper the four new checkpoint prefixes

Usage: patch_vision.py <path-to-model.py>
"""
import re
import sys

IMPORT_ANCHOR = "def _env_flag(*names: str) -> bool:"
IMPORT_BLOCK = '''# --- DeepSeek-V4-Flash-Vision-Exp: vision tower + aligner -------------------
try:
    from vllm.models.deepseek_v4.nvidia.ds4v_vision import build_vision_modules
except Exception:  # pragma: no cover - text-only checkpoints must still import
    build_vision_modules = None


def _ds4v_image_token_id(vllm_config, default: int = 129264) -> int:
    """Resolve `<|deepseek_image|>` to its token id.

    Looked up from the real tokenizer so this survives a vocab change; the
    default is the id in the 2026-08-31 Vision-Exp release, used only if the
    tokenizer cannot be loaded here.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer, trust_remote_code=True
        )
        tid = tok.convert_tokens_to_ids("<｜deepseek_image｜>")
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    return default
# ---------------------------------------------------------------------------


'''

INIT_ANCHOR = "        self.hc_head_op = HCHeadOp()"
INIT_BLOCK = """
        # --- Vision-Exp: tower + aligner + the four special embeddings ------
        # Present only when the checkpoint carries them, so this file still
        # loads DeepSeek-V4-Flash-0731 (text-only) unchanged.
        self.vision = None
        self.aligner = None
        n_vision_layers = int(getattr(config, "vision_n_layers", 0) or 0)
        if n_vision_layers > 0:
            if build_vision_modules is None:
                raise RuntimeError(
                    "checkpoint declares vision_n_layers=%d but ds4v_vision "
                    "could not be imported" % n_vision_layers
                )
            self.vision, self.aligner = build_vision_modules(config)
            # Not sharded: the tower is ~410M params and the reference runs it
            # on one device. Replicating beats an all-gather here.
            for _name in ("image_start", "image_end", "image_newline", "image_pad"):
                setattr(
                    self,
                    _name,
                    nn.Parameter(
                        torch.empty(config.hidden_size,
                                    dtype=vllm_config.model_config.dtype),
                        requires_grad=False,
                    ),
                )
            self.max_image_tokens = int(getattr(config, "vision_max_n_token", 384))
            # The DSpark proposer (and vLLM's VLM plumbing generally) expects the
            # standard `image_token_index` field once a model reports as
            # multimodal. DeepSeek's config doesn't carry one, so publish it here
            # from the tokenizer's `<|deepseek_image|>` id.
            if not hasattr(config, "image_token_index"):
                config.image_token_index = _ds4v_image_token_id(vllm_config)
        # --------------------------------------------------------------------
"""

# 4. keep the MLP stacked-params mapping off the vision/aligner weights.
# The mapping rewrites any name containing ".w1"/".w3" into "gate_up_proj" for
# fused MLPs. The ported ViT MLP and the aligner both legitimately use w1/w2,
# so without this guard `aligner.w1.bias` becomes `aligner.gate_up_proj.bias`
# and load_weights dies with a KeyError. Falling through to the generic
# `else:` branch loads them with default_weight_loader, which is correct —
# neither module is sharded.
LOADER_ANCHOR = """            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below)."""
LOADER_BLOCK = """            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Vision-Exp: the ViT MLP and the aligner use w1/w2 names that
                # collide with the fused-MLP mapping below. Neither is sharded,
                # so let them fall through to the generic loader.
                if name.startswith(("vision.", "aligner.")):
                    continue
                # Skip non-stacked layers and experts (experts handled below)."""

# 5. Vision-Exp's MoE gate carries a routing-correction bias on EVERY layer —
# including the first `num_hash_layers` hash-MoE layers, which 0731 did not have
# and which this file explicitly skips ("hash MoE doesn't use
# e_score_correction_bias"). It also adds a SECOND bias, `bias_vl`, applied to
# vision tokens, so experts are routed differently for image vs text tokens.
# Both are new in this release and have no vLLM equivalent.
GATE_ANCHOR = """        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )"""
GATE_BLOCK = """        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        # --- Vision-Exp gate biases ----------------------------------------
        # Guarded on vision_n_layers so text-only DeepSeek-V4-Flash-0731 keeps
        # exactly its previous behaviour (no bias on hash layers, no bias_vl).
        self.gate.e_score_correction_bias_vl = None
        if int(getattr(config, "vision_n_layers", 0) or 0) > 0:
            if self.gate.e_score_correction_bias is None:
                # hash-MoE layers now carry one too
                self.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32),
                    requires_grad=False,
                )
            self.gate.e_score_correction_bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        # --------------------------------------------------------------------"""

MAPPER_ANCHOR = '            "mtp.": "model.mtp.",'
MAPPER_BLOCK = '''            "mtp.": "model.mtp.",
            # Vision-Exp additions
            "vision.": "model.vision.",
            "aligner.": "model.aligner.",
            "image_start": "model.image_start",
            "image_end": "model.image_end",
            "image_newline": "model.image_newline",
            "image_pad": "model.image_pad",'''


# 7 + 8. Declare multimodal support and wire the image path.
# vLLM's serving layer rejects images with "is not a multimodal model" unless the
# class implements SupportsMultiModal AND a processor is registered for it.
MM_CLASS_ANCHOR = """class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model"""
MM_CLASS_BLOCK = '''@MULTIMODAL_REGISTRY.register_processor(
    DS4VMultiModalProcessor,
    info=DS4VProcessingInfo,
    dummy_inputs=DS4VDummyInputsBuilder,
)
class DeepseekV4ForCausalLM(nn.Module, SupportsPP, SupportsMultiModal):
    model_cls = DeepseekV4Model

    # DeepSeek-V4's first `num_hash_layers` MoE layers route experts by TOKEN ID
    # (gate.tid2eid), not by hidden state. vLLM's multimodal path normally hands
    # the model inputs_embeds with input_ids=None, which makes hash routing
    # raise "DeepSeek V4 hash MoE routing requires input_ids." This flag keeps
    # the raw token ids alongside the embeddings.
    requires_raw_input_tokens: bool = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return DS4V_IMAGE_PLACEHOLDER
        return None

    def get_language_model(self):
        """Return self, not a sub-module.

        Standard vLLM VLMs keep the vision tower as a sibling of a separate
        `language_model`. DeepSeek-V4-Flash-Vision-Exp instead carries the ViT
        and aligner INSIDE the decoder stack, so the language model and this
        wrapper are the same object. Callers (notably the DSpark proposer, which
        does `get_language_model().model` / `.lm_head` to share embeddings and
        the LM head) need `self` for those lookups to resolve.
        """
        return self

    def embed_multimodal(self, **kwargs: object):
        """Run the ViT + aligner and return one assembled block per image.

        The whole block is returned — aligner outputs at IMAGE slots and the
        learned image_start / image_pad / image_newline / image_end vectors at
        the others — because every slot was emitted as a placeholder token.
        """
        patches = kwargs.get("image_patches")
        grids = kwargs.get("image_grid")
        if patches is None or grids is None:
            return []
        if self.model.vision is None:
            raise RuntimeError(
                "image input received but this checkpoint has no vision tower"
            )
        # `flat_from_sizes` hands back ONE concatenated [sum(n_patches),3,p,p]
        # tensor, not a per-image list. Iterating it directly walks individual
        # patches and feeds the patch embedder a [3,196] row instead of an
        # [N,588] block ("mat1 and mat2 shapes cannot be multiplied").
        if isinstance(patches, torch.Tensor) and patches.dim() == 4:
            counts = kwargs.get("num_patches")
            if counts is not None:
                sizes = [int(c) for c in torch.as_tensor(counts).flatten().tolist()]
                if sum(sizes) == patches.shape[0]:
                    patches = list(torch.split(patches, sizes, dim=0))
            if isinstance(patches, torch.Tensor):
                # single image
                patches = [patches]
        if isinstance(grids, torch.Tensor) and grids.dim() == 2:
            grids = list(grids)
        blocks = []
        for p, g in zip(patches, grids):
            g = g.flatten()
            # grid = (n_vit_h, n_vit_w, n_llm_h, n_llm_w)
            blocks.append(
                ds4v_assemble_image_block(
                    self.model, p, int(g[0]), int(g[1]), int(g[2]), int(g[3])
                )
            )
        return blocks'''

MM_IMPORT_ANCHOR = "# --- DeepSeek-V4-Flash-Vision-Exp: vision tower + aligner -------------------"
MM_IMPORT_BLOCK = """# --- DeepSeek-V4-Flash-Vision-Exp: multimodal plumbing ----------------------
try:
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.interfaces import SupportsMultiModal
    from vllm.models.deepseek_v4.nvidia.ds4v_mm import (
        IMAGE_PLACEHOLDER as DS4V_IMAGE_PLACEHOLDER,
        DS4VDummyInputsBuilder,
        DS4VMultiModalProcessor,
        DS4VProcessingInfo,
        assemble_image_block as ds4v_assemble_image_block,
    )
except Exception as _ds4v_mm_err:  # pragma: no cover
    raise
# ---------------------------------------------------------------------------

# --- DeepSeek-V4-Flash-Vision-Exp: vision tower + aligner -------------------"""

# 10. Make embed_input_ids multimodal-aware.
# The existing one-arg version silently ignores multimodal embeddings, so images
# would never be spliced in. It must also NOT route text embedding through
# get_language_model() — that returns self here, which would recurse forever.
EMBED_ANCHOR = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)"""
EMBED_BLOCK = '''    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed tokens, splicing image-block embeddings at placeholder slots.

        Text embedding goes straight to `self.model.embed_input_ids` rather than
        `self.get_language_model()` — the latter returns `self` for this model,
        which would recurse.
        """
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError(
                "is_multimodal mask is required when multimodal embeddings are "
                "provided"
            )
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )'''

SUFFIX_ANCHOR = '            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",'
SUFFIX_BLOCK = '''            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".ffn.gate.bias_vl": ".ffn.gate.e_score_correction_bias_vl",'''


def main(path: str) -> int:
    src = open(path).read()

    if "ds4v_vision" in src:
        print("already patched; nothing to do")
        return 0

    for anchor, name in ((IMPORT_ANCHOR, "import"),
                         (INIT_ANCHOR, "__init__"),
                         (LOADER_ANCHOR, "load_weights"),
                         (MAPPER_ANCHOR, "mapper"),
                         (SUFFIX_ANCHOR, "suffix"),
                         (GATE_ANCHOR, "gate"),
                         # MM_IMPORT_ANCHOR is created by edit 1, so it cannot be
                         # validated against the unpatched source.
                         (MM_CLASS_ANCHOR, "mm_class"),
                         (EMBED_ANCHOR, "embed_input_ids")):
        if src.count(anchor) != 1:
            print(f"ERROR: {name} anchor found {src.count(anchor)} times, expected 1",
                  file=sys.stderr)
            return 2

    src = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK + IMPORT_ANCHOR, 1)
    src = src.replace(INIT_ANCHOR, INIT_ANCHOR + INIT_BLOCK, 1)
    src = src.replace(LOADER_ANCHOR, LOADER_BLOCK, 1)
    src = src.replace(MAPPER_ANCHOR, MAPPER_BLOCK, 1)
    src = src.replace(SUFFIX_ANCHOR, SUFFIX_BLOCK, 1)
    src = src.replace(GATE_ANCHOR, GATE_BLOCK, 1)
    src = src.replace(MM_IMPORT_ANCHOR, MM_IMPORT_BLOCK, 1)
    src = src.replace(MM_CLASS_ANCHOR, MM_CLASS_BLOCK, 1)
    src = src.replace(EMBED_ANCHOR, EMBED_BLOCK, 1)

    open(path, "w").write(src)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
