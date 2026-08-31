"""Multimodal plumbing for DeepSeek-V4-Flash-Vision-Exp in vLLM.

The checkpoint ships its own preprocessing (`inference/image_processor.py`) and
no HuggingFace processor, so the pipeline here mirrors PixtralHF: override
`_call_hf_processor` and do the work ourselves.

Design note — why every block slot is a placeholder token:
    The reference builds an image block containing four kinds of slot:
    IMAGE (aligner output), plus IMAGE_START / IMAGE_PAD / IMAGE_NEWLINE /
    IMAGE_END, which are *learned parameter vectors*, not vocabulary entries.
    Rather than reproduce the reference's out-of-vocab `vocab_size + type`
    sentinels, we emit `num_tokens` copies of the real in-vocab placeholder
    token and return the ENTIRE assembled block from `embed_multimodal`. That
    reproduces `merge_image_embeddings` exactly while keeping every token id in
    vocabulary.

Known fidelity gap (documented, not silently ignored):
    The reference also computes `get_image_visible()` and threads per-token
    left/right visibility into the sparse attention so tokens inside an image
    span attend bidirectionally. That is NOT ported here — image tokens use the
    standard causal sparse pattern. Expect some quality loss versus the
    reference on image-heavy prompts.
"""
from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: F401  (re-exported)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Slot types, matching inference/image_processor.py
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


# --------------------------------------------------------------------------
# Geometry + preprocessing: verbatim port of inference/image_processor.py
# --------------------------------------------------------------------------
def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width,
                   max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size,
                downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def image_to_patches(image, cfg):
    """PIL.Image -> (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)."""
    from PIL import ImageOps

    p = int(cfg.vision_patch_size)
    image = image.convert("RGB")
    width, height = image.size
    max_wh = getattr(cfg, "vision_max_wh_ratio", None)
    if max_wh is not None and width > height * max_wh:
        width = height * max_wh
    min_px = getattr(cfg, "vision_min_pixels", 0)
    if 0 < width * height < min_px:
        ratio = (min_px / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p,
        int(cfg.vision_downsample_ratio), int(cfg.vision_max_n_token))
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh is not None and image.width >= max_wh * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4)
    patches = patches.reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int = 0):
    """N-layout slot types (final order) + the aligner-row order for IMAGE slots."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64)
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat([
        torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_START]),
        types[order],
        torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_END]),
    ])
    return types, perm


# --------------------------------------------------------------------------
# vLLM plumbing
# --------------------------------------------------------------------------
class DS4VProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Registration is class-level, so text-only DeepSeek-V4-Flash-0731 also
        # lands here. Report zero image support when the checkpoint has no
        # vision tower, which keeps it a pure text model.
        cfg = self.get_hf_config()
        if int(getattr(cfg, "vision_n_layers", 0) or 0) <= 0:
            return {"image": 0}
        return {"image": None}

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        from PIL import Image

        cfg = self.get_hf_config()
        dummy = Image.new("RGB", (max(image_width, 1), max(image_height, 1)))
        _, _, _, n_llm_h, n_llm_w = image_to_patches(dummy, cfg)
        types, _ = build_image_block(n_llm_h, n_llm_w, 0)
        return int(types.numel())

    def get_max_image_tokens(self) -> int:
        cfg = self.get_hf_config()
        # vision_max_n_token is the budget the resize solver targets.
        return int(cfg.vision_max_n_token) + COMPRESS_PAD_TO + 2


class DS4VDummyInputsBuilder(BaseDummyInputsBuilder[DS4VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        cfg = self.info.get_hf_config()
        # A square image near the token budget exercises the worst case.
        side = int(cfg.vision_patch_size) * int(cfg.vision_downsample_ratio) * 6
        overrides = (mm_options or {}).get("image")
        return {
            "image": self._get_dummy_images(
                width=side, height=side, num_images=num_images, overrides=overrides
            )
        }


class DS4VMultiModalProcessor(BaseMultiModalProcessor[DS4VProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        cfg = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        # vLLM hands the HF-processor convention: plural "images".
        images = mm_data.get("images") or mm_data.get("image") or []
        if not isinstance(images, list):
            images = [images]

        all_patches, grids, counts = [], [], []
        for img in images:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_to_patches(img, cfg)
            all_patches.append(patches)
            # types/perm are a pure function of the grid, so carry the grid only
            # and rebuild them in embed_multimodal. Keeps every mm field either
            # fixed-size-per-image or flat-with-sizes.
            grids.append(torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w],
                                      dtype=torch.int64))
            counts.append(patches.shape[0])

        # Expand each placeholder into its full block HERE, not via
        # _get_prompt_updates. vLLM sets is_update_applied=True for the
        # text+mm path and then only *searches* the returned ids, so an
        # unexpanded prompt yields "0 prompt placeholders". Expanding here is
        # also what makes the position-dependent COMPRESS_PAD_TO alignment
        # expressible at all — a replacement callable only receives item_idx.
        placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        raw_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        expanded: list[int] = []
        img_i = 0
        for tok in raw_ids:
            if tok != placeholder_id or img_i >= len(grids):
                expanded.append(tok)
                continue
            n_llm_h, n_llm_w = int(grids[img_i][2]), int(grids[img_i][3])
            types, _ = build_image_block(n_llm_h, n_llm_w, 0)
            expanded.extend([placeholder_id] * int(types.numel()))
            img_i += 1
        input_ids = expanded
        out = {"input_ids": torch.tensor([input_ids], dtype=torch.int64)}
        if all_patches:
            # flat_from_sizes wants one concatenated tensor plus per-item sizes
            out["image_patches"] = torch.cat(all_patches, dim=0)
            out["num_patches"] = torch.tensor(counts, dtype=torch.int64)
            out["image_grid"] = torch.stack(grids)
        return BatchFeature(out)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Patch counts differ per image, so `batched` cannot split them — it
        # yields the wrong item count and blows up in _merge_mm_kwargs with
        # "IndexError: list index out of range". Same shape as deepseek_vl2.py.
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            image_patches=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            image_grid=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        cfg = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            n = self.info.get_num_image_tokens(
                image_width=size.width, image_height=size.height)
            # Every slot is the placeholder: embed_multimodal returns the whole
            # block, learned START/PAD/NEWLINE/END vectors included.
            return PromptUpdateDetails.select_token_id([placeholder_id] * n, placeholder_id)

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder_id],
                replacement=get_replacement,
            )
        ]


def assemble_image_block(model, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w):
    """encode_image + slot assembly, mirroring reference merge_image_embeddings.

    types/perm are rebuilt here from the grid rather than carried through the
    multimodal kwargs — they are a pure function of (n_llm_h, n_llm_w).
    """
    types, perm = build_image_block(int(n_llm_h), int(n_llm_w), 0)
    device = next(model.vision.parameters()).device
    dtype = next(model.vision.parameters()).dtype
    embeds = model.aligner(
        model.vision(patches.to(device=device, dtype=dtype), int(n_vit_h), int(n_vit_w)),
        int(n_vit_h), int(n_vit_w),
    )[perm.to(device)]
    params = torch.stack([
        model.image_start, model.image_pad, model.image_pad,
        model.image_newline, model.image_end,
    ]).to(device=device, dtype=embeds.dtype)
    types = types.to(device)
    block = params[types]
    block[types == IMAGE] = embeds
    return block
