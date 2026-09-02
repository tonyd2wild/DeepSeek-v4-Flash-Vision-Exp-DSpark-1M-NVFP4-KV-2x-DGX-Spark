"""DeepSeek-V4-Flash-Vision-Exp vision tower + aligner, ported for vLLM.

Numerics are a verbatim port of the checkpoint's own reference implementation
(`inference/vision.py` shipped inside DeepSeek-V4-Flash-Vision-Exp) so that
outputs match the reference bit-for-bit modulo dtype.

Deliberately NOT tensor-parallel sharded. The tower is ~410M params (~0.8 GiB
bf16) and the reference runs it on a single device; replicating it on every TP
rank costs less than the all-gather would and keeps the port faithful. Only the
language model is sharded.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: str = "cpu"):
    """2D RoPE tables over the (n_h, n_w) patch grid. Cached: one image shape
    recurs across every block and usually across requests."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
    if device != "cpu":
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    """fp32 RMSNorm, matching the reference (vLLM's fused RMSNorm differs in
    accumulation order; keep the reference's to preserve numerics)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size**2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        # Full bidirectional attention over one image — no mask.
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, inter_dim: int):
        super().__init__()
        self.norm1 = VisionRMSNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = VisionRMSNorm(dim)
        self.mlp = VisionMLP(dim, inter_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        dim = config.vision_dim
        n_heads = config.vision_n_heads
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(config.vision_patch_size, dim)
        self.blocks = nn.ModuleList(
            [VisionBlock(dim, n_heads, config.vision_inter_dim)
             for _ in range(config.vision_n_layers)]
        )
        self.norm = VisionRMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, str(x.device)
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """Pixel-shuffle downsample by `vision_downsample_ratio`, then 2-layer GELU MLP
    into the language model's hidden size."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


def build_vision_modules(config):
    """Return (vision, aligner) or (None, None) when the checkpoint is text-only."""
    if int(getattr(config, "vision_n_layers", 0) or 0) <= 0:
        return None, None
    return ViT(config), Aligner(config)
