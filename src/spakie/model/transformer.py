"""Single-stream transformer over [caption tokens | 256 palette-index pixels].

Design notes
- Pixels are categorical: each is one of `n_colors` palette entries, or [MASK].
  The model predicts a distribution over palette entries for every pixel.
- The palette is an arbitrary list of 4096 colours, so index proximity means
  nothing. Each colour's embedding is therefore a learned vector plus a learned
  projection of its actual RGBA value, and the output layer is tied to the same
  table. Similar colours start out with similar embeddings and similar logits.
- Image tokens get 2D rotary position encoding (rows on half of each head's
  dimensions, columns on the other half) plus a learned absolute position.
  Text tokens get learned absolute positions and no rotation.
- There is no timestep input: in masked diffusion the number of masked pixels
  already tells the model the noise level.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from spakie.config import ModelConfig
from spakie.data.tokenizer import PAD


def _rope_tables(side: int, text_len: int, head_dim: int) -> tuple[mx.array, mx.array]:
    """cos/sin tables [text_len + side*side, head_dim // 2]; text rows are identity."""
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    n_freq = head_dim // 4
    # Wavelengths from 2 px up to ~4x the sprite side.
    freqs = 2 * math.pi / (2.0 * (2.0 * side) ** (np.arange(n_freq) / max(n_freq - 1, 1)))
    ys, xs = np.divmod(np.arange(side * side), side)
    ang = np.concatenate([ys[:, None] * freqs[None], xs[:, None] * freqs[None]], axis=1)
    ang = np.concatenate([np.zeros((text_len, 2 * n_freq)), ang], axis=0).astype(np.float32)
    return mx.array(np.cos(ang)), mx.array(np.sin(ang))


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, H, S, head_dim]; cos/sin: [S, head_dim // 2]
    half = x.shape[-1] // 2
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, mask: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, D // self.heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        scale = 1.0 / math.sqrt(D // self.heads)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, S, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array, mask: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), mask, cos, sin)
        return x + self.fc2(nn.gelu_approx(self.fc1(self.norm2(x))))


class PixelTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, palette: np.ndarray):
        super().__init__()
        assert cfg.text_vocab and cfg.text_len and cfg.n_colors, "model config is missing dataset meta"
        assert len(palette) == cfg.n_colors
        self.cfg = cfg
        self.mask_id = cfg.n_colors  # one extra input-only token
        d, n_px = cfg.dim, cfg.side * cfg.side

        # RGBA features per input token: palette colours, then an all-zero row for [MASK].
        feats = np.zeros((cfg.n_colors + 1, 5), dtype=np.float32)
        feats[: cfg.n_colors, :3] = palette.astype(np.float32) / 127.5 - 1.0
        feats[1 : cfg.n_colors, 3] = 1.0  # opaque
        feats[0, :3] = 0.0  # transparent has no colour
        feats[0, 4] = 1.0  # transparent flag
        self._color_feats = mx.array(feats)  # leading underscore: constant, not a parameter

        self.color_emb = mx.random.normal((cfg.n_colors + 1, d)) * 0.02
        self.rgb_proj = nn.Linear(5, d, bias=False)
        self.img_pos = mx.random.normal((n_px, d)) * 0.02
        self.text_emb = mx.random.normal((cfg.text_vocab, d)) * 0.02
        self.text_pos = mx.random.normal((cfg.text_len, d)) * 0.02

        self.blocks = [Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        self.norm = nn.RMSNorm(d)
        self.out_proj = nn.Linear(d, d)
        self.out_bias = mx.zeros((cfg.n_colors,))

        self._cos, self._sin = _rope_tables(cfg.side, cfg.text_len, d // cfg.heads)

    def __call__(self, img: mx.array, text: mx.array) -> mx.array:
        """img [B, 256] ids in [0, n_colors] (n_colors = MASK); text [B, L] -> logits [B, 256, n_colors]."""
        L = text.shape[1]
        table = self.color_emb + self.rgb_proj(self._color_feats.astype(self.color_emb.dtype))
        x = mx.concatenate([self.text_emb[text] + self.text_pos, table[img] + self.img_pos], axis=1)

        # Padding captions tokens are never attended to. Image tokens are always valid.
        pad = mx.concatenate([text == PAD, mx.zeros(img.shape, dtype=mx.bool_)], axis=1)
        mask = mx.where(pad, -mx.inf, 0.0).astype(x.dtype)[:, None, None, :]

        for blk in self.blocks:
            x = blk(x, mask, self._cos, self._sin)

        h = self.out_proj(self.norm(x[:, L:]))
        return h @ table[: self.cfg.n_colors].T + self.out_bias

    def num_params(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))
