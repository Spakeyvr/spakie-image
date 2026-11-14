"""Masked (absorbing-state) discrete diffusion objective.

Forward process: at noise level t in (0, 1], every pixel is independently
replaced by [MASK] with probability t. The model predicts the original pixel at
masked positions. Weighting the cross-entropy by 1/t gives a variational upper
bound on the negative log-likelihood (the MDLM objective), so the number we
report in bits per pixel is comparable across runs and model sizes.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def sample_t(batch: int, eps: float) -> mx.array:
    """Stratified noise levels: one per batch row, evenly covering (eps, 1]."""
    u = (mx.random.uniform(shape=(1,)) + mx.arange(batch) / batch) % 1.0
    return eps + (1.0 - eps) * u


def corrupt(img: mx.array, t: mx.array, u: mx.array, mask_id: int) -> tuple[mx.array, mx.array]:
    masked = u < t[:, None]
    return mx.where(masked, mask_id, img), masked


def diffusion_loss(model, img: mx.array, text: mx.array, t: mx.array, u: mx.array) -> mx.array:
    """Mean NLL bound in bits per pixel. img [B, N] int, t [B], u [B, N] uniform noise."""
    x, masked = corrupt(img, t, u, model.mask_id)
    logits = model(x, text).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, img, reduction="none")
    per_image = (ce * masked).sum(axis=1) / t
    return per_image.mean() / (img.shape[1] * math.log(2.0))
