"""Iterative unmasking sampler with classifier-free guidance.

Starts from an all-[MASK] canvas (or a partly known one, for inpainting) and
reveals pixels over `steps` model calls. Two reveal orders:

  "random"      the exact reverse of the training corruption: every masked pixel
                is revealed with the same probability each step. Most diverse.
  "confidence"  reveal the pixels the model is most sure about first, with
                annealed Gumbel noise on the ranking (MaskGIT-style). Cleaner
                outlines, somewhat less variety.
"""

from __future__ import annotations

import math

import mlx.core as mx


def _log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _guided_log_probs(model, x: mx.array, text: mx.array, empty: mx.array, cfg: float) -> mx.array:
    if cfg == 1.0:
        return _log_softmax(model(x, text).astype(mx.float32))
    both = model(mx.concatenate([x, x]), mx.concatenate([text, empty])).astype(mx.float32)
    cond, uncond = mx.split(_log_softmax(both), 2)
    return _log_softmax(uncond + cfg * (cond - uncond))


def sample(
    model,
    text: mx.array,
    empty: mx.array,
    steps: int = 24,
    cfg: float = 3.0,
    temperature: float = 1.0,
    order: str = "confidence",
    order_noise: float = 1.0,
    init: mx.array | None = None,
    known: mx.array | None = None,
) -> mx.array:
    """Generate [B, N] palette indices.

    text/empty: [B, L] caption ids and the matching empty prompts.
    init/known: optional [B, N] starting pixels and a bool mask of which to keep.
    """
    B = text.shape[0]
    N = model.cfg.side * model.cfg.side
    x = mx.full((B, N), model.mask_id, dtype=mx.int32)
    if init is not None and known is not None:
        x = mx.where(known, init.astype(mx.int32), x)

    for i in range(steps):
        t = 1.0 - i / steps
        s = 1.0 - (i + 1) / steps
        # Cosine schedule: few pixels revealed early, when context is scarce.
        mt, ms = math.sin(0.5 * math.pi * t) ** 2, math.sin(0.5 * math.pi * s) ** 2
        frac = 1.0 if i == steps - 1 else (mt - ms) / mt

        logp = _guided_log_probs(model, x, text, empty, cfg)
        proposal = mx.random.categorical(logp / temperature, axis=-1).astype(mx.int32)
        masked = x == model.mask_id

        if order == "random" or i == steps - 1:
            reveal = masked & (mx.random.uniform(shape=x.shape) < frac)
        else:
            conf = mx.take_along_axis(logp, proposal[..., None], axis=-1).squeeze(-1)
            score = conf + order_noise * t * mx.random.gumbel(shape=x.shape)
            score = mx.where(masked, score, -mx.inf)
            n_masked = masked.sum(axis=1, keepdims=True)
            k = mx.maximum(mx.ceil(n_masked * frac).astype(mx.int32), 1)
            ranked = mx.sort(score, axis=1)[:, ::-1]
            threshold = mx.take_along_axis(ranked, mx.minimum(k, N) - 1, axis=1)
            reveal = masked & (score >= threshold)

        x = mx.where(reveal, proposal, x)
        mx.eval(x)
    return x
