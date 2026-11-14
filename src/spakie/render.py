"""Turn palette-index sprites into RGBA images and preview grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_rgba(tokens: np.ndarray, palette: np.ndarray, side: int = 16) -> np.ndarray:
    """[N, side*side] palette indices -> uint8 [N, side, side, 4]. Index 0 is transparent."""
    tokens = np.asarray(tokens).reshape(-1, side, side)
    alpha = (tokens != 0).astype(np.uint8) * 255
    return np.concatenate([palette[tokens], alpha[..., None]], axis=-1)


def sprite_image(tokens: np.ndarray, palette: np.ndarray, side: int = 16, scale: int = 1) -> Image.Image:
    img = Image.fromarray(to_rgba(tokens, palette, side)[0], "RGBA")
    if scale > 1:
        img = img.resize((side * scale, side * scale), Image.NEAREST)
    return img


def save_grid(
    tokens: np.ndarray,
    palette: np.ndarray,
    path: str | Path,
    cols: int = 8,
    scale: int = 8,
    side: int = 16,
    pad: int = 2,
    background: tuple[int, int, int] = (38, 40, 48),
) -> None:
    """Save sprites as one opaque PNG on a flat background, nearest-neighbour upscaled."""
    rgba = to_rgba(tokens, palette, side).astype(np.float32)
    n = len(rgba)
    rows = (n + cols - 1) // cols
    cell = side + pad
    canvas = np.empty((rows * cell + pad, cols * cell + pad, 3), dtype=np.float32)
    canvas[:] = background
    for i in range(n):
        y, x = pad + (i // cols) * cell, pad + (i % cols) * cell
        a = rgba[i, ..., 3:] / 255.0
        canvas[y : y + side, x : x + side] = rgba[i, ..., :3] * a + canvas[y : y + side, x : x + side] * (1 - a)
    img = Image.fromarray(canvas.astype(np.uint8), "RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
