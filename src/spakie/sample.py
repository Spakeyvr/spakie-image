"""Generate sprites from text.

  spakie-sample --run runs/base "red dragon" "gold key with a gem" -n 8
  spakie-sample --run runs/base "green potion" --inpaint sprite.png --keep top

Writes a preview grid (one row per prompt) and each sprite as a native 16x16 RGBA PNG.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from spakie.checkpoint import load_for_sampling
from spakie.diffusion.sampler import sample
from spakie.render import save_grid, sprite_image


def png_to_tokens(path: str, palette: np.ndarray, side: int) -> np.ndarray:
    """Quantise an RGBA image to the nearest palette entries (alpha < 128 -> transparent)."""
    img = np.asarray(Image.open(path).convert("RGBA").resize((side, side), Image.NEAREST)).astype(np.int32)
    rgb, alpha = img[..., :3].reshape(-1, 3), img[..., 3].reshape(-1)
    dist = ((rgb[:, None, :] - palette[None, 1:].astype(np.int32)) ** 2).sum(-1)
    return np.where(alpha < 128, 0, dist.argmin(1) + 1).astype(np.int32)


def keep_mask(region: str, side: int) -> np.ndarray:
    """Which pixels of the input image to keep; the rest are regenerated."""
    m = np.zeros((side, side), dtype=bool)
    h = side // 2
    {"top": lambda: m.__setitem__(slice(0, h), True),
     "bottom": lambda: m.__setitem__(slice(h, side), True),
     "left": lambda: m.__setitem__((slice(None), slice(0, h)), True),
     "right": lambda: m.__setitem__((slice(None), slice(h, side)), True)}[region]()
    return m.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("-n", type=int, default=8, help="sprites per prompt")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=3.0, help="guidance scale; 1 disables guidance")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--order", default="confidence", choices=["confidence", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--scale", type=int, default=8, help="upscale factor for the preview grid")
    ap.add_argument("--inpaint", help="RGBA image to start from")
    ap.add_argument("--keep", default="top", choices=["top", "bottom", "left", "right"])
    args = ap.parse_args()

    model, tok, palette, state = load_for_sampling(args.run, args.checkpoint)
    side = model.cfg.side
    mx.random.seed(args.seed)

    captions = [p for p in args.prompts for _ in range(args.n)]
    text = mx.array(tok.encode(captions))
    empty = mx.array(tok.empty(len(captions)))

    init = known = None
    if args.inpaint:
        base = png_to_tokens(args.inpaint, palette, side)
        init = mx.array(np.tile(base, (len(captions), 1)))
        known = mx.array(np.tile(keep_mask(args.keep, side), (len(captions), 1)))

    out = np.array(
        sample(model, text, empty, steps=args.steps, cfg=args.cfg, temperature=args.temperature,
               order=args.order, init=init, known=known)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_grid(out, palette, out_dir / "grid.png", cols=args.n, scale=args.scale, side=side)
    for i, caption in enumerate(captions):
        slug = re.sub(r"[^a-z0-9]+", "-", caption.lower()).strip("-")[:48] or "sprite"
        sprite_image(out[i : i + 1], palette, side).save(out_dir / f"{slug}_{i % args.n:02d}.png")
    print(f"step {state['step']} checkpoint -> {len(captions)} sprites in {out_dir}/ (grid.png: one row per prompt)")


if __name__ == "__main__":
    main()
