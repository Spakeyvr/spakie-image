"""Evaluate a checkpoint.

  spakie-eval --run runs/base

Reports
  - validation NLL bound (bits/pixel) on the full validation split
  - a memorisation check: for sprites generated from validation captions, the
    share of pixels that match the closest training sprite. Values near 100%
    mean the model is copying training data rather than drawing new sprites.
    The same number for real validation sprites is printed as the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from spakie.checkpoint import load_for_sampling
from spakie.config import Config
from spakie.data.dataset import PixelData
from spakie.diffusion.sampler import sample
from spakie.render import save_grid
from spakie.train import evaluate


def nearest_train(samples: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each sample: (index of closest training sprite, fraction of matching pixels)."""
    idx, match = [], []
    for s in samples.astype(train.dtype):
        best_i, best_m = 0, -1
        for start in range(0, len(train), 200_000):
            m = (train[start : start + 200_000] == s).sum(axis=1)
            j = int(m.argmax())
            if m[j] > best_m:
                best_i, best_m = start + j, int(m[j])
        idx.append(best_i)
        match.append(best_m / samples.shape[1])
    return np.array(idx), np.array(match)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--n", type=int, default=64, help="sprites for the memorisation check")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = Config.load(run / "config.json")
    model, tok, palette, state = load_for_sampling(run, args.checkpoint)
    data = PixelData(cfg.train.data_dir, subset=cfg.train.subset)

    val = evaluate(model, data, args.batch_size, 10**9, cfg.train.t_eps)
    print(f"checkpoint step {state['step']}: val {val:.4f} bits/px")

    mx.random.seed(0)
    pick = np.linspace(0, len(data.val) - 1, args.n).astype(int)
    text = mx.array(data.val.text[pick].astype(np.int32))
    out = np.array(sample(model, text, mx.array(tok.empty(args.n)), steps=args.steps, cfg=args.cfg))

    idx, match = nearest_train(out, data.train.tokens)
    _, ref = nearest_train(data.val.tokens[pick], data.train.tokens)
    print(
        f"closest training sprite, share of identical pixels:\n"
        f"  generated:        mean {match.mean():.1%}  max {match.max():.1%}  exact copies {(match == 1).sum()}/{args.n}\n"
        f"  real validation:  mean {ref.mean():.1%}  max {ref.max():.1%}  exact copies {(ref == 1).sum()}/{args.n}"
    )

    # Rows alternate: generated sprites, then each one's nearest training sprite.
    cols = 16
    rows = []
    for s in range(0, args.n, cols):
        rows += [out[s : s + cols], data.train.tokens[idx[s : s + cols]].astype(np.int32)]
    path = run / "eval_nearest.png"
    save_grid(np.concatenate(rows), palette, path, cols=cols, scale=6)
    print(f"wrote {path} (odd rows: generated, even rows: nearest training sprite)")


if __name__ == "__main__":
    main()
