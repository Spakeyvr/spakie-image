"""Run directories and checkpoints.

runs/<name>/
  config.json  palette.npy  tokenizer.json     everything sampling needs
  metrics.jsonl  samples/step_*.png
  last/  best/                                  model, ema, optimizer, state.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from spakie.config import Config
from spakie.data.tokenizer import CaptionTokenizer
from spakie.model import PixelTransformer


def save_checkpoint(run: Path, name: str, model, ema: dict, optimizer, state: dict) -> None:
    tmp = run / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "ema.safetensors"), dict(tree_flatten(ema)))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (tmp / "state.json").write_text(json.dumps(state, indent=2))
    dst = run / name
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def load_tree(path: Path) -> dict:
    return tree_unflatten(list(mx.load(str(path)).items()))


def load_for_sampling(
    run: str | Path, which: str = "best", ema: bool = True
) -> tuple[PixelTransformer, CaptionTokenizer, np.ndarray, dict]:
    """Returns (model in eval mode, tokenizer, palette, checkpoint state)."""
    run = Path(run)
    ckpt = run / which
    if not ckpt.exists():
        ckpt = run / "last"
    cfg = Config.load(run / "config.json")
    palette = np.load(run / "palette.npy", allow_pickle=False)
    model = PixelTransformer(cfg.model, palette)
    weights = load_tree(ckpt / ("ema.safetensors" if ema else "model.safetensors"))
    model.update(weights)
    model.eval()
    mx.eval(model.parameters())
    tok = CaptionTokenizer.load(run / "tokenizer.json", cfg.model.text_len)
    state = json.loads((ckpt / "state.json").read_text())
    return model, tok, palette, state
