"""Train the pixel-art model.

  spakie-train --config configs/base.toml --run runs/base
  spakie-train --run runs/base --resume            # continue from runs/base/last
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from spakie.checkpoint import load_tree, save_checkpoint
from spakie.config import Config
from spakie.data.dataset import PixelData
from spakie.data.tokenizer import CaptionTokenizer
from spakie.diffusion.loss import diffusion_loss, sample_t
from spakie.diffusion.sampler import sample
from spakie.model import PixelTransformer
from spakie.render import save_grid

PREVIEW_PROMPTS = 12
PREVIEW_SEEDS = 4


def evaluate(model, data: PixelData, batch_size: int, max_batches: int, t_eps: float) -> float:
    """Validation NLL bound in bits/pixel, with fixed noise so numbers are comparable."""
    rng = np.random.default_rng(1234)
    total, n = 0.0, 0
    for img, text in data.val_batches(batch_size, max_batches):
        B = len(img)
        t = t_eps + (1 - t_eps) * ((rng.random() + np.arange(B) / B) % 1.0)
        u = rng.random(img.shape, dtype=np.float32)
        loss = diffusion_loss(model, mx.array(img), mx.array(text), mx.array(t.astype(np.float32)), mx.array(u))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def write_preview(model, tok: CaptionTokenizer, prompts: list[str], palette, path: Path, steps: int, cfg: float):
    """Grid with one column per prompt and one row per seed."""
    mx.random.seed(0)
    text = mx.array(np.tile(tok.encode(prompts), (PREVIEW_SEEDS, 1)))
    empty = mx.array(tok.empty(text.shape[0]))
    out = sample(model, text, empty, steps=steps, cfg=cfg)
    save_grid(np.array(out), palette, path, cols=len(prompts), scale=6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="TOML config (not needed with --resume)")
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, help="override train.max_steps")
    ap.add_argument("--subset", type=int, help="override train.subset")
    args = ap.parse_args()

    run = Path(args.run)
    if args.resume:
        cfg = Config.load(run / "config.json")
    else:
        if not args.config:
            ap.error("--config is required unless --resume is given")
        if (run / "last").exists():
            ap.error(f"{run} already has a checkpoint; use --resume or pick a new --run")
        cfg = Config.load(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.subset is not None:
        cfg.train.subset = args.subset
    tc = cfg.train

    data = PixelData(tc.data_dir, subset=tc.subset)
    cfg.model.text_vocab = data.meta["text_vocab"]
    cfg.model.text_len = data.meta["text_len"]
    cfg.model.n_colors = data.meta["n_colors"]
    cfg.model.side = data.meta["side"]

    run.mkdir(parents=True, exist_ok=True)
    (run / "samples").mkdir(exist_ok=True)
    (run / "config.json").write_text(cfg.to_json())
    shutil.copy(data.dir / "palette.npy", run / "palette.npy")
    shutil.copy(data.dir / "tokenizer.json", run / "tokenizer.json")
    tok = CaptionTokenizer.load(run / "tokenizer.json", cfg.model.text_len)

    mx.random.seed(tc.seed)
    model = PixelTransformer(cfg.model, data.palette)
    ema_model = PixelTransformer(cfg.model, data.palette)
    ema_model.eval()

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-7, tc.lr, tc.warmup_steps),
            optim.cosine_decay(tc.lr, max(tc.max_steps - tc.warmup_steps, 1), tc.lr_end),
        ],
        [tc.warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.99], weight_decay=tc.weight_decay)

    step, best_val = 0, float("inf")
    if args.resume:
        state = json.loads((run / "last" / "state.json").read_text())
        step, best_val = state["step"], state["best_val"]
        model.update(load_tree(run / "last" / "model.safetensors"))
        ema = load_tree(run / "last" / "ema.safetensors")
        optimizer.init(model.trainable_parameters())
        optimizer.state = load_tree(run / "last" / "optimizer.safetensors")
        print(f"resumed from step {step}")
    else:
        ema = tree_map(lambda p: mx.array(p), model.trainable_parameters())
    mx.eval(model.parameters(), ema)

    # Preview prompts: a fixed spread of validation captions, plus the unseen-in-val check
    # that the same prompt gives different sprites for different seeds.
    val_caps = data.val_captions()
    prompts = [val_caps[i] for i in np.linspace(0, len(val_caps) - 1, PREVIEW_PROMPTS).astype(int)]
    (run / "preview_prompts.txt").write_text("\n".join(prompts) + "\n")

    print(
        f"model: {model.num_params() / 1e6:.1f}M params | dim {cfg.model.dim} depth {cfg.model.depth} "
        f"heads {cfg.model.heads} | train sprites {len(data.train)} | "
        f"{len(data.train) // tc.batch_size} steps/epoch | max_steps {tc.max_steps}"
    )

    compute_dtype = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}[tc.precision]

    def loss_fn(params, img, text, t, u):
        # Mixed precision: run the network on a low-precision cast of the float32
        # weights. Gradients flow back through the cast, so they arrive in float32.
        model.update(tree_map(lambda p: p.astype(compute_dtype), params))
        return diffusion_loss(model, img, text, t, u)

    loss_and_grad = mx.value_and_grad(loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(img, text, t, u):
        params = model.trainable_parameters()
        loss, grads = loss_and_grad(params, img, text, t, u)
        model.update(params)  # put the float32 weights back
        grads, gnorm = optim.clip_grad_norm(grads, tc.grad_clip)
        optimizer.update(model, grads)
        return loss, gnorm

    @mx.compile
    def ema_step(ema, params, decay):
        return tree_map(lambda e, p: e + (1.0 - decay) * (p - e), ema, params)

    metrics = open(run / "metrics.jsonl", "a")
    batches = data.train_batches(tc.batch_size, tc.caption_dropout, tc.seed, start_step=step)
    window_loss, window_gnorm, window_n = 0.0, 0.0, 0
    t_window = time.perf_counter()

    while step < tc.max_steps:
        img, text = next(batches)
        img, text = mx.array(img), mx.array(text)
        t = sample_t(tc.batch_size, tc.t_eps)
        u = mx.random.uniform(shape=img.shape)
        loss, gnorm = train_step(img, text, t, u)
        # EMA warms up so early checkpoints are not dominated by the random init.
        decay = min(tc.ema_decay, (1.0 + step) / (10.0 + step))
        ema = ema_step(ema, model.trainable_parameters(), mx.array(decay))
        mx.eval(state, ema, loss, gnorm)
        step += 1

        window_loss += loss.item()
        window_gnorm += gnorm.item()
        window_n += 1

        if step % tc.log_every == 0 or step == tc.max_steps:
            dt = time.perf_counter() - t_window
            rec = {
                "step": step,
                "loss": window_loss / window_n,
                "gnorm": window_gnorm / window_n,
                "lr": float(optimizer.learning_rate.item()),
                "it_per_s": window_n / dt,
                "epoch": step * tc.batch_size / len(data.train),
            }
            print(
                f"step {step:>7} | loss {rec['loss']:.4f} bits/px | gnorm {rec['gnorm']:.2f} | "
                f"lr {rec['lr']:.2e} | {rec['it_per_s']:.2f} it/s | epoch {rec['epoch']:.2f}",
                flush=True,
            )
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            window_loss, window_gnorm, window_n = 0.0, 0.0, 0
            t_window = time.perf_counter()

        if step % tc.eval_every == 0 or step == tc.max_steps:
            ema_model.update(ema)
            val = evaluate(ema_model, data, tc.batch_size, tc.val_batches, tc.t_eps)
            write_preview(
                ema_model, tok, prompts, data.palette, run / "samples" / f"step_{step:07d}.png",
                tc.preview_steps, tc.preview_cfg,
            )
            improved = val < best_val
            best_val = min(best_val, val)
            ckpt_state = {"step": step, "val": val, "best_val": best_val}
            save_checkpoint(run, "last", model, ema, optimizer, ckpt_state)
            if improved:
                save_checkpoint(run, "best", model, ema, optimizer, ckpt_state)
            peak = mx.get_peak_memory() / 2**30
            print(f"step {step:>7} | val {val:.4f} bits/px (EMA){' *' if improved else ''} | peak mem {peak:.1f} GB", flush=True)
            metrics.write(json.dumps({"step": step, "val": val}) + "\n")
            metrics.flush()
            t_window = time.perf_counter()

    print(f"done. best val {best_val:.4f} bits/px. checkpoints in {run}/")


if __name__ == "__main__":
    main()
