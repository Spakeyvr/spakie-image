# spakie-image

A text-to-pixel-art model trained from scratch in [MLX](https://github.com/ml-explore/mlx) on Apple silicon. Type a caption, get 16×16 sprites with transparency.

It is a masked discrete diffusion model: each sprite is 256 palette indices, the model learns to fill in masked pixels given a caption, and sampling starts from a fully masked canvas and reveals pixels over a few steps. There is no VAE and nothing pretrained, including the text side.

## Setup

Requires an Apple silicon Mac, Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Data

Trained on [Pixel Art Atlas 1.4M](https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m) (1,409,404 synthetic 16×16 sprites with captions, 4,097-colour palette). This downloads the 259 MB zip and writes training arrays and a caption tokenizer to `data/prepared`:

```bash
uv run spakie-prepare
```

## Train

```bash
uv run spakie-train --config configs/tiny.toml --run runs/tiny   # smoke test, a couple of minutes
uv run spakie-train --config configs/base.toml --run runs/base   # 42.4M parameters
uv run spakie-train --run runs/base --resume                     # continue after stopping
```

Every `eval_every` steps the run writes a validation score, a preview grid in `runs/<name>/samples/`, and a checkpoint.

The base config took about 17 hours for 100,000 steps on an M5 Max (40-core GPU) and reached 1.53 bits per pixel on the validation split.

## Generate

```bash
uv run spakie-sample --run runs/base "red dragon" "gold key with a gem" -n 8
uv run spakie-web --run runs/base        # web UI at http://127.0.0.1:7860
uv run spakie-eval --run runs/base       # validation score and a check for copied training sprites
```

Short captions in the dataset's style work best. The model only knows words that appear in the dataset's captions, and it is weak at combining concepts ("blue mushroom house" gives a blue mushroom).

## Layout

```
configs/                 tiny.toml, base.toml
src/spakie/
  data/                  prepare.py, tokenizer.py, dataset.py
  model/transformer.py   single-stream transformer over [caption | pixels]
  diffusion/             loss.py (masked diffusion objective), sampler.py
  train.py  sample.py  eval.py  web.py  render.py  checkpoint.py  config.py
tests/
```

Trained weights and the dataset are not in this repository.

## Licence

The code is MIT licensed, see [LICENSE](LICENSE). The dataset has its own terms; see its dataset card.
