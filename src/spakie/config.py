"""Run configuration, loaded from a TOML file with [model] and [train] tables."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ModelConfig:
    dim: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    # Filled in from the prepared dataset's meta.json, not from the TOML file.
    text_vocab: int = 0
    text_len: int = 0
    n_colors: int = 0  # palette size including transparent (index 0)
    side: int = 16


@dataclass
class TrainConfig:
    data_dir: str = "data/prepared"
    batch_size: int = 256
    lr: float = 3e-4
    lr_end: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    caption_dropout: float = 0.1
    t_eps: float = 1e-3
    log_every: int = 50
    eval_every: int = 2000
    val_batches: int = 16
    subset: int = 0  # >0: train on only the first N sprites (overfit test)
    seed: int = 0
    # Compute precision for the training forward/backward pass. Weights, optimizer
    # state and the loss stay float32; only activations run in the lower precision.
    precision: str = "bf16"  # "fp32" | "bf16" | "fp16"
    # Sampling settings used for the preview grids written during training.
    preview_steps: int = 24
    preview_cfg: float = 3.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        def build(cls, values):
            known = {f.name for f in fields(cls)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
            return cls(**values)

        return Config(
            model=build(ModelConfig, d.get("model", {})),
            train=build(TrainConfig, d.get("train", {})),
        )

    @staticmethod
    def load(path: str | Path) -> "Config":
        path = Path(path)
        if path.suffix == ".json":
            return Config.from_dict(json.loads(path.read_text()))
        with open(path, "rb") as f:
            return Config.from_dict(tomllib.load(f))
