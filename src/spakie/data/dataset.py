"""In-memory dataset. The whole thing is ~800 MB, so there is no streaming or decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from spakie.data.tokenizer import BOS, PAD


@dataclass
class Split:
    tokens: np.ndarray  # uint16 [N, 256]
    text: np.ndarray  # int16 [N, text_len]

    def __len__(self) -> int:
        return len(self.tokens)


class PixelData:
    def __init__(self, data_dir: str | Path, subset: int = 0):
        d = Path(data_dir)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.palette = np.load(d / "palette.npy", allow_pickle=False)
        self.train = Split(
            np.load(d / "train_tokens.npy", allow_pickle=False),
            np.load(d / "train_text.npy", allow_pickle=False),
        )
        self.val = Split(
            np.load(d / "val_tokens.npy", allow_pickle=False),
            np.load(d / "val_text.npy", allow_pickle=False),
        )
        if subset > 0:
            self.train = Split(self.train.tokens[:subset], self.train.text[:subset])

    def val_captions(self) -> list[str]:
        return json.loads((self.dir / "val_captions.json").read_text())

    def train_batches(
        self, batch_size: int, caption_dropout: float, seed: int, start_step: int = 0
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Endless shuffled batches of (image tokens int32, text ids int32).

        Each epoch is a fresh permutation seeded by (seed, epoch), so resuming at
        `start_step` continues the same data order. With probability
        `caption_dropout` a caption is replaced by the empty prompt, which is what
        makes classifier-free guidance possible at sampling time.
        """
        n = len(self.train)
        per_epoch = max(n // batch_size, 1)
        step = start_step
        while True:
            epoch, offset = divmod(step, per_epoch)
            rng = np.random.default_rng([seed, epoch])
            perm = rng.permutation(n)
            for b in range(offset, per_epoch):
                idx = np.sort(perm[b * batch_size : (b + 1) * batch_size])
                if len(idx) < batch_size:  # tiny subsets: sample with replacement
                    idx = rng.integers(0, n, batch_size)
                img = self.train.tokens[idx].astype(np.int32)
                text = self.train.text[idx].astype(np.int32)
                drop = np.random.default_rng([seed, step, 1]).random(batch_size) < caption_dropout
                if drop.any():
                    text[drop] = PAD
                    text[drop, 0] = BOS
                yield img, text
                step += 1

    def val_batches(self, batch_size: int, max_batches: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for b in range(min(max_batches, len(self.val) // batch_size)):
            s = slice(b * batch_size, (b + 1) * batch_size)
            yield self.val.tokens[s].astype(np.int32), self.val.text[s].astype(np.int32)
