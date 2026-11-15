"""Caption tokenizer: a small BPE vocabulary trained from scratch on the dataset captions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

PAD, BOS, UNK = 0, 1, 2
SPECIALS = ["[PAD]", "[BOS]", "[UNK]"]


def train_tokenizer(captions: Iterable[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        end_of_word_suffix="</w>",
        show_progress=False,
    )
    tok.train_from_iterator(captions, trainer)
    assert [tok.token_to_id(s) for s in SPECIALS] == [PAD, BOS, UNK]
    return tok


class CaptionTokenizer:
    """Encodes captions to fixed-length id rows: [BOS] tokens... [PAD]...

    The empty caption (just [BOS]) is the unconditional prompt used for
    classifier-free guidance.
    """

    def __init__(self, tok: Tokenizer, text_len: int):
        self.tok = tok
        self.text_len = text_len

    @classmethod
    def load(cls, path: str | Path, text_len: int) -> "CaptionTokenizer":
        return cls(Tokenizer.from_file(str(path)), text_len)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, captions: list[str]) -> np.ndarray:
        out = np.full((len(captions), self.text_len), PAD, dtype=np.int32)
        out[:, 0] = BOS
        for i, enc in enumerate(self.tok.encode_batch(captions)):
            ids = enc.ids[: self.text_len - 1]
            out[i, 1 : 1 + len(ids)] = ids
        return out

    def lengths(self, captions: list[str]) -> np.ndarray:
        """Untruncated token counts including [BOS]."""
        return np.array([len(e.ids) + 1 for e in self.tok.encode_batch(captions)])

    def empty(self, n: int) -> np.ndarray:
        out = np.full((n, self.text_len), PAD, dtype=np.int32)
        out[:, 0] = BOS
        return out
