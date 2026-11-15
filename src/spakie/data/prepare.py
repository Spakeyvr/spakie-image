"""Download the Pixel Art Atlas zip and turn it into training-ready arrays.

Output (data/prepared):
  train_tokens.npy / val_tokens.npy   uint16 [N, 256] palette indices, row-major
  train_text.npy   / val_text.npy     int16  [N, text_len] caption token ids
  palette.npy                         uint8  [n_colors, 3]; index 0 is transparent
  tokenizer.json, val_captions.json, val_subjects.json, meta.json, preview.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from spakie.data.tokenizer import CaptionTokenizer, train_tokenizer

URL = "https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m/resolve/main/pixel-art-16x16-1.4m.zip"
ZIP_NAME = "pixel-art-16x16-1.4m.zip"
MEMBERS = ["tokens.npy", "palette.npy", "captions.json", "objects.json", "split.npy"]


def fetch(raw: Path) -> Path:
    extracted = raw / "extracted"
    if all((extracted / m).exists() for m in MEMBERS):
        return extracted
    zip_path = raw / "zip" / ZIP_NAME
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        subprocess.run(["curl", "-L", "--fail", "-o", str(zip_path), URL], check=True)
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in MEMBERS:  # extract only the known members, by name
            with z.open(m) as src, open(extracted / m, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
    return extracted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--text-len", type=int, default=32)
    args = ap.parse_args()

    src = fetch(Path(args.raw))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokens = np.load(src / "tokens.npy", mmap_mode="r", allow_pickle=False)
    palette = np.load(src / "palette.npy", allow_pickle=False)
    split = np.load(src / "split.npy", allow_pickle=False)
    with open(src / "captions.json", encoding="utf-8") as f:
        captions = json.load(f)
    with open(src / "objects.json", encoding="utf-8") as f:
        subjects = json.load(f)

    n = len(captions)
    assert tokens.shape == (n, 256) and split.shape == (n,) and len(subjects) == n
    assert palette.ndim == 2 and palette.shape[1] == 3 and palette.dtype == np.uint8
    side = 16
    train_idx = np.flatnonzero(split == 0)
    val_idx = np.flatnonzero(split == 1)
    print(f"{n} sprites: {len(train_idx)} train, {len(val_idx)} val, palette {len(palette)}")

    tokens = np.asarray(tokens)
    assert int(tokens.max()) < len(palette)
    np.save(out / "train_tokens.npy", tokens[train_idx])
    np.save(out / "val_tokens.npy", tokens[val_idx])
    np.save(out / "palette.npy", palette)

    train_caps = [captions[i] for i in train_idx]
    val_caps = [captions[i] for i in val_idx]
    print("training tokenizer")
    raw_tok = train_tokenizer(train_caps, args.vocab_size)
    raw_tok.save(str(out / "tokenizer.json"))
    tok = CaptionTokenizer(raw_tok, args.text_len)

    lens = tok.lengths(train_caps[::10])
    truncated = float((lens > args.text_len).mean())
    print(
        f"caption tokens: mean {lens.mean():.1f}, p99 {np.percentile(lens, 99):.0f}, "
        f"max {lens.max()}, truncated at {args.text_len}: {truncated:.4%}"
    )

    def encode_all(caps: list[str]) -> np.ndarray:
        parts = [tok.encode(caps[i : i + 65536]) for i in range(0, len(caps), 65536)]
        return np.concatenate(parts).astype(np.int16)

    np.save(out / "train_text.npy", encode_all(train_caps))
    np.save(out / "val_text.npy", encode_all(val_caps))
    (out / "val_captions.json").write_text(json.dumps(val_caps))
    (out / "val_subjects.json").write_text(json.dumps([subjects[i] for i in val_idx]))

    meta = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_colors": int(len(palette)),
        "side": side,
        "text_vocab": tok.vocab_size,
        "text_len": args.text_len,
        "source": URL,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    from spakie.render import save_grid

    save_grid(tokens[val_idx[:64]], palette, out / "preview.png", cols=16, scale=6)
    print(f"wrote {out}/  meta: {meta}")


if __name__ == "__main__":
    main()
