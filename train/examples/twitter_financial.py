"""A public benchmark anyone can rerun: YN on finance tweets, before and after training.

    .venv/bin/python train/examples/twitter_financial.py              # download, train, score
    .venv/bin/python train/examples/twitter_financial.py --prepare-only

Data: Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment on
Hugging Face, MIT licence): 11,932 finance tweets labelled Bearish, Bullish or Neutral,
already split into train (9,543) and validation (2,388). YN trains on the train split
and is scored on the validation split, which it never sees.

The data goes to data/twitter-financial/ and the model to models/twitter-financial/,
both git-ignored. Training takes about 40 minutes on a 16 GB Apple Silicon Mac with the
settings below and peaks around 5 GB; see docs/TRAINING.md.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "twitter-financial"
SOURCE = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/"
SPLITS = {"train": "sent_train.csv", "valid": "sent_valid.csv"}
NAMES = {"0": "Bearish", "1": "Bullish", "2": "Neutral"}  # from the dataset card
LABELS = ["Bearish", "Bullish", "Neutral"]
TEMPLATE = "This tweet is {label_lower} about the market."


def prepare() -> dict[str, Path]:
    """Download both splits once and rewrite them as text,label CSVs."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, name in SPLITS.items():
        raw = DATA / name
        if not raw.exists():
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(SOURCE + name, raw)
        path = DATA / f"{split}.csv"
        with raw.open(newline="") as f, path.open("w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["body", "label"])
            n = 0
            for row in csv.DictReader(f):
                w.writerow([row["text"], NAMES[row["label"]]])
                n += 1
        print(f"{split}: {n} tweets -> {path}", flush=True)
        out[split] = path
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--epochs", default="1")
    args = p.parse_args()
    paths = prepare()
    if args.prepare_only:
        return
    env = {**os.environ,
           # Fail instead of swapping if the GPU's share grows (see docs/TRAINING.md).
           "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3"),
           "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")}
    cmd = [sys.executable, str(ROOT / "train" / "train_choices.py"),
           "--data", str(paths["train"]), "--test", str(paths["valid"]),
           "--text-cols", "body", "--label-col", "label",
           "--labels", *LABELS, "--template", TEMPLATE,
           "--out", str(ROOT / "models" / "twitter-financial"),
           "--epochs", args.epochs, "--val-frac", "0.05",
           "--batch", "4", "--max-len", "128", "--checkpointing", "--log-every", "100"]
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
