"""The public benchmark's data step (the download is faked; no training here)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "examples"))
import twitter_financial as tf  # noqa: E402


def test_prepare_turns_number_labels_into_names_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "DATA", tmp_path)
    fetched = []

    def fake_download(url, dest):
        fetched.append(url)
        Path(dest).write_text('text,label\n"$AAPL beats estimates",1\n"Stocks slide",0\n"Fed meets today",2\n')

    monkeypatch.setattr(tf.urllib.request, "urlretrieve", fake_download)
    paths = tf.prepare()
    rows = list(csv.DictReader(paths["train"].open()))
    assert [(r["body"], r["label"]) for r in rows] == [
        ("$AAPL beats estimates", "Bullish"), ("Stocks slide", "Bearish"), ("Fed meets today", "Neutral")]
    assert len(fetched) == 2
    tf.prepare()                                     # a second run reuses the download
    assert len(fetched) == 2


def test_labels_and_template_suit_decide():
    assert set(tf.NAMES.values()) == set(tf.LABELS)
    assert tf.TEMPLATE.rstrip()[-1] in ".!?"         # so decide() uses it word for word
