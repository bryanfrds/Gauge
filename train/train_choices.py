# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune Gauge on your own labelled examples, so `decide` learns your labels.

    .venv/bin/python train/train_choices.py --data labelled.csv \\
        --labels Positive Neutral Negative \\
        --template "This post is {label_lower} about {entity}." \\
        --out models/my-sentiment

The CSV needs a text column (or several, joined) and a label column. Any other
column can appear in the template, so each example's options can name its own
subject: {entity} above. {label} and {label_lower} are the option being scored.

Training matches how `decide` scores at inference: each text is paired with one
statement per label, the entailment logits are softmaxed across the statements,
and cross-entropy pushes up the right one. So the result is still a Gauge model: ask
it with the same statements, e.g.

    gauge decide "<post>" "This post is positive about Acme Bank." \\
        "This post is neutral about Acme Bank." "This post is negative about Acme Bank."

with GAUGE_MODEL pointing at --out.

After training, one number is fitted on the validation rows: a temperature that
softens (or sharpens) every score so a stated 0.9 is right about 90% of the time.
It is saved to gauge_calibration.json beside the model, with the calibration error
(ECE) before and after.

Labelled data is often private (customer messages, social posts), and a model
trained on it can leak it. Keep --data and --out outside this repository; the
.gitignore already excludes data/ and models/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gauge.model import DEFAULT_MODEL, DEFAULT_THRESHOLD  # noqa: E402

MAX_CHARS = 2000  # long posts: the opening carries the sentiment, and it keeps steps fast
# A temperature fitted on a handful of rows, or on rows the model got all right, says
# nothing useful (the all-right fit runs towards 0: "always certain"). Below this many
# rows, or with no mistakes to learn from, keep scores as they are.
MIN_CALIBRATION_ROWS = 50
TEMPERATURE_RANGE = (0.05, 20.0)


def read_rows(path: str, text_cols: list[str], label_col: str, labels: list[str]) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            parts = [r.get(c) or "" for c in text_cols]
            # A title the body opens with is said once. Only a prefix counts: a
            # short title like "Fees" can appear anywhere in a body and still say
            # something the body doesn't lead with.
            text = "\n".join(p for i, p in enumerate(parts)
                             if p and not any(q.startswith(p) for q in parts[i + 1:]))
            if text.strip() and r.get(label_col) in labels:
                rows.append({**r, "_text": text[:MAX_CHARS], "_label": r[label_col]})
    return rows


def lr_factor(step: int, steps: int, warmup: int) -> float:
    """Linear warm-up to 1, then linear decay towards 0 at the last step."""
    return min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup)))


def statements(row: dict, labels: list[str], template: str) -> list[str]:
    # {label} is the option being scored, even when the CSV has a "label" column too
    # (the default --label-col): that column holds the right answer, not the option.
    return [template.format_map({**row, "label": lab, "label_lower": lab.lower()})
            for lab in labels]


def choice_logits(model, tok, batch, labels, template, device, entail, max_len):
    """(len(batch), len(labels)) entailment logits, pairs ordered text-major."""
    texts, hyps = [], []
    for r in batch:
        for s in statements(r, labels, template):
            texts.append(r["_text"])
            hyps.append(s)
    enc = tok(texts, hyps, truncation="only_first", max_length=max_len, padding=True,
              return_tensors="pt").to(device)
    return model(**enc).logits[:, entail].float().view(len(batch), len(labels))


@torch.inference_mode()
def all_logits(model, tok, rows, labels, template, device, entail, max_len) -> torch.Tensor:
    """(len(rows), len(labels)) choice logits on the CPU."""
    model.eval()
    return torch.cat([choice_logits(model, tok, rows[i:i + 16], labels, template, device,
                                    entail, max_len).cpu()
                      for i in range(0, len(rows), 16)])


def ece(confs: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: the gap between stated confidence and accuracy,
    averaged over equal-width confidence bins and weighted by how many land in each."""
    err = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if b / bins < c <= (b + 1) / bins]
        if idx:
            gap = sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx)
            err += len(idx) / len(confs) * abs(gap)
    return err


def fit_temperature(logits: torch.Tensor, gold: torch.Tensor) -> float:
    """The one temperature T, within TEMPERATURE_RANGE, for which softmax(logits / T)
    best matches the right answers (lowest log-loss). Above 1 softens over-confident
    scores. It never changes which answer wins, only how sure it sounds.

    Log-loss is convex in 1/T, so it has one dip along log T, and a ternary search
    finds it without gradients (it works the same inside inference mode)."""
    logits = logits.float()
    loss = lambda log_t: torch.nn.functional.cross_entropy(
        logits / math.exp(log_t), gold).item()
    low, high = (math.log(t) for t in TEMPERATURE_RANGE)
    for _ in range(100):   # shrinks the range by a third each time: far past float precision
        a, b = low + (high - low) / 3, high - (high - low) / 3
        if loss(a) <= loss(b):
            high = b
        else:
            low = a
    return math.exp((low + high) / 2)


def choose_temperature(logits: torch.Tensor, gold: torch.Tensor) -> tuple[float, str | None]:
    """The temperature to save, and why it was left at 1 when it was."""
    if len(gold) < MIN_CALIBRATION_ROWS:
        return 1.0, f"only {len(gold)} validation rows; need {MIN_CALIBRATION_ROWS}"
    if bool((logits.argmax(-1) == gold).all()):
        return 1.0, "no validation mistakes to calibrate against"
    temp = round(fit_temperature(logits, gold), 4)
    # The fit minimises log-loss, not ECE. On a model training already left well
    # calibrated, the two can disagree and the "fix" makes stated confidence worse.
    before, after = val_ece(logits, gold, 1.0), val_ece(logits, gold, temp)
    if after >= before:
        return 1.0, (f"the fitted temperature {temp} didn't lower validation ECE "
                     f"({before:.4f} -> {after:.4f})")
    return temp, None


def val_ece(logits: torch.Tensor, gold: torch.Tensor, temperature: float) -> float:
    probs = (logits.float() / temperature).softmax(-1)
    return ece(probs.max(-1).values.tolist(), (probs.argmax(-1) == gold).tolist())


def report(logits: torch.Tensor, rows: list[dict], labels: list[str],
           temperature: float = 1.0, threshold: float = DEFAULT_THRESHOLD) -> dict:
    probs = (logits / temperature).softmax(-1)
    preds = probs.argmax(-1).tolist()
    confs = probs.max(-1).values.tolist()
    gold = [labels.index(r["_label"]) for r in rows]
    correct = [p == g for p, g in zip(preds, gold)]
    sure = [c >= threshold for c in confs]
    n_sure = sum(sure)
    return {
        "n": len(rows),
        "accuracy": round(sum(correct) / len(rows), 4),
        "ece": round(ece(confs, correct), 4),
        "sure_rate": round(n_sure / len(rows), 4),
        "sure_accuracy": round(sum(c for c, s in zip(correct, sure) if s) / n_sure, 4)
        if n_sure else None,
        "confusion": dict(collections.Counter(
            f"{labels[g]}->{labels[p]}" for g, p in zip(gold, preds))),
    }


def gold_of(rows: list[dict], labels: list[str]) -> torch.Tensor:
    return torch.tensor([labels.index(r["_label"]) for r in rows])


def _at_least_one(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="labelled CSV")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--template", required=True,
                   help="statement per label; {label}, {label_lower} and CSV columns")
    p.add_argument("--text-cols", nargs="+", default=["title", "body"])
    p.add_argument("--label-col", default="label")
    p.add_argument("--out", required=True)
    p.add_argument("--test", help="extra labelled CSV to report on (not trained on)")
    p.add_argument("--test-label-col", help="label column in --test (default: --label-col)")
    p.add_argument("--base", default=DEFAULT_MODEL)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-rows", type=int, help="use only this many rows (quick runs)")
    p.add_argument("--epochs", type=_at_least_one, default=1)
    p.add_argument("--batch", type=int, default=8, help="texts per step (x labels pairs)")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpointing", action="store_true",
                   help="recompute activations in the backward pass: slower, far less memory")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = read_rows(args.data, args.text_cols, args.label_col, args.labels)
    random.Random(args.seed).shuffle(rows)
    if args.max_rows:
        rows = rows[:args.max_rows]
    if not rows:
        sys.exit(f"no usable rows in {args.data} (labels {args.labels}, "
                 f"label column {args.label_col!r})")
    # Check the template against the data now, not after the model has loaded.
    try:
        hyps = statements(rows[0], args.labels, args.template)
    except KeyError as e:
        sys.exit(f"--template uses {e}, which is not a column in {args.data}")
    if not all(h.rstrip()[-1:] in ".!?" for h in hyps):
        # decide() wraps a statement without end punctuation in its own template, so
        # the model would be asked something other than what it was trained on.
        sys.exit("--template must end in . ! or ? so decide() uses it word for word")
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"train {len(train)}  val {len(val)}  device {device}  "
          f"labels {dict(collections.Counter(r['_label'] for r in train))}", flush=True)
    test = None
    if args.test:
        test = read_rows(args.test, args.text_cols, args.test_label_col or args.label_col,
                         args.labels)

    tok = AutoTokenizer.from_pretrained(args.base)
    # The hub copy stores weights in float16. Adam updates in float16 underflow and the
    # second step turns every weight into NaN, so train in float32.
    model = AutoModelForSequenceClassification.from_pretrained(args.base).float().to(device)
    entail = {v.lower(): int(k) for k, v in model.config.id2label.items()}["entailment"]
    # The embedding table is over half the weights (a 128k-word vocabulary). Leaving
    # it frozen roughly halves the optimizer's memory and barely moves accuracy.
    for prm in model.base_model.embeddings.parameters():
        prm.requires_grad = False
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    trainable = [prm for prm in model.parameters() if prm.requires_grad]

    lg = lambda rs: all_logits(model, tok, rs, args.labels, args.template, device, entail,
                               args.max_len)
    before = {"val": report(lg(val), val, args.labels)}
    if test:
        before["test"] = report(lg(test), test, args.labels)
    print("before:", json.dumps(before), flush=True)

    steps = args.epochs * math.ceil(len(train) / args.batch)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps, warmup))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best, step, t0 = None, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = train[:]
        random.Random(args.seed + epoch).shuffle(order)
        for i in range(0, len(order), args.batch):
            batch = order[i:i + args.batch]
            logits = choice_logits(model, tok, batch, args.labels, args.template, device,
                                   entail, args.max_len)
            gold = torch.tensor([args.labels.index(r["_label"]) for r in batch],
                                device=device)
            loss = loss_fn(logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                left = (time.time() - t0) / step * (steps - step)
                print(f"  step {step}/{steps}  loss {loss.item():.3f}  "
                      f"{time.time() - t0:.0f}s, ~{left / 60:.0f} min left", flush=True)
        logits = {"val": lg(val)}
        if test:
            logits["test"] = lg(test)
        sets = {"val": val, "test": test}
        after = {k: report(v, sets[k], args.labels) for k, v in logits.items()}
        print(f"epoch {epoch + 1}:", json.dumps(after), flush=True)
        if best is None or after["val"]["accuracy"] > best["val"]["accuracy"]:
            best = after
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            temp, skipped = choose_temperature(logits["val"], gold_of(val, args.labels))
            temp = round(temp, 4)   # what's saved is what's measured
            calibrated = {k: report(v, sets[k], args.labels, temp) for k, v in logits.items()}
            calibration = {"temperature": temp, "fitted_on": f"{len(val)} validation rows",
                           # val is what T was fitted on, so its "after" flatters; test is
                           # the fair one.
                           "ece": {k: {"before": after[k]["ece"], "after": calibrated[k]["ece"]}
                                   for k in after}}
            if skipped:
                calibration["not_fitted"] = skipped
                print(f"  temperature left at 1: {skipped}", flush=True)
            (Path(args.out) / "gauge_calibration.json").write_text(
                json.dumps(calibration, indent=2))
            print(f"  temperature {temp:.3f}: ECE " + ", ".join(
                f"{k} {after[k]['ece']:.3f} -> {calibrated[k]['ece']:.3f}" for k in after),
                flush=True)
            (Path(args.out) / "gauge_training.json").write_text(json.dumps(
                {"base": args.base, "epoch": epoch + 1, "before": before, "after": after,
                 "labels": args.labels, "template": args.template,
                 "args": {k: v for k, v in vars(args).items() if k not in ("data", "test")}},
                indent=2))
    print(f"saved best (val accuracy {best['val']['accuracy']:.3f}) to {args.out}")


if __name__ == "__main__":
    main()
