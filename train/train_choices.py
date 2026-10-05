"""Fine-tune YN on your own labelled examples, so `decide` learns your labels.

    .venv/bin/python train/train_choices.py --data labelled.csv \\
        --labels Positive Neutral Negative \\
        --template "This post is {label_lower} about {entity}." \\
        --out models/my-sentiment

The CSV needs a text column (or several, joined) and a label column. Any other
column can appear in the template, so each example's options can name its own
subject: {entity} above. {label} and {label_lower} are the option being scored.

Training matches how `decide` scores at inference: each text is paired with one
statement per label, the entailment logits are softmaxed across the statements,
and cross-entropy pushes up the right one. So the result is still a YN model: ask
it with the same statements, e.g.

    yn decide "<post>" "This post is positive about Acme Bank." \\
        "This post is neutral about Acme Bank." "This post is negative about Acme Bank."

with YN_MODEL pointing at --out.

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
from yn.model import DEFAULT_MODEL, DEFAULT_THRESHOLD  # noqa: E402

MAX_CHARS = 2000  # long posts: the opening carries the sentiment, and it keeps steps fast


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
def evaluate(model, tok, rows, labels, template, device, entail, max_len,
             threshold=DEFAULT_THRESHOLD) -> dict:
    model.eval()
    preds, confs = [], []
    for i in range(0, len(rows), 16):
        probs = choice_logits(model, tok, rows[i:i + 16], labels, template, device,
                              entail, max_len).softmax(-1).cpu()
        preds += probs.argmax(-1).tolist()
        confs += probs.max(-1).values.tolist()
    gold = [labels.index(r["_label"]) for r in rows]
    correct = [p == g for p, g in zip(preds, gold)]
    sure = [c >= threshold for c in confs]
    n_sure = sum(sure)
    return {
        "n": len(rows),
        "accuracy": round(sum(correct) / len(rows), 4),
        "sure_rate": round(n_sure / len(rows), 4),
        "sure_accuracy": round(sum(c for c, s in zip(correct, sure) if s) / n_sure, 4)
        if n_sure else None,
        "confusion": dict(collections.Counter(
            f"{labels[g]}->{labels[p]}" for g, p in zip(gold, preds))),
    }


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

    ev = lambda rs: evaluate(model, tok, rs, args.labels, args.template, device, entail,
                             args.max_len)
    before = {"val": ev(val)}
    if test:
        before["test"] = ev(test)
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
        after = {"val": ev(val)}
        if test:
            after["test"] = ev(test)
        print(f"epoch {epoch + 1}:", json.dumps(after), flush=True)
        if best is None or after["val"]["accuracy"] > best["val"]["accuracy"]:
            best = after
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            (Path(args.out) / "yn_training.json").write_text(json.dumps(
                {"base": args.base, "epoch": epoch + 1, "before": before, "after": after,
                 "labels": args.labels, "template": args.template,
                 "args": {k: v for k, v in vars(args).items() if k not in ("data", "test")}},
                indent=2))
    print(f"saved best (val accuracy {best['val']['accuracy']:.3f}) to {args.out}")


if __name__ == "__main__":
    main()
