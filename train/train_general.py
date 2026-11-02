# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Train one Gauge model on many kinds of decision at once, for as long as you let it.

    python train/train_general.py --out models/general --hours 8
    python train/train_general.py --out models/general --smoke   # 3-minute check first

This is roadmap Phase 2: one model, many decisions. It mixes public classification
datasets (topic, sentiment, emotion, intent, toxicity, ...) into one stream. Each
example is a text plus a list of statements, one per possible answer, and the model
learns to score the right statement highest, exactly as `decide` asks it.

A separate set of datasets is never trained on. They measure what matters: how
well the model handles decisions it has not been trained on. Three of them (question
type, spam, irony) are kinds of decision missing from the mix entirely; the other two
are new datasets of a kind it has seen (intent, sentiment).

It trains until --hours runs out (or you press Ctrl+C), checks itself every
--eval-every-min minutes and keeps the best model so far in --out, so a crash or an
early stop still leaves a usable model. Progress is written to
--out/gauge_training.json after every check.

Datasets download from Hugging Face on first use (a few GB). A dataset that won't
load is skipped with a warning rather than stopping the run. Check each dataset's
licence before publishing a model trained on it (spec section 4).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from gauge.model import DEFAULT_MODEL, DEFAULT_THRESHOLD  # noqa: E402
from train_choices import choose_temperature, ece, lr_factor  # noqa: E402

MAX_CHARS = 2000

# Trained on. "names" overrides the dataset's own label names (same order as its label
# ids); "pretty" rewrites individual names so the statements read naturally.
TRAIN_TASKS = [
    dict(name="ag_news", ids=["fancyzhx/ag_news"], text=["text"],
         template="This news article is about {label}.",
         names=["world news", "sports", "business", "science and technology"]),
    dict(name="dbpedia", ids=["fancyzhx/dbpedia_14"], text=["title", "content"],
         template="This text describes {label}.",
         names=["a company", "a school or university", "an artist", "an athlete",
                "a politician or official", "a vehicle", "a building", "a natural place",
                "a village", "an animal", "a plant", "a music album", "a film",
                "a book or written work"]),
    dict(name="yahoo_topics", ids=["community-datasets/yahoo_answers_topics"],
         text=["question_title", "question_content", "best_answer"], label="topic",
         template="This question is about {label}.",
         names=["society and culture", "science and mathematics", "health",
                "education and reference", "computers and the internet", "sports",
                "business and finance", "entertainment and music",
                "family and relationships", "politics and government"]),
    dict(name="yelp_stars", ids=["Yelp/yelp_review_full"], text=["text"],
         template="The reviewer gave this {label}.",
         names=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
    dict(name="imdb", ids=["stanfordnlp/imdb"], text=["text"],
         template="This movie review is {label}.", names=["negative", "positive"]),
    dict(name="emotion", ids=["dair-ai/emotion"], config="split", text=["text"],
         eval_split="validation", template="The writer of this text is feeling {label}.",
         names=["sad", "happy", "loving", "angry", "afraid", "surprised"]),
    dict(name="tweet_sentiment", ids=["cardiffnlp/tweet_eval"], config="sentiment",
         text=["text"], template="This tweet is {label}.",
         names=["negative", "neutral", "positive"]),
    dict(name="banking_intent", ids=["PolyAI/banking77", "legacy-datasets/banking77"],
         text=["text"], template="This customer message is about {label}."),
    dict(name="hate_offensive", ids=["SetFit/hate_speech_offensive"], text=["text"],
         template="This tweet contains {label}.",
         pretty={"neither": "nothing hateful or offensive"}),
    dict(name="toxicity", ids=["SetFit/toxic_conversations"], text=["text"],
         template="This comment is {label}."),
    dict(name="bbc_news", ids=["SetFit/bbc-news"], text=["text"],
         template="This news article is about {label}.",
         pretty={"tech": "technology", "sport": "sports"}),
]

# Never trained on. question_type, spam and irony are new kinds of decision;
# assistant_intent and finance_tweets are new datasets of kinds in the mix.
HELD_OUT_TASKS = [
    dict(name="question_type", ids=["CogComp/trec"], config="default",
         revision="refs/convert/parquet", text=["text"], label="coarse_label",
         template="This question asks for {label}.",
         names=["an abbreviation", "a thing", "a description", "a person", "a place",
                "a number"]),
    dict(name="spam", ids=["SetFit/enron_spam"], text=["text"],
         template="This email is {label}.", pretty={"ham": "a normal email"}),
    dict(name="irony", ids=["cardiffnlp/tweet_eval"], config="irony", text=["text"],
         template="This tweet is {label}.", names=["not ironic", "ironic"]),
    dict(name="assistant_intent", ids=["clinc/clinc_oos"], config="plus", text=["text"],
         label="intent", template="This request is about {label}.",
         pretty={"oos": "something else", "yes": "saying yes", "no": "saying no",
                 "text": "sending a text message"}),
    dict(name="finance_tweets", ids=["zeroshot/twitter-financial-news-sentiment"],
         text=["text"], eval_split="validation",
         template="This tweet is {label} about the market.",
         names=["bearish", "bullish", "neutral"]),
]


def label_names(spec: dict, ds) -> list[str]:
    col = spec.get("label", "label")
    if "names" in spec:
        raw = spec["names"]
    elif hasattr(ds.features[col], "names"):
        raw = ds.features[col].names
    elif "label_text" in ds.column_names:
        seen = dict(zip(ds[col], ds["label_text"]))
        raw = [str(seen.get(i, i)) for i in range(max(seen) + 1)]
    else:
        raise ValueError(f"no label names for {spec['name']}; add 'names' to its spec")
    pretty = spec.get("pretty", {})
    return [pretty.get(n, str(n).replace("_", " ")) for n in raw]


def load_task(spec: dict, split: str, n: int, seed: int) -> tuple[list[dict], list[str]]:
    from datasets import load_dataset

    errors = []
    for ds_id in spec["ids"]:
        try:
            # Script-only datasets load from the hub's parquet conversion (spec "revision").
            ds = load_dataset(ds_id, spec.get("config"), split=split,
                              revision=spec.get("revision"))
            break
        except Exception as e:  # noqa: BLE001 - any failure means "try the next id"
            errors.append(f"{ds_id}: {e}")
    else:
        raise RuntimeError("; ".join(errors))
    names = label_names(spec, ds)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    col = spec.get("label", "label")
    rows = []
    for r in ds:
        text = "\n".join(str(r[c]).strip() for c in spec["text"] if r.get(c))
        lab = r.get(col)
        if text.strip() and lab is not None and 0 <= int(lab) < len(names):
            rows.append({"text": text[:MAX_CHARS], "gold": int(lab), "task": spec["name"]})
    return rows, names


def pick_options(gold: int, n_labels: int, k: int, rng: random.Random) -> tuple[list[int], int]:
    """Label ids to offer (the right one plus up to k-1 others, shuffled), and where
    the right one ended up."""
    if n_labels <= k:
        ids = list(range(n_labels))
    else:
        ids = [gold] + rng.sample([i for i in range(n_labels) if i != gold], k - 1)
    rng.shuffle(ids)
    return ids, ids.index(gold)


def as_item(row: dict, tasks: dict, k: int, rng: random.Random) -> tuple[str, list[str], int]:
    spec, names = tasks[row["task"]]
    ids, gold = pick_options(row["gold"], len(names), k, rng)
    return row["text"], [spec["template"].format(label=names[i]) for i in ids], gold


def forward(model, tok, items, device, entail, max_len, amp) -> list[torch.Tensor]:
    """One entailment score per (text, statement) pair, split back per text."""
    texts, hyps, counts = [], [], []
    for text, statements, _ in items:
        texts += [text] * len(statements)
        hyps += statements
        counts.append(len(statements))
    enc = tok(texts, hyps, truncation="only_first", max_length=max_len, padding=True,
              return_tensors="pt").to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
        out = model(**enc).logits[:, entail].float()
    return list(out.split(counts))


@torch.inference_mode()
def eval_logits(model, tok, items, device, entail, max_len, amp) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded (n, most options) logits, -inf where a text had fewer options, and gold."""
    model.eval()
    per = []
    for i in range(0, len(items), 16):
        per += [t.cpu() for t in forward(model, tok, items[i:i + 16], device, entail,
                                         max_len, amp)]
    width = max(len(t) for t in per)
    logits = torch.full((len(per), width), float("-inf"))
    for i, t in enumerate(per):
        logits[i, :len(t)] = t
    return logits, torch.tensor([g for _, _, g in items])


def score(logits: torch.Tensor, gold: torch.Tensor, temperature: float = 1.0) -> dict:
    probs = (logits / temperature).softmax(-1)
    confs, preds = probs.max(-1)
    correct = (preds == gold).tolist()
    sure = [c >= DEFAULT_THRESHOLD for c in confs.tolist()]
    return {"n": len(correct), "accuracy": round(sum(correct) / len(correct), 4),
            "ece": round(ece(confs.tolist(), correct), 4),
            "sure_rate": round(sum(sure) / len(sure), 4)}


def evaluate(model, tok, sets: dict, device, entail, max_len, amp) -> tuple[dict, dict]:
    """Per-task scores plus their mean, and the raw logits for calibration."""
    out, raw = {}, {}
    for name, items in sets.items():
        raw[name] = eval_logits(model, tok, items, device, entail, max_len, amp)
        out[name] = score(*raw[name])
    if out:
        out["mean_accuracy"] = round(sum(v["accuracy"] for v in out.values()) / len(out), 4)
    return out, raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=DEFAULT_MODEL,
                   help="e.g. MoritzLaurer/deberta-v3-large-zeroshot-v2.0 on a 12 GB GPU")
    p.add_argument("--hours", type=float, default=8.0, help="stop training after this long")
    p.add_argument("--per-task", type=int, default=20000, help="training rows per task")
    p.add_argument("--val-per-task", type=int, default=200,
                   help="rows per trained task for picking the best model and calibrating")
    p.add_argument("--eval-per-task", type=int, default=500, help="rows per held-out task")
    p.add_argument("--max-options", type=int, default=8,
                   help="answers offered per text; tasks with more labels get a random subset")
    p.add_argument("--batch", type=int, default=4, help="texts per step")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-every-min", type=float, default=60)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--checkpointing", action="store_true",
                   help="recompute activations in the backward pass: slower, far less memory")
    p.add_argument("--smoke", action="store_true",
                   help="tiny 3-minute run to check everything loads and saves")
    args = p.parse_args()
    if args.smoke:
        args.per_task, args.val_per_task, args.eval_per_task = 64, 16, 16
        args.hours, args.eval_every_min, args.log_every = 0.05, 1.0, 5

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, train, val_sets, held_sets, skipped = {}, [], {}, {}, {}
    eval_rng = random.Random(args.seed + 1)   # fixed options, so every check is comparable
    for spec in TRAIN_TASKS + HELD_OUT_TASKS:
        held = spec in HELD_OUT_TASKS
        try:
            ev_rows, names = load_task(spec, spec.get("eval_split", "test"),
                                       args.eval_per_task if held else args.val_per_task,
                                       args.seed)
            tasks[spec["name"]] = (spec, names)
            items = [as_item(r, tasks, args.max_options, eval_rng) for r in ev_rows]
            (held_sets if held else val_sets)[spec["name"]] = items
            if not held:
                rows, _ = load_task(spec, spec.get("split", "train"), args.per_task, args.seed)
                # Options are built from the check split's names; a training label they
                # lack would crash mid-run, so fail now instead.
                assert max(r["gold"] for r in rows) < len(names), f"{spec['name']}: label ids"
                train += rows
            print(f"{'held out' if held else 'train   '}  {spec['name']}: "
                  f"{len(names)} labels, {len(items)} check rows"
                  + ("" if held else f", {len(rows)} training rows"), flush=True)
        except Exception as e:  # noqa: BLE001 - skip a broken dataset, keep the run
            skipped[spec["name"]] = str(e)[:300]
            print(f"SKIPPED {spec['name']}: {str(e)[:300]}", flush=True)
    if not train or not held_sets:
        sys.exit("nothing to train on or nothing held out; see the SKIPPED lines above")

    tok = AutoTokenizer.from_pretrained(args.base)
    # Hub weights may be float16; Adam on float16 weights turns them into NaN. Keep
    # float32 weights and let autocast run the maths in bfloat16 on the GPU.
    model = AutoModelForSequenceClassification.from_pretrained(args.base).float().to(device)
    entail = {v.lower(): int(k) for k, v in model.config.id2label.items()}["entailment"]
    emb = getattr(model.base_model, "embeddings", None)
    if emb is not None:   # over half the weights; frozen, it halves optimizer memory
        for prm in emb.parameters():
            prm.requires_grad = False
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    trainable = [prm for prm in model.parameters() if prm.requires_grad]

    run = lambda sets: evaluate(model, tok, sets, device, entail, args.max_len, amp)
    t_eval = time.time()
    before_val, _ = run(val_sets)
    before_held, _ = run(held_sets)
    eval_secs = time.time() - t_eval
    print(f"device {device} (bf16 {amp})  base {args.base}  {len(train)} training rows",
          flush=True)
    print("before, held out:", json.dumps(before_held), flush=True)

    start = time.time()
    deadline = start + args.hours * 3600
    reserve = eval_secs * 1.5 + 60          # time kept back for the last check and save
    warmup = 5 if args.smoke else 300
    plan = {"total": 10 ** 9}               # replaced once the speed is known
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, plan["total"], warmup))

    history, best, oom_skips = [], None, 0

    def save_model(calibration: dict) -> bool:
        """Write weights, tokenizer and temperature to a side folder, then move each
        file into --out, so a crash or Ctrl+C mid-save never leaves a half-written or
        mismatched model. Returns False (and keeps training) if the move fails, e.g.
        because something on Windows has the old model file open."""
        tmp = out_dir / ".saving"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            (tmp / "gauge_calibration.json").write_text(json.dumps(calibration, indent=2))
            # Weights first, temperature last: if a locked file stops the move, the old
            # temperature never ends up beside new weights.
            order = lambda f: (f.name != "model.safetensors", f.name == "gauge_calibration.json")
            for f in sorted(tmp.iterdir(), key=order):
                os.replace(f, out_dir / f.name)
            return True
        except OSError as e:
            print(f"  SAVE FAILED, keeping the previous best: {e}", flush=True)
            return False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def check(step: int, epoch: int) -> None:
        nonlocal best
        val, val_raw = run(val_sets)
        held, held_raw = run(held_sets)
        entry = {"step": step, "epoch": epoch, "minutes": round((time.time() - start) / 60, 1),
                 "val_mean": val["mean_accuracy"], "held_out": held}
        history.append(entry)
        print(f"check at step {step}: val mean {val['mean_accuracy']:.3f}, held-out mean "
              f"{held['mean_accuracy']:.3f}", flush=True)
        if best is None or val["mean_accuracy"] > best["val_mean"]:
            # Fit the temperature on the trained families' check rows; the held-out
            # tasks then show whether it carries over to new decisions.
            logits = torch.cat([F.pad(l, (0, args.max_options - l.shape[1]),
                                      value=float("-inf")) for l, _ in val_raw.values()])
            gold = torch.cat([g for _, g in val_raw.values()])
            temp, why = choose_temperature(logits, gold)
            temp = round(temp, 4)
            held_cal = {k: score(*v, temperature=temp) for k, v in held_raw.items()}
            cal = {"temperature": temp, "fitted_on": f"{len(gold)} rows from trained tasks",
                   "ece_held_out": {k: {"before": held[k]["ece"], "after": held_cal[k]["ece"]}
                                    for k in held_raw}}
            if why:
                cal["not_fitted"] = why
            if save_model(cal):
                best = entry   # only once the files on disk match it
                print(f"  saved (best so far), temperature {temp}", flush=True)
        (out_dir / "gauge_training.json").write_text(json.dumps(
            {"base": args.base, "before": {"val": before_val, "held_out": before_held},
             "best": best, "history": history, "skipped": skipped,
             "out_of_memory_steps_skipped": oom_skips,
             "tasks": {k: {"labels": n, "template": s["template"],
                           "held_out": s in HELD_OUT_TASKS} for k, (s, n) in tasks.items()},
             "args": vars(args)}, indent=2))

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    step, epoch, last_check = 0, 0, time.time()
    try:
        while time.time() < deadline - reserve:
            epoch += 1
            order = train[:]
            rng.shuffle(order)
            for i in range(0, len(order), args.batch):
                if time.time() >= deadline - reserve:
                    break
                model.train()
                items = []
                for r in order[i:i + args.batch]:
                    # Vary how many answers are offered, so the model doesn't learn
                    # shortcuts tied to one option count (spec section 4).
                    k = args.max_options if rng.random() < 0.7 else rng.randint(2, args.max_options)
                    items.append(as_item(r, tasks, k, rng))
                try:
                    outs = forward(model, tok, items, device, entail, args.max_len, amp)
                    loss = sum(loss_fn(o.unsqueeze(0), torch.tensor([g], device=device))
                               for o, (_, _, g) in zip(outs, items)) / len(items)
                    loss.backward()
                except torch.OutOfMemoryError:
                    # One unusually long batch shouldn't end the night: drop it.
                    outs = loss = None
                    opt.zero_grad()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    oom_skips += 1
                    print(f"  out of memory on one batch, skipped ({oom_skips} so far)",
                          flush=True)
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step == 50 or (step > 50 and step % 200 == 0):
                    per_step = (time.time() - start) / step   # includes checks so far
                    plan["total"] = step + int((deadline - reserve - time.time()) / per_step)
                if step % args.log_every == 0:
                    left = max(0, deadline - time.time()) / 60
                    print(f"  step {step}/~{plan['total'] if plan['total'] < 10 ** 9 else '?'}"
                          f"  epoch {epoch}  loss {loss.item():.3f}  ~{left:.0f} min left",
                          flush=True)
                if time.time() - last_check >= args.eval_every_min * 60:
                    check(step, epoch)
                    last_check = time.time()
    except KeyboardInterrupt:
        print("stopped early (Ctrl+C): running the last check and saving", flush=True)
    except Exception:  # noqa: BLE001 - still save what the night produced
        traceback.print_exc()
        print("training stopped on an error: running the last check and saving", flush=True)
    check(step, epoch)
    if best is None:
        sys.exit(f"no model could be saved to {out_dir}; see SAVE FAILED above")
    print(f"done: best at step {best['step']}, held-out mean "
          f"{before_held['mean_accuracy']:.3f} -> {best['held_out']['mean_accuracy']:.3f}, "
          f"saved in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
