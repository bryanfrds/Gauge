# Training Gauge on your own labels

The stand-in model judges any statement, but not especially well. Shown a few
thousand of your own labelled examples, it learns your labels. `train/train_choices.py`
does that, and the result is still a Gauge model: you ask it with `decide` as before.

## Run it

```bash
.venv/bin/python train/train_choices.py --data labelled.csv \
    --labels Positive Neutral Negative \
    --template "This post is {label_lower} about {entity}." \
    --label-col label --out models/my-sentiment \
    --batch 4 --max-len 128 --checkpointing
```

Then point Gauge at it and ask with the same statements:

```bash
GAUGE_MODEL=models/my-sentiment gauge decide "<post>" \
    "This post is positive about Acme Bank." \
    "This post is neutral about Acme Bank." \
    "This post is negative about Acme Bank."
```

- The CSV needs a text column (default: `title` and `body`, joined) and a label
  column. Any other column can go in `--template`, so each example can name its own
  subject.
- It holds back 15% of the rows (`--val-frac`) and reports accuracy on them before
  and after, with a confusion count per label. `--test other.csv` reports on a
  second set too.
- **Labelled data is often private, and a model trained on it can leak it.** Keep
  both the data and the trained model out of the repository: `data/` and `models/`
  are git-ignored.

## How it trains

Each text is paired with one statement per label. The model's entailment score for
each pair is softmaxed across the labels, and cross-entropy pushes up the right one.
That's the same scoring `decide` uses. Two things differ at use time: `decide` reads
up to 512 tokens where training reads `--max-len` (128 above) and the first 2,000
characters, and the template must end in `.`, `!` or `?` so `decide` uses the
statements word for word (the script refuses one that doesn't). Label smoothing of
0.1 (`--label-smoothing`) keeps the trained confidence from saturating.

## Honest confidence (calibration)

A model that says 0.9 should be right about 90% of the time. Raw scores rarely are,
so after training the script fits one number on the validation rows: a
**temperature** that softens (above 1) or sharpens (below 1) every score. It never
changes which answer wins, only how sure it sounds. It stays between 0.05 and 20, and
is left at 1 (scores as they are) when there are fewer than 50 validation rows or the
model got every one right: neither says anything about how sure it should be. It is
also left at 1 when it doesn't lower the validation ECE. The fit minimises log-loss,
not ECE, and on a model training already left honest the two can disagree (see the
finance tweets below). `not_fitted` in the file says which applied.

It is saved as `gauge_calibration.json` beside the model, with the **ECE** (expected
calibration error: the average gap between stated confidence and actual accuracy)
before and after. The validation "after" is measured on the rows the temperature was
fitted on, so it flatters; pass `--test` for a fair number. The target (spec §3.3) is
ECE under 0.05.

## Memory and speed on a laptop

Measured on a 16 GB Apple Silicon Mac, default model, on the Apple GPU:

| Settings | Peak memory | Speed |
|---|---|---|
| `--batch 8 --max-len 256` | 14 GB (swapping) | unusable |
| `--batch 4 --max-len 128` | over a 3.5 GB GPU cap | out of memory |
| `--batch 4 --max-len 128 --checkpointing` | 4.8 GB | ~1 s per step |

Cap the GPU's share so a run fails instead of swapping:
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2`. The
embedding table is frozen, which halves the optimizer's memory.

One trap, now handled in the script: the default model's weights are stored as
float16, and Adam updating float16 weights turns every one into NaN on the second
step. The model reported "Positive" for everything after a few steps. The script
loads weights as float32.

## First results

Three-way sentiment toward a named company on a private set of social-media posts,
with labels from a prompted LLM (DeepSeek). The question was whether a small local
model can match it. Scored on 1,800 held-out posts (80% Neutral):

| | Agrees with the LLM | Neutral caught | Negative caught | Positive caught |
|---|---|---|---|---|
| Stand-in model, untrained | 51% | | | |
| Round 1: 10,200 posts, natural mix | 88% | 96% | 59% | 51% |
| Round 2: + 5,150 balanced posts | 80% | 79% | 85% | 82% |

Round 1 mostly learned to say Neutral. Round 2 caught far more of the non-neutral
posts but over-corrected. About half its "Negative" answers were Neutral posts.
Shifting round 2's scores by the real label mix (log of real share ÷ training
share, added per label) gave 86% overall, with 79% of Negatives caught. That shift
is not built into the script yet.

Each round took 25–45 minutes on the laptop. Neither round matches the LLM yet. The
next step is more data, which means a few hours on the laptop or a GPU.

## Public benchmark: finance tweets

The results above use private data, so nobody else can check them. This one uses
public data and reruns with one command:

```bash
.venv/bin/python train/examples/twitter_financial.py
```

It downloads Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`
on Hugging Face, MIT licence): 11,931 finance tweets labelled Bearish, Bullish or
Neutral, already split into train and validation. Gauge trains on the 9,543 train
tweets (5% held back to pick the best epoch), using the statement *"This tweet is
{bearish / bullish / neutral} about the market."* It is scored on all 2,388
validation tweets, which it never sees.

| | Accuracy | Bearish caught | Bullish caught | Neutral caught |
|---|---|---|---|---|
| Always "Neutral" | 65.6% | 0% | 0% | 100% |
| Gauge, untrained | 73.6% | 86% | 86% | 67% |
| **Gauge, one epoch (21 min)** | **90.5%** | **87%** | **84%** | **93%** |

Its confidence is honest without help: on the validation tweets its stated confidence
is off by 0.028 on average (ECE), against 0.153 untrained and the 0.05 target. The
fitted temperature, 1.03, raised ECE on the 477 held-back training tweets it was
fitted on (0.048 → 0.055), so training keeps the scores as they are. On the
validation tweets it would have raised ECE to 0.044.

When the trained model is at least 85% sure (93% of tweets), it is right 94% of the
time. Untrained, Gauge read many neutral headlines as bullish or bearish (518 of 1,566).
Training mostly taught it what "neutral" means for this data.

Measured on a 16 GB Apple Silicon Mac, 2026-10-06, with the script's settings
(`--batch 4 --max-len 128 --checkpointing`). The training process used 3.2 GB when
checked mid-run (macOS `footprint`, one reading, not a recorded peak). Expect up to
the 4.8 GB in the table above: that was a peak, on longer posts.

## Calibration at use time

When `GAUGE_MODEL` points at a model directory with a `gauge_calibration.json` (written
by training), `decide` divides its scores by the saved temperature before turning them
into confidences, so a stated 0.9 is right about 90% of the time. `check` doesn't: the
temperature was fitted on the scores `decide` produces. Delete the file to go back to
uncalibrated scores.

It applies to every `decide` call and so to `route` too, whatever the options, not
only the labels it was trained on. A temperature above 1 lowers confidence, so fewer
answers clear the `sure` threshold (and `--exit-code` returns 2 more often).

After retraining into the same folder, rerun `gauge export-onnx` if you use the ONNX
backend: an export is matched by model name, so the old weights would otherwise run
with the new temperature.
