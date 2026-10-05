# Training YN on your own labels

The stand-in model judges any statement, but not especially well. Shown a few
thousand of your own labelled examples, it learns your labels. `train/train_choices.py`
does that, and the result is still a YN model: you ask it with `decide` as before.

## Run it

```bash
.venv/bin/python train/train_choices.py --data labelled.csv \
    --labels Positive Neutral Negative \
    --template "This post is {label_lower} about {entity}." \
    --label-col label --out models/my-sentiment \
    --batch 4 --max-len 128 --checkpointing
```

Then point YN at it and ask with the same statements:

```bash
YN_MODEL=models/my-sentiment yn decide "<post>" \
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
That's the same scoring `decide` uses, so training and use can't drift apart.

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
