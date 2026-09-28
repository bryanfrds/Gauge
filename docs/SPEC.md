# YN — Technical Spec

Status: **draft**. This describes the intended design. Nothing is built yet.

## 1. Scope

YN is a **non-generative decision model**. Given an input text, a question and a set of
candidate answers (labels), it returns a probability for each label. It never generates
free text.

**In scope (v1):** English text input; single-choice decisions (exactly one label);
yes/no, multi-class and ordinal scales (low/normal/high); 2–50 labels per request.

**Out of scope (v1):** multi-choice (several labels at once), numeric regression,
images/audio, languages other than English, long documents over the context limit.

## 2. Interface contract

### Request

```json
{
  "input": "string, required",
  "question": "string, optional — defaults to 'Which answer fits best?'",
  "answers": ["string", "..."],
  "descriptions": { "answer": "optional one-line meaning of each answer" }
}
```

- `answers`: 2–50 unique, non-empty strings. Order must not change the result.
- `descriptions`: optional. They help when labels are terse, such as `P1`, `P2`.

### Response

```json
{
  "answer": "billing",
  "confidence": 0.94,
  "scores": { "billing": 0.94, "shipping": 0.03, "technical": 0.01, "other": 0.02 },
  "model": "yn-base-v0.1"
}
```

- `scores` sums to 1.0. `answer` = argmax, and `confidence` = its score.
- The output is always valid against this schema, because it comes from scoring the
  given labels, not from generated text. That removes the whole class of
  "model returned malformed JSON" bugs.

## 3. Model design

### 3.1 Approach: label-scoring cross-encoder

Each candidate answer is scored against the input separately:

```
[CLS] question [SEP] input [SEP] answer: description [SEP]  →  encoder  →  score
```

The scores for all answers go through a softmax (turns raw scores into probabilities
that add up to 1) to produce `scores`.

Why this shape:

- **New labels without retraining.** The label is part of the input text, so the model
  learns "does this answer fit this input" in general, not a fixed set of classes.
- **Can't return an invalid answer.** It only ranks what it was given.
- **Order-independent** by construction, because every answer is scored on its own.

Trade-off: cost grows with the number of answers (one pass per answer). Mitigations,
in order of preference: batch all answers in one forward pass; cache the input
encoding; later, try a bi-encoder (encode input and labels separately, compare
vectors) for large label sets, and accept some accuracy loss.

### 3.2 Base model

Start from a small open **encoder** model (reads text and produces a score, but can't
write text). Primary candidate: **ModernBERT-base** (~150M parameters, Apache 2.0,
8k-token context). Fallback: DeBERTa-v3-base.

Size targets:

| Variant | Params | Target latency (CPU, 4 labels) |
|---|---|---|
| `yn-small` | ~30–50M | < 20 ms |
| `yn-base` | ~150M | < 60 ms |

Base model choice is confirmed in Phase 1 by benchmark, not assumed. See the roadmap.

### 3.3 Confidence calibration

Raw model scores are usually over-confident. After training:

1. Fit **temperature scaling** (a single number that softens or sharpens all scores)
   on a held-out calibration set.
2. Report **ECE** (expected calibration error: the average gap between stated
   confidence and actual accuracy) on the test set. Target: **ECE < 0.05**.
3. Publish a reliability chart with every release.

Calibration is a release gate, not an afterthought (see §6).

## 4. Training data

The model must learn "does answer X fit input Y" across many kinds of decisions.

**Sources (all must allow redistribution of derived models):**

1. **Public classification datasets**, recast into (input, question, answers) form:
   sentiment, topic, intent, spam, toxicity, natural-language inference
   (yes/no/maybe), and so on.
2. **Synthetic tasks**: use an LLM (large language model) to write new
   question/answer-set pairs and label examples. Any synthetic data must be
   spot-checked by a person, and its share of the mix reported.
3. **Hard negatives**: near-miss answers (for example, "billing" vs "refunds"), so the
   model learns fine distinctions instead of keyword matching.

**Augmentations:** shuffle answer order, paraphrase labels, drop or add descriptions,
and vary the number of answers, so the model doesn't learn shortcuts.

**Rules:**
- Every dataset is listed in `data/SOURCES.md` with its license and link.
- Test tasks are **held out entirely**: no examples from test task families appear in
  training. That's the only honest way to measure "works on new decisions."
- No personal data. Datasets are screened before use.

## 5. Evaluation

| Suite | Measures | Pass bar (v0.1) |
|---|---|---|
| Held-out tasks | Accuracy on decision types never seen in training | Within 5 pts of a GPT-class LLM prompted zero-shot |
| Calibration | ECE on held-out tasks | < 0.05 |
| Order test | Same result when answers are shuffled | 100% (guaranteed by design, test anyway) |
| Latency | p50 / p95 on a stated CPU | Meets §3.2 targets |
| Cost | $ per 1M decisions vs an LLM API | Reported, no bar |

Baselines to report alongside YN: an existing open zero-shot classifier (NLI-based),
and a prompted LLM. Publish the numbers even when YN loses.

## 6. Release gates

A version ships only when:

1. every §5 bar is met on the held-out suite,
2. the model card (training data, limits, known failure cases) is written,
3. evaluation is reproducible from a single command in the repo.

## 7. Packaging

- **Python library:** `pip install yn-model`, then `yn.decide(input, answers)`.
- **HTTP server:** one `POST /decide` endpoint implementing §2.
- **CLI:** `yn "<question>" [--answers a,b,c]`, reading input from stdin and printing
  `answer confidence`.
- **MCP server:** `yn-mcp`, so Claude Code, Codex and other agents can call YN as a
  tool. Full design in [CLAUDE-AND-CODEX.md](CLAUDE-AND-CODEX.md).
- **Weights:** published on Hugging Face, in safetensors format, plus ONNX (a portable
  format for fast CPU inference).
- **License:** Apache 2.0 for code and weights.

## 8. Planned repo layout

```
yn-ai-model/
  README.md
  LICENSE
  docs/            this spec, overview, roadmap, model card
  data/            dataset builders + SOURCES.md (no raw data committed)
  yn/              library: model, scoring, calibration
  train/           training scripts and configs
  eval/            evaluation suites and baselines
  server/          HTTP server
  tests/
```

## 9. Open questions

- **ModernBERT vs DeBERTa** as the base. Decide by the Phase 1 benchmark.
- **Maximum label count** before switching to the bi-encoder path.
- **Multi-choice** (several true labels): v2, or a flag in v1?
- **Non-English support:** which languages, if any, for v2.
- **Compute budget** for training runs (one rented GPU vs more).
