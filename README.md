# Gauge

Gauge is a small, open-source AI that **answers yes or no**, true or false, and tells you
how sure it is. That's all it does.

> **Status:** early. The `gauge` command and the Claude/Codex tool work today on a
> borrowed stand-in model. Gauge's own model isn't trained yet. See
> [docs/CLAUDE-AND-CODEX.md](docs/CLAUDE-AND-CODEX.md) to try it.

## It's not a chatbot

Chat AIs like ChatGPT are **LLMs** (large language models). They write replies word by
word, like someone typing. That makes them great at conversation, but:

- they're **slow**, because every word takes time,
- they're **expensive**, because they're huge and need powerful computers,
- they can **ramble or go off-script**: you ask for "yes" or "no" and get a paragraph,
  or an answer you didn't expect.

Gauge doesn't write anything. It reads your question and gives back one of the answers
you allowed, plus a number showing how confident it is. There's no typing and no
rambling, so it can't give you an answer you didn't allow.

## How it works

1. **You give it something to judge.** A message, a review, a support ticket, any text.
2. **You ask a question about it.** "Is this spam?"
3. **Gauge scores each possible answer.** It checks how well "true" fits and how well
   "false" fits, and gives each one a score.
4. **You get the winner and its confidence.** `false`, 97% sure.

Because it only scores answers instead of writing them, it's much smaller than an LLM.
It can run on a normal laptop and answer in a few thousandths of a second.

## Example: true or false

You send:

```json
{
  "input": "Congratulations!! You won a free iPhone, click here to claim",
  "question": "Is this spam?",
  "answers": ["true", "false"]
}
```

Gauge sends back:

```json
{
  "answer": "true",
  "confidence": 0.98,
  "scores": { "true": 0.98, "false": 0.02 }
}
```

Your program reads `answer` and acts on it. You don't need to read or parse any text.

## Example: more than two choices

Yes/no is the core, but the same trick works for any short list of answers. Gauge scores
every option and picks the best one.

You send:

```json
{
  "input": "My card was charged twice for the same order",
  "question": "Which team should handle this?",
  "answers": ["billing", "shipping", "technical", "other"]
}
```

Gauge sends back:

```json
{
  "answer": "billing",
  "confidence": 0.94,
  "scores": { "billing": 0.94, "shipping": 0.03, "technical": 0.01, "other": 0.02 }
}
```

Your program uses the answer directly. You don't need to parse any text.

## Works with Claude and Codex (planned)

Gauge plugs into Claude Code and Codex as a tool through MCP (Model Context Protocol, the
standard way to give them extra tools). They hand Gauge the quick yes/no calls, like
"is this email urgent?" across 400 emails, and only think hard about the ones Gauge isn't
sure about. It's faster and uses far less of your limits.

It can also suggest **which AI model should handle a task** (`gauge route`), so quick
jobs go to cheap, fast models and hard ones go to the strongest.

Details: [docs/CLAUDE-AND-CODEX.md](docs/CLAUDE-AND-CODEX.md)

## Running it lighter and faster

By default Gauge runs on PyTorch, which is a machine-learning *training* framework. Gauge
only ever runs a trained model, so all that extra machinery is dead weight.

Export the model once and it runs on ONNX Runtime instead — a small program that only
knows how to run models:

```
pip install 'gauge-model[onnx,export]'
gauge export-onnx
```

That's it. Gauge picks up the export automatically from then on. On a plain CPU:

| | Memory | Startup | Four inputs |
|---|---|---|---|
| PyTorch | 709 MB | 4.0 s | 0.99 s |
| ONNX Runtime | 608 MB | 0.6 s | 0.27 s |

Same answers. Scores agree within 0.002 — close enough that the verdict doesn't move,
though a confidence sitting within 0.002 of your threshold could land either side of
`sure`.

The backend already imports PyTorch lazily, so nothing loads it once an export
exists. Dropping it from the install — and the couple of gigabytes it adds to a
container image — needs it moved out of the core dependencies, which hasn't happened
yet.

Set `GAUGE_BACKEND=torch` to force the old path, or `GAUGE_ONNX_DIR` to keep the export
somewhere specific (useful for a read-only container volume).

## Teaching it your own labels

Out of the box Gauge is a generalist. Shown a few thousand labelled examples, it learns
your task. On a public set of finance tweets (bearish / bullish / neutral), one
21-minute training run on a laptop took it from **73.6% to 90.5%** on tweets it never
saw. Rerun it yourself with `train/examples/twitter_financial.py`; the method,
memory use and results are in [docs/TRAINING.md](docs/TRAINING.md).

## Why this exists

Most AI models write sentences, and that's slow and expensive when all you need is a
quick decision. Many jobs inside an app only need one:

- Is this message spam? **yes / no**
- Which team should get this ticket? **billing / shipping / technical**
- How urgent is this? **low / normal / high**
- Is this safe to show? **safe / unsafe**

Commercial "decision-only" models exist, such as [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
from TypeSafe AI, which launched in September 2026. Gauge is an open-source take on the
same idea, so anyone can run, inspect and improve it.

Others have built open single-task versions, such as one for predicting sales. Gauge aims
to be the open **general-purpose** one. See [Related work](docs/SPEC.md#9-related-work).

## Goals

1. **One model, many decisions.** Give it a new list of answers and it works without
   retraining.
2. **Honest confidence.** When it says 90% sure, it should be right about 90% of the time.
3. **Fast and cheap.** It runs on a normal CPU (the ordinary processor, no graphics card) in milliseconds.
4. **Fully open.** Code, model weights, training recipe and evaluation are all public.

## Docs

| Doc | For | What's in it |
|---|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Everyone | What Gauge does and doesn't do, in plain English |
| [docs/SPEC.md](docs/SPEC.md) | Builders | How it works: model, input/output format, training, testing |
| [docs/CLAUDE-AND-CODEX.md](docs/CLAUDE-AND-CODEX.md) | Everyone | How Claude and Codex use Gauge as a tool |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Everyone | The build order, phase by phase |
| [docs/ONNX.md](docs/ONNX.md) | Builders | Running without PyTorch: setup, measurements, what was rejected |
| [docs/TRAINING.md](docs/TRAINING.md) | Builders | Training on your own labels, laptop memory, and the benchmarks |

## License

[Apache 2.0](LICENSE)
