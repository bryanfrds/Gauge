# YN AI Model

An open-source **decision model**: a small, fast AI that picks an answer from a list
you give it and tells you how sure it is. It doesn't chat or write text.

> **Status:** design docs only. No code or trained model yet.

## The idea in one example

You send:

```json
{
  "input": "My card was charged twice for the same order",
  "question": "Which team should handle this?",
  "answers": ["billing", "shipping", "technical", "other"]
}
```

YN sends back:

```json
{
  "answer": "billing",
  "confidence": 0.94,
  "scores": { "billing": 0.94, "shipping": 0.03, "technical": 0.01, "other": 0.02 }
}
```

Your program uses the answer directly. You don't need to parse any text.

## Why this exists

Most AI models write sentences, and that's slow and expensive when all you need is a
quick decision. Many jobs inside an app only need one:

- Is this message spam? **yes / no**
- Which team should get this ticket? **billing / shipping / technical**
- How urgent is this? **low / normal / high**
- Is this safe to show? **safe / unsafe**

Commercial "decision-only" models exist, such as [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
from TypeSafe AI, which launched in September 2026. YN is an open-source take on the
same idea, so anyone can run, inspect and improve it.

## Goals

1. **One model, many decisions.** Give it a new list of answers and it works without
   retraining.
2. **Honest confidence.** When it says 90% sure, it should be right about 90% of the time.
3. **Fast and cheap.** It runs on a normal CPU (the ordinary processor, no graphics card) in milliseconds.
4. **Fully open.** Code, model weights, training recipe and evaluation are all public.

## Docs

| Doc | For | What's in it |
|---|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Everyone | What YN does and doesn't do, in plain English |
| [docs/SPEC.md](docs/SPEC.md) | Builders | How it works: model, input/output format, training, testing |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Everyone | The build order, phase by phase |

## License

[Apache 2.0](LICENSE)
