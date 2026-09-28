# YN AI Model

YN is a small, open-source AI that **answers yes or no**, true or false, and tells you
how sure it is. That's all it does.

> **Status:** design docs only. No code or trained model yet.

## It's not a chatbot

Chat AIs like ChatGPT are **LLMs** (large language models). They write replies word by
word, like someone typing. That makes them great at conversation, but:

- they're **slow**, because every word takes time,
- they're **expensive**, because they're huge and need powerful computers,
- they can **ramble or go off-script**: you ask for "yes" or "no" and get a paragraph,
  or an answer you didn't expect.

YN doesn't write anything. It reads your question and gives back one of the answers
you allowed, plus a number showing how confident it is. There's no typing and no
rambling, so it can't give you an answer you didn't allow.

## How it works

1. **You give it something to judge.** A message, a review, a support ticket, any text.
2. **You ask a question about it.** "Is this spam?"
3. **YN scores each possible answer.** It checks how well "true" fits and how well
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

YN sends back:

```json
{
  "answer": "true",
  "confidence": 0.98,
  "scores": { "true": 0.98, "false": 0.02 }
}
```

Your program reads `answer` and acts on it. You don't need to read or parse any text.

## Example: more than two choices

Yes/no is the core, but the same trick works for any short list of answers. YN scores
every option and picks the best one.

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

## Works with Claude and Codex (planned)

YN plugs into Claude Code and Codex as a tool through MCP (Model Context Protocol, the
standard way to give them extra tools). They hand YN the quick yes/no calls, like
"is this email urgent?" across 400 emails, and only think hard about the ones YN isn't
sure about. It's faster and uses far less of your limits.

Details: [docs/CLAUDE-AND-CODEX.md](docs/CLAUDE-AND-CODEX.md)

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

Others have built open single-task versions, such as one for predicting sales. YN aims
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
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Everyone | What YN does and doesn't do, in plain English |
| [docs/SPEC.md](docs/SPEC.md) | Builders | How it works: model, input/output format, training, testing |
| [docs/CLAUDE-AND-CODEX.md](docs/CLAUDE-AND-CODEX.md) | Everyone | How Claude and Codex use YN as a tool |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Everyone | The build order, phase by phase |

## License

[Apache 2.0](LICENSE)
