# Using Gauge with Claude and Codex

Status: **working, on a stand-in model.** The `gauge` command and the `gauge-mcp` server are
built and tested with Claude Code and Codex. They currently run on a borrowed
open model (see [Stand-in model](#stand-in-model)) until Gauge's own model is trained.

## In plain English

Claude and Codex are smart but slow and pricey per question. Gauge is fast and nearly free,
but it can only answer yes/no or pick from a list.

So they work as a team:

- **Claude or Codex** does the thinking, planning and writing.
- **Gauge** handles the quick yes/no calls, often hundreds of them, in milliseconds.
- If Gauge isn't sure, it says so, and Claude or Codex decides that one itself.

Gauge runs on your own computer. Claude and Codex talk to it through **MCP** (Model
Context Protocol), the standard way to give them extra tools. Once it's set up, Gauge
just appears as a tool they can use.

### Example

You ask Claude: *"Go through my 400 unread emails and draft replies to the urgent ones."*

1. Claude asks Gauge "Is this urgent? true/false" for all 400 emails in one batch.
2. Gauge says which are urgent, which aren't, and which it's unsure about.
3. Claude reads the unsure ones itself, then writes replies to the urgent ones.

Without Gauge, Claude would have to read and think about all 400 emails, which is slower
and uses far more of your usage limit.

## Install

Requires Python 3.10+. From the repo folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

The first run downloads the stand-in model (~370 MB) into the Hugging Face cache.

### Connect to Claude Code

```bash
claude mcp add -s user gauge -- "$PWD/.venv/bin/gauge-mcp"
```

`-s user` makes Gauge available in every project. Check it with `claude mcp get gauge`.

### Connect to Codex

```bash
codex mcp add gauge -- "$PWD/.venv/bin/gauge-mcp"
```

Check it with `codex mcp get gauge`. To remove Gauge later, run `claude mcp remove -s user gauge`
or `codex mcp remove gauge`.

### Memory

Every open chat starts its own Gauge server, so an idle one holds almost nothing. The model
runs in a separate process that starts on the first question and stops after two quiet
minutes, which hands all of its memory back. Measured with the ONNX backend: the server
itself uses about 60 MB, the model process about 610 MB while it is running, and the
first answer after a pause takes up to about a second while the model process starts again.

| Variable | Effect |
|---|---|
| `GAUGE_IDLE_UNLOAD` | Seconds without a question before the model process stops (default 120). `0` keeps the model loaded inside the server for the whole chat, and loads it as soon as the server starts. |

## Tools

| Tool | Does | Input | Output |
|---|---|---|---|
| `gauge_check` | Is a statement true of the text? | `input`, `claim` | `answer` ("true"/"false"), `confidence`, `scores`, `sure` |
| `gauge_decide` | Pick the best option | `input`, `options[]` (2–50) | `answer`, `confidence`, `scores`, `sure` |
| `gauge_check_batch` | `gauge_check` over many texts | `inputs[]`, `claim` | `results[]`, one per input, in order |
| `gauge_decide_batch` | `gauge_decide` over many texts | `inputs[]`, `options[]` | `results[]`, one per input, in order |
| `gauge_route` | Which AI model should do this task? | `task`, optional `routes[]` | `answer` (model name), `confidence`, `scores`, `sure` |
| `gauge_route_batch` | `gauge_route` over many tasks | `tasks[]`, optional `routes[]` | `results[]`, one per task, in order |

- **`claim` is a statement, not a question.** Write "This email is spam.", not
  "Is this spam?". The stand-in model judges whether the text supports a statement.
- **Options:** short labels (`billing`) work, but full statements ending in a period
  work much better. In testing, "The customer is asking about a delivery." scored
  **0.98** where the bare label `shipping` only scored 0.55.
- **`sure`** is true when confidence ≥ **0.85** (set `GAUGE_THRESHOLD` to change it).
  When it's false, the agent should decide that item itself. That's the handoff.
- **Batch tools matter most:** one call for 100 items is far faster than 100 calls.
- A bad request (for example, 1 option or empty text) returns an error message saying
  what's wrong, so the agent can fix it and retry.
- **Limits per call:** 1,000 inputs, 20,000 input × option pairs, 20,000 characters
  per input, and 1,000 characters (and 400 tokens, the model's word pieces) per claim
  or option. Larger jobs get a clear error
  asking the agent to split them.
- Parallel calls are safe. They run one at a time on the model.

The tool descriptions tell the agent when to use Gauge (filtering, flagging, sorting,
routing) and when not to (reasoning, facts, maths, writing).

## Terminal command: `gauge`

```bash
gauge check "This email is spam." "You won a free iPhone, click here!"
# true	0.99

gauge decide "My card was charged twice" billing shipping technical   # text, then options
# billing	0.98

echo "My card was charged twice" | gauge decide - billing shipping technical   # - reads stdin
gauge decide -o billing -o shipping "My card was charged twice"   # the -o form still works

gauge route "Fix the typo in the README title"
# claude-haiku-4-5	0.90

gauge check "This message is urgent." --lines < subjects.txt     # one result per line
gauge decide --json -o "..." -o "..." "text"                      # full JSON output
```

A low-confidence result gets `unsure` appended. With `--exit-code`, the exit status
carries the answer, which is useful in scripts and hooks:

| Command | 0 | 1 | 2 |
|---|---|---|---|
| `check` | true | false | not sure |
| `decide` | sure | — | not sure |

Errors always use other codes, with or without `--exit-code`: **64** means a bad request
(missing input, bad option or flag), and **3** means a setup or model problem. A script
or hook must treat anything other than 0, 1 or 2 as "no answer", never as "false".

Limits are the same as for the MCP tools: 1,000 inputs (lines with `--lines`), 20,000
characters per input, and claims or options up to 1,000 characters and 400 tokens. A
longer document gets exit code 64 rather than being silently cut. Split it first.

Each `gauge` run loads the model, which takes about 3–4 seconds. For many decisions,
use `--lines` or the MCP server, which keeps the model loaded for a couple of minutes
between questions (see [Memory](#memory)).

## Model routing: `gauge route`

Suggests which AI model should handle a task, so cheap tasks go to cheap models. It
uses `decide` under the hood: each model has a sentence describing the tasks it suits.

**Built-in routes** (the current Claude models, cheapest first):

| Model | Suits |
|---|---|
| `claude-haiku-4-5` | Quick, simple tasks: small edits, lookups, renaming, formatting, one-line answers |
| `claude-sonnet-5` | Everyday tasks: writing a function, fixing an ordinary bug, a short document |
| `claude-opus-5-5` | Hard tasks: tricky debugging, system design, changes across many files |
| `claude-fable-5-1` | Very hard, long or high-stakes work: big multi-step projects, deep research, problems other models failed at |

**Your own routes** (for example Codex models): a JSON file with a list of
`{"model": "...", "when": "A sentence describing tasks it suits."}`. Pass it with
`gauge route --routes FILE`, set `GAUGE_ROUTES=FILE` to change the default everywhere, or
pass `routes` to the MCP tools.

Write each `when` as a full sentence. Unlike `decide` options, route descriptions are
not wrapped in a template, so a bare label like "quick edits" scores worse. `GAUGE_ROUTES`
is re-read on every call, so edits take effect without a restart. A broken `GAUGE_ROUTES`
file is reported as a setup error (exit 3), never as a problem with the request.

```json
[
  {"model": "gpt-5-codex-mini", "when": "A quick, simple edit."},
  {"model": "gpt-5-codex", "when": "A hard change across many files."}
]
```

**Accuracy on the stand-in model: a rough suggestion, not a decision.** On 12 test
tasks (3 per model) it picked the intended model **8 times**. Every miss was off by one
level (for example Haiku instead of Sonnet), never cheapest-vs-most-expensive. It was
unsure about 11 of the 12, which is the honest signal. When `sure` is false, choose
yourself or go one model up.

The tasks are in `eval/route_tasks.json`. Re-measure with
`.venv/bin/python eval/route_eval.py`, for example after changing route wording or
swapping in a new model. Twelve tasks is a smoke test, not a benchmark.

## Automatic checks via hooks (idea, not built)

MCP depends on the agent *choosing* to call Gauge. For checks that should **always** run,
Claude Code **hooks** can call `gauge --exit-code` directly. Hooks are commands that run
automatically before or after the agent does something. For example, before any shell
command: "This command deletes files." If true and Gauge is sure, block it or ask the user.

## Stand-in model

Until Gauge's own model is trained (Roadmap Phases 1–3), Gauge runs on
[`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
(MIT license). It's an open "zero-shot" model: it can judge whether text supports a
statement it was never trained on. Set `GAUGE_MODEL` to try another model of the same
kind.

**Speed** (Apple Silicon Mac, model already loaded): about 0.1 s for one decision, and
about 0.9 s for 100 decisions on the Mac's GPU (2.5 s on CPU). Gauge uses the GPU
automatically when there is one. Set `GAUGE_DEVICE=cpu` to force the CPU.

**Accuracy: good enough to try, not to trust blindly.** In an 8-email "is this urgent?"
test:

- Claude Code used Gauge correctly: one batch call, then it judged the unsure ones itself.
- The stand-in got 5–7 of 8 right depending on how the statement was worded, and it
  was sometimes **confidently wrong**. For example, it marked "invoice 60 days overdue,
  service suspension tomorrow" as not urgent with 91% confidence.
- The larger version of the same model wasn't reliably better.

This is the gap Gauge's own training is meant to close: accuracy across many decision
types (Phase 2) and confidence you can trust (Phase 3). Until then, treat `sure` as a
hint, and keep a person or the agent in the loop for anything that matters.
