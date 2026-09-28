# Using YN with Claude and Codex

Status: **working, on a stand-in model.** The `yn` command and the `yn-mcp` server are
built and tested with Claude Code and Codex. They currently run on a borrowed
open model (see [Stand-in model](#stand-in-model)) until YN's own model is trained.

## In plain English

Claude and Codex are smart but slow and pricey per question. YN is fast and nearly free,
but it can only answer yes/no or pick from a list.

So they work as a team:

- **Claude or Codex** does the thinking, planning and writing.
- **YN** handles the quick yes/no calls, often hundreds of them, in milliseconds.
- If YN isn't sure, it says so, and Claude or Codex decides that one itself.

YN runs on your own computer. Claude and Codex talk to it through **MCP** (Model
Context Protocol), the standard way to give them extra tools. Once it's set up, YN
just appears as a tool they can use.

### Example

You ask Claude: *"Go through my 400 unread emails and draft replies to the urgent ones."*

1. Claude asks YN "Is this urgent? true/false" for all 400 emails in one batch.
2. YN says which are urgent, which aren't, and which it's unsure about.
3. Claude reads the unsure ones itself, then writes replies to the urgent ones.

Without YN, Claude would have to read and think about all 400 emails, which is slower
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
claude mcp add -s user yn -- "$PWD/.venv/bin/yn-mcp"
```

`-s user` makes YN available in every project. Check it with `claude mcp get yn`.

### Connect to Codex

```bash
codex mcp add yn -- "$PWD/.venv/bin/yn-mcp"
```

Check it with `codex mcp get yn`. To remove YN later, run `claude mcp remove -s user yn`
or `codex mcp remove yn`.

## Tools

| Tool | Does | Input | Output |
|---|---|---|---|
| `yn_check` | Is a statement true of the text? | `input`, `claim` | `answer` ("true"/"false"), `confidence`, `scores`, `sure` |
| `yn_decide` | Pick the best option | `input`, `options[]` (2–50) | `answer`, `confidence`, `scores`, `sure` |
| `yn_check_batch` | `yn_check` over many texts | `inputs[]`, `claim` | `results[]`, one per input, in order |
| `yn_decide_batch` | `yn_decide` over many texts | `inputs[]`, `options[]` | `results[]`, one per input, in order |

- **`claim` is a statement, not a question.** Write "This email is spam.", not
  "Is this spam?". The stand-in model judges whether the text supports a statement.
- **Options:** short labels (`billing`) work, but full statements ending in a period
  work much better. In testing, "The customer is asking about a delivery." scored
  **0.98** where the bare label `shipping` only scored 0.55.
- **`sure`** is true when confidence ≥ **0.85** (set `YN_THRESHOLD` to change it).
  When it's false, the agent should decide that item itself. That's the handoff.
- **Batch tools matter most:** one call for 100 items is far faster than 100 calls.
- A bad request (for example, 1 option or empty text) returns an error message saying
  what's wrong, so the agent can fix it and retry.
- **Limits per call:** 1,000 inputs, 20,000 input × option pairs, 20,000 characters
  per input, and 1,000 characters per claim or option. Larger jobs get a clear error
  asking the agent to split them.
- Parallel calls are safe. They run one at a time on the model.

The tool descriptions tell the agent when to use YN (filtering, flagging, sorting,
routing) and when not to (reasoning, facts, maths, writing).

## Terminal command: `yn`

```bash
yn check "This email is spam." "You won a free iPhone, click here!"
# true	0.99

echo "My card was charged twice" | yn decide -o billing -o shipping -o technical
# billing	0.98

yn check "This message is urgent." --lines < subjects.txt     # one result per line
yn decide --json -o "..." -o "..." "text"                      # full JSON output
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

Each `yn` run loads the model, which takes about 3–4 seconds. For many decisions,
use `--lines` or the MCP server, which loads the model once and keeps it in memory.

## Automatic checks via hooks (idea, not built)

MCP depends on the agent *choosing* to call YN. For checks that should **always** run,
Claude Code **hooks** can call `yn --exit-code` directly. Hooks are commands that run
automatically before or after the agent does something. For example, before any shell
command: "This command deletes files." If true and YN is sure, block it or ask the user.

## Stand-in model

Until YN's own model is trained (Roadmap Phases 1–3), YN runs on
[`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
(MIT license). It's an open "zero-shot" model: it can judge whether text supports a
statement it was never trained on. Set `YN_MODEL` to try another model of the same
kind.

**Speed** (Apple Silicon Mac, model already loaded): about 0.1 s for one decision, and
about 0.9 s for 100 decisions on the Mac's GPU (2.5 s on CPU). YN uses the GPU
automatically when there is one. Set `YN_DEVICE=cpu` to force the CPU.

**Accuracy: good enough to try, not to trust blindly.** In an 8-email "is this urgent?"
test:

- Claude Code used YN correctly: one batch call, then it judged the unsure ones itself.
- The stand-in got 5–7 of 8 right depending on how the statement was worded, and it
  was sometimes **confidently wrong**. For example, it marked "invoice 60 days overdue,
  service suspension tomorrow" as not urgent with 91% confidence.
- The larger version of the same model wasn't reliably better.

This is the gap YN's own training is meant to close: accuracy across many decision
types (Phase 2) and confidence you can trust (Phase 3). Until then, treat `sure` as a
hint, and keep a person or the agent in the loop for anything that matters.
