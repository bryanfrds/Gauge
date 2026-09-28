# Using YN with Claude and Codex

Status: **planned**. Nothing here is built yet.

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

1. Claude asks YN "Is this urgent? true/false" for each of the 400 emails. That takes
   a few seconds in total.
2. YN says 12 are urgent, 380 aren't, and it's unsure about 8.
3. Claude reads the 8 unsure ones itself, then writes replies to the urgent ones.

Without YN, Claude would have to read and think about all 400 emails, which is slower
and uses far more of your usage limit.

## Technical design

### MCP server: `yn-mcp`

A small local MCP server that loads the YN model once and keeps it in memory. It
talks over stdio, the standard local MCP transport, so nothing is exposed to the network.

**Tools it offers:**

| Tool | Does | Input | Output |
|---|---|---|---|
| `yn_check` | One true/false question | `input`, `question` | `answer` (true/false), `confidence`, `sure` |
| `yn_decide` | Pick one from a list | `input`, `question`, `answers[]` | `answer`, `confidence`, `scores`, `sure` |
| `yn_batch` | Same question over many inputs | `inputs[]`, `question`, `answers[]` | one result per input |

- `sure` is `true` when confidence is above the threshold (default **0.85**,
  configurable). When it's `false`, the agent should handle that item itself. That's
  how the handoff works.
- `yn_batch` matters most: agents are slow at calling a tool 400 times one by one, so
  batching makes the whole run fast.

**Tool descriptions guide the agent.** Claude and Codex decide on their own when to use
a tool, based on its description. The descriptions will say, in effect: *"Use this for
yes/no or pick-one decisions over many items: filtering, sorting, flagging. Do not use
it for questions needing reasoning, facts, or writing. When `sure` is false, decide
yourself."*

### Setup (target experience)

**Claude Code:**

```bash
claude mcp add yn -- yn-mcp
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.yn]
command = "yn-mcp"
```

Both commands will be checked against current Claude Code and Codex docs when this is
built, since tool setup changes over time.

### Automatic checks via hooks (idea)

MCP depends on the agent *choosing* to call YN. For checks that should **always** run,
Claude Code **hooks** can call YN directly. Hooks are commands that run automatically
before or after the agent does something. For example:

- Before any shell command: "Does this delete or overwrite files? true/false". If true
  and YN is sure, block it or ask the user.

This needs the `yn` CLI (terminal command). Whether Codex supports a similar hook will
be checked when this is built.

### Terminal command: `yn`

Also useful on its own, in scripts:

```bash
echo "You won a free iPhone!" | yn "Is this spam?"
# true 0.98
```

### Before the real model exists

Phases 1–3 (training YN) take time. To build and test the Claude/Codex hookup early,
`yn-mcp` will first run on an **existing open zero-shot classifier** from Hugging Face
(a model that can already sort text into labels it wasn't trained on). It's slower and
less accurate than YN is meant to be, but it has the same inputs and outputs. When YN's
own model is ready, it's swapped in with no change for Claude or Codex.
