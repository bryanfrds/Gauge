# YN — Roadmap

Each phase ends with something you can check, so we find out early if the idea doesn't
work.

## Phase 0 — Docs ✅

- README, plain-English overview, technical spec, this roadmap.

## Phase 0.5 — Claude/Codex hookup with a stand-in model ✅ (working)

Result: Claude Code used YN correctly on a real task. The stand-in model's accuracy
and confidence are not reliable enough to trust unattended, which is what Phases 1–3
fix. Details in [CLAUDE-AND-CODEX.md](CLAUDE-AND-CODEX.md#stand-in-model).

- Build `yn-mcp` and the `yn` CLI on an existing open zero-shot classifier
  (see [CLAUDE-AND-CODEX.md](CLAUDE-AND-CODEX.md)).
- Connect it to Claude Code and Codex and try it on a real task.
- **Done when:** Claude or Codex uses YN for batch yes/no decisions and hands the
  unsure ones back to itself. This also tells us early whether agents actually use it.

## Phase 1 — Prove it on one decision

- Pick one real decision (for example, support-ticket routing).
- Fine-tune a small open model on it, and compare ModernBERT vs DeBERTa.
- Compare against a prompted LLM on accuracy, speed and cost.
- **Done when:** we have numbers showing whether a small model is good enough for this.

## Phase 2 — One model, many decisions

- Build the label-scoring model from the spec (§3.1).
- Assemble the multi-task training mix (§4), with test tasks held out.
- **Done when:** it handles decision types it never saw in training, within the §5 bar.

## Phase 3 — Honest confidence

- Add calibration (§3.3) and publish the reliability chart.
- **Done when:** ECE (the gap between stated and actual confidence) < 0.05 on held-out tasks.

## Phase 4 — Package and release v0.1

- Python library, HTTP server, ONNX export, model card.
- Publish weights on Hugging Face.
- **Done when:** all release gates in the spec (§6) pass.

## Phase 5 — Later

- Smaller/faster variant (`yn-small`)
- Multi-choice decisions
- Other languages
- Bi-encoder mode for large label lists
