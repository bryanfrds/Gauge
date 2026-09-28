"""`yn-mcp`: MCP server so Claude Code, Codex and other agents can use YN as a tool.

Runs over stdio (local only). Stdout carries the protocol, so never print to it.
"""

from __future__ import annotations

import functools
import sys
import threading

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from yn.model import get_decider

WHEN_TO_USE = (
    "Use for fast, cheap decisions over text, especially many items at once: "
    "filtering, flagging, sorting, routing. Do NOT use for questions that need "
    "reasoning, outside facts, maths or writing. If a result has sure=false, "
    "judge that item yourself."
)

mcp = MCPServer(
    "yn",
    instructions="YN is a small local decision model. " + WHEN_TO_USE,
    log_level="WARNING",
)


def _bad_input_to_tool_error(fn):
    """Show the agent why a request was rejected, instead of a generic crash."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise ToolError(str(e)) from e

    return wrapper


@mcp.tool(description="Is a statement true of the text? Returns answer 'true'/'false', "
          "confidence 0-1 and sure. Write `claim` as a plain statement, not a question, "
          "e.g. 'This email is spam.' " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_check(input: str, claim: str) -> dict:
    return get_decider().check(input, claim).to_dict()


@mcp.tool(description="Pick the option that best fits the text. Returns answer, "
          "confidence, scores per option and sure. Options may be short labels "
          "('billing') or full statements ending in a period ('The customer wants a "
          "refund.'); statements work better. 2-50 options. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_decide(input: str, options: list[str]) -> dict:
    return get_decider().decide(input, options).to_dict()


@mcp.tool(description="yn_check over many texts in one call. Much faster than calling "
          "yn_check repeatedly. Returns one result per input, in order. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_check_batch(inputs: list[str], claim: str) -> dict:
    return {"results": [r.to_dict() for r in get_decider().check_many(inputs, claim)]}


@mcp.tool(description="yn_decide over many texts in one call. Much faster than calling "
          "yn_decide repeatedly. Returns one result per input, in order. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_decide_batch(inputs: list[str], options: list[str]) -> dict:
    return {"results": [r.to_dict() for r in get_decider().decide_many(inputs, options)]}


def _warm_up() -> None:
    try:
        get_decider().load()
    except Exception as e:  # surfaced again on the first tool call
        print(f"yn-mcp: model load failed: {e}", file=sys.stderr)


def main() -> None:
    # Load the model in the background so the MCP handshake isn't delayed.
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
