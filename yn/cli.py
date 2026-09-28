"""`yn` terminal command.

    yn check "This email is spam." "You won a free iPhone!"
    echo "My card was charged twice" | yn decide -o billing -o shipping -o technical
    yn check "This is urgent." --lines < subjects.txt

Exit codes with --exit-code (for scripts and hooks):
    check:  0 = true, 1 = false, 2 = not sure
    decide: 0 = sure, 2 = not sure
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from yn.model import Decider

EXIT_FALSE, EXIT_UNSURE, EXIT_USAGE = 1, 2, 64


def _two_places(confidence: float) -> str:
    """Round down, so 0.8496 shows as 0.84, never as a 0.85 that looks sure."""
    return f"{math.floor(round(confidence * 100, 6)) / 100:.2f}"


def _read_inputs(text: str | None, lines: bool) -> list[str]:
    if text is None:
        if sys.stdin.isatty():
            raise SystemExit("yn: no input. Pass text as an argument or pipe it in.")
        text = sys.stdin.read()
    if lines:
        return [ln for ln in text.splitlines() if ln.strip()]
    return [text]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yn", description="Fast true/false and pick-one decisions.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print full JSON results")
    common.add_argument("--lines", action="store_true", help="treat each input line separately")
    common.add_argument("--threshold", type=float, help="confidence needed to count as sure")
    common.add_argument("--exit-code", action="store_true", help="set exit code from the answer")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", parents=[common], help="is a statement true of the text?")
    c.add_argument("claim", help='statement to test, e.g. "This email is spam."')
    c.add_argument("text", nargs="?", help="text to judge (default: stdin)")

    d = sub.add_parser("decide", parents=[common], help="pick the best option for the text")
    d.add_argument("-o", "--option", action="append", required=True, dest="options",
                   help="an option; repeat for each (labels or full statements)")
    d.add_argument("text", nargs="?", help="text to judge (default: stdin)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _read_inputs(args.text, args.lines)
    try:
        decider = Decider(threshold=args.threshold)
        if args.cmd == "check":
            results = decider.check_many(inputs, args.claim)
        else:
            results = decider.decide_many(inputs, args.options)
    except (ValueError, RuntimeError) as e:
        print(f"yn: {e}", file=sys.stderr)
        return EXIT_USAGE

    for r in results:
        if args.json:
            print(json.dumps(r.to_dict()))
        else:
            print(f"{r.answer}\t{_two_places(r.confidence)}" + ("" if r.sure else "\tunsure"))

    if not args.exit_code:
        return 0
    if any(not r.sure for r in results):
        return EXIT_UNSURE
    if args.cmd == "check" and any(r.answer == "false" for r in results):
        return EXIT_FALSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
