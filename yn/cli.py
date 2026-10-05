"""`yn` terminal command.

    yn check "This email is spam." "You won a free iPhone!"
    yn decide "My card was charged twice" billing shipping technical
    echo "My card was charged twice" | yn decide - billing shipping technical
    yn check "This is urgent." --lines < subjects.txt
    yn route "Fix the typo in the README title"        # which model should do this?

Exit codes with --exit-code (for scripts and hooks):
    check:  0 = true, 1 = false, 2 = not sure
    decide, route: 0 = sure, 2 = not sure
Always, with or without --exit-code:
    64 = bad request (usage, input), 3 = setup or model error.
    Anything other than 0/1/2 is an error. Never read it as an answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from yn.model import DEFAULT_MODEL, Decider, InputError
from yn.route import read_routes_file, route_many

EXIT_FALSE, EXIT_UNSURE, EXIT_ERROR, EXIT_USAGE = 1, 2, 3, 64


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        # argparse's default exit 2 would read as "not sure".
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _threshold(value: str) -> float:
    try:
        t = float(value)
    except ValueError:
        t = -1.0
    if not 0 <= t <= 1:
        raise argparse.ArgumentTypeError(f"must be a number from 0 to 1, got {value!r}")
    return t


def _two_places(confidence: float) -> str:
    """Round down, so 0.8496 shows as 0.84, never as a 0.85 that looks sure."""
    return f"{math.floor(round(confidence * 100, 6)) / 100:.2f}"


def _read_inputs(text: str | None, lines: bool) -> list[str]:
    if text is None:
        if sys.stdin.isatty():
            raise InputError("no input. Pass text as an argument or pipe it in.")
        text = sys.stdin.read()
    if not lines:
        return [text]
    inputs = [ln for ln in text.splitlines() if ln.strip()]
    if not inputs:
        raise InputError("--lines got no non-blank lines")
    return inputs


def _stdin_has_data() -> bool:
    """True when stdin is a pipe or a file, not a terminal or /dev/null."""
    import stat

    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError, AttributeError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISREG(mode)


def _decide_args(args) -> None:
    """Split `yn decide TEXT OPTION OPTION...` into text and options.

    With -o, the only positional is the text, as before. Without it, the first is the
    text ("-" for stdin) and the rest are the options.
    """
    words = args.words
    if args.options:
        if len(words) > 1:
            raise InputError("with -o, pass the text as one quoted argument, or list the "
                             "options after the text instead of using -o")
        args.text = words[0] if words else None
    else:
        if len(words) < 3:
            raise InputError('give the text, then at least two options: '
                             'yn decide "hi" greeting question (or - as the text to '
                             'read it from stdin)')
        args.text, args.options = words[0], words[1:]
        if args.text != "-" and _stdin_has_data():
            # Dropping -o from an old `... | yn decide -o a -o b` command would
            # otherwise judge the first option as the text, without a word.
            print("yn: warning: stdin ignored; the first argument is the text. "
                  "Use - as the text to read it from stdin.", file=sys.stderr)
    if args.text == "-":
        args.text = None


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="yn", description="Fast true/false and pick-one decisions.")
    common = _Parser(add_help=False)
    common.add_argument("--json", action="store_true", help="print full JSON results")
    common.add_argument("--lines", action="store_true", help="treat each input line separately")
    common.add_argument("--threshold", type=_threshold, help="confidence needed to count as sure")
    common.add_argument("--exit-code", action="store_true", help="set exit code from the answer")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    c = sub.add_parser("check", parents=[common], help="is a statement true of the text?")
    c.add_argument("claim", help='statement to test, e.g. "This email is spam."')
    c.add_argument("text", nargs="?", help="text to judge (default: stdin)")

    d = sub.add_parser("decide", parents=[common], help="pick the best option for the text")
    d.add_argument("-o", "--option", action="append", dest="options",
                   help="an option; repeat for each. Or list the options after the text.")
    d.add_argument("words", nargs="*", metavar="TEXT_AND_OPTIONS",
                   help='the text, then its options: yn decide "hi" greeting question. '
                        'Use - for the text to read it from stdin.')

    r = sub.add_parser("route", parents=[common], help="pick a model for a task")
    r.add_argument("--routes", metavar="FILE",
                   help='JSON list of {"model": ..., "when": ...} (default: YN_ROUTES or '
                        "built-in Claude models)")
    r.add_argument("text", nargs="?", help="task description (default: stdin)")

    e = sub.add_parser("export-onnx",
                       help="export the model for the faster, lighter ONNX backend")
    e.add_argument("--model", help="model to export (default: YN_MODEL, else the built-in)")
    e.add_argument("--out", metavar="DIR",
                   help="where to write it (default: YN_ONNX_DIR, else the yn cache)")
    return p


def _export_onnx(args) -> int:
    from yn.onnx_backend import _slug, export, export_dir

    model = args.model or os.environ.get("YN_MODEL") or DEFAULT_MODEL
    out = export(model, args.out)
    print(f"exported {model} to {out}", file=sys.stderr)
    # Only claim automatic pickup when this really is the directory YN will look in
    # for the model it will actually run. --out or --model can make it neither.
    runs_this_model = model == (os.environ.get("YN_MODEL") or DEFAULT_MODEL)
    if runs_this_model and out == export_dir(model):
        print("the onnx backend is used automatically from now on; "
              "set YN_BACKEND=torch to opt out.", file=sys.stderr)
    elif out.name != _slug(model):
        # YN looks in YN_ONNX_DIR/<slug>, so no setting can point it at this folder.
        print(f"to use it, the folder must be named {_slug(model)}: re-run with "
              f"--out {out.parent / _slug(model)}", file=sys.stderr)
    else:
        needs = [] if runs_this_model else [f"YN_MODEL={model}"]
        if out != export_dir(model):
            needs.append(f"YN_ONNX_DIR={out.parent}")
        print("to use it, set " + " and ".join(needs), file=sys.stderr)
    return 0


def _parse(argv: list[str] | None):
    """parse_args, except decide's options may sit on either side of a flag.

    argparse stops a list of positionals at the first flag, so `yn decide hi a --json b`
    would leave "b" unrecognized. Words left over like that are more options.
    """
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        if args.cmd == "decide" and not any(e.startswith("-") and e != "-" for e in extra):
            args.words += extra
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.cmd == "export-onnx":
        try:
            return _export_onnx(args)
        except ImportError as e:
            print(f"yn: export needs the export extra: pip install 'yn[export]' ({e})",
                  file=sys.stderr)
            return EXIT_ERROR
        except Exception as e:  # bad model name, no entailment label, disk full
            print(f"yn: error: {type(e).__name__}: {e}", file=sys.stderr)
            return EXIT_ERROR
    try:
        if args.cmd == "decide":
            _decide_args(args)
        inputs = _read_inputs(args.text, args.lines)
        decider = Decider(threshold=args.threshold)
        if args.cmd == "check":
            results = decider.check_many(inputs, args.claim)
        elif args.cmd == "decide":
            results = decider.decide_many(inputs, args.options)
        else:
            routes = read_routes_file(args.routes) if args.routes else None
            results = route_many(decider, inputs, routes)
    except InputError as e:
        print(f"yn: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:  # model missing, bad config, library failure
        print(f"yn: error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR

    try:
        for r in results:
            if args.json:
                print(json.dumps(r.to_dict()))
            else:
                print(f"{r.answer}\t{_two_places(r.confidence)}" + ("" if r.sure else "\tunsure"))
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader stopped early (e.g. `| head -1`). Exit quietly, and not with 1,
        # which would read as "false". Point stdout at devnull so Python's own
        # flush at exit doesn't raise again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_ERROR

    if not args.exit_code:
        return 0
    if any(not r.sure for r in results):
        return EXIT_UNSURE
    if args.cmd == "check" and any(r.answer == "false" for r in results):
        return EXIT_FALSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
