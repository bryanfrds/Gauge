# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Pick a model for a task (`gauge route`).

A route is a model plus a statement describing the tasks it suits. Routing is
`Decider.decide_many` over those statements, with answers mapped back to model names.
"""

from __future__ import annotations

import json
import os

from gauge.model import MAX_OPTIONS, MAX_STATEMENT_CHARS, Decider, Decision, InputError

# Current Claude models, cheapest first. Statements are full sentences because the
# stand-in model scores those far better than bare labels.
DEFAULT_ROUTES = [
    {"model": "claude-haiku-4-5",
     "when": "This is a quick, simple task, like a small edit, a lookup, renaming "
             "something, formatting, or a one-line answer."},
    {"model": "claude-sonnet-5",
     "when": "This is an everyday task of moderate size, like writing a function, "
             "fixing an ordinary bug, or drafting a short document."},
    {"model": "claude-opus-5-5",
     "when": "This is a hard task that needs careful reasoning, like debugging a tricky "
             "problem, designing a system, or changing many files."},
    {"model": "claude-fable-5-1",
     "when": "This is a very hard, long or high-stakes task, like a large multi-step "
             "project, deep research, or a problem other models failed at."},
]


def check_routes(routes) -> list[tuple[str, str]]:
    """Validate routes; return (model, statement) pairs. Raises InputError."""
    if not isinstance(routes, list) or not 2 <= len(routes) <= MAX_OPTIONS:
        raise InputError(f"routes must be a list of 2-{MAX_OPTIONS} items")
    pairs = []
    for i, r in enumerate(routes):
        if not isinstance(r, dict) or set(r) != {"model", "when"}:
            raise InputError(f'route {i} must have exactly "model" and "when"')
        model, when = r["model"], r["when"]
        if not all(isinstance(v, str) and v.strip() for v in (model, when)):
            raise InputError(f'route {i}: "model" and "when" must be non-empty text')
        when = when.strip()
        # A statement, so the model judges it directly instead of via a label template.
        # Length is checked after adding the period, on exactly what gets scored.
        statement = when if when[-1] in ".!?" else when + "."
        if len(statement) > MAX_STATEMENT_CHARS:
            raise InputError(f'route {i}: "when" is too long ({len(statement)} characters '
                             f"with its final period); the limit is {MAX_STATEMENT_CHARS}")
        pairs.append((model.strip(), statement))
    if len({m for m, _ in pairs}) != len(pairs):
        raise InputError("each route needs a different model")
    if len({w for _, w in pairs}) != len(pairs):
        raise InputError('each route needs a different "when" description')
    return pairs


def read_routes_file(path: str) -> list[dict]:
    """Routes from a JSON file: [{"model": ..., "when": ...}, ...]."""
    try:
        with open(path, encoding="utf-8") as f:
            routes = json.load(f)
    except (OSError, ValueError) as e:
        raise InputError(f"can't read routes file {path}: {e}") from e
    check_routes(routes)
    return routes


def default_routes() -> list[dict]:
    """GAUGE_ROUTES (a JSON file path) if set, else the built-in Claude routes.

    A bad GAUGE_ROUTES file is a setup problem, so it raises RuntimeError, not InputError.
    """
    path = os.environ.get("GAUGE_ROUTES")
    if not path:
        return DEFAULT_ROUTES
    try:
        return read_routes_file(path)
    except InputError as e:
        raise RuntimeError(f"GAUGE_ROUTES: {e}") from e


def route_many(decider: Decider, tasks: list[str], routes: list[dict] | None = None,
               ) -> list[Decision]:
    """Pick a model for each task. `answer` and `scores` use model names.

    Routes from GAUGE_ROUTES are server config, so problems with them raise RuntimeError
    (never InputError), and the caller isn't told its request was bad.
    """
    from_config = routes is None
    try:
        pairs = check_routes(default_routes() if from_config else routes)
        decider.check_statements([when for _, when in pairs])
    except InputError as e:
        if from_config:
            raise RuntimeError(f"GAUGE_ROUTES: {e}") from e
        raise
    model_for = {when: model for model, when in pairs}
    results = decider.decide_many(tasks, [when for _, when in pairs])
    return [
        Decision(model_for[r.answer], r.confidence,
                 {model_for[w]: s for w, s in r.scores.items()}, r.sure, r.model)
        for r in results
    ]
