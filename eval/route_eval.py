# Copyright 2026 bryanfrds (https://github.com/bryanfrds)
# SPDX-License-Identifier: Apache-2.0
"""Measure `gauge route` on eval/route_tasks.json with the built-in routes.

    .venv/bin/python eval/route_eval.py

Twelve hand-written tasks is a smoke test, not a benchmark: use it to compare
route wordings or models, not to claim accuracy.
"""

import json
from pathlib import Path

from gauge.model import get_decider
from gauge.route import DEFAULT_ROUTES, route_many

ORDER = [r["model"] for r in DEFAULT_ROUTES]  # cheapest first
cases = json.loads((Path(__file__).parent / "route_tasks.json").read_text())
results = route_many(get_decider(), [c["task"] for c in cases])

right = off_by_more = sure = 0
for c, r in zip(cases, results):
    gap = abs(ORDER.index(r.answer) - ORDER.index(c["expect"]))
    right += gap == 0
    off_by_more += gap > 1
    sure += r.sure
    mark = "ok" if gap == 0 else f"off by {gap}"
    print(f"{mark:9} {r.answer:17} {r.confidence:.2f}{'' if r.sure else '?'}  {c['task'][:60]}")
n = len(cases)
print(f"\n{right}/{n} right, {off_by_more} off by 2+ levels, {sure}/{n} sure")
