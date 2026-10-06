"""The credits the Apache 2.0 licence obliges redistributors to keep."""

from __future__ import annotations

from pathlib import Path

import pytest

import gauge
from gauge.cli import main

ROOT = Path(__file__).resolve().parents[1]
HEADER = "# Copyright 2026 bryanfrds (https://github.com/bryanfrds)\n# SPDX-License-Identifier: Apache-2.0\n"


def test_version_names_the_author(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "bryanfrds" in out and gauge.__version__ in out


def test_notice_file_credits_the_author():
    notice = (ROOT / "NOTICE").read_text()
    assert "Copyright 2026 bryanfrds" in notice
    assert "Apache License, Version 2.0" in notice


def test_the_claude_and_codex_tool_credits_the_author():
    from gauge.mcp_server import mcp

    assert "by bryanfrds" in mcp.instructions


@pytest.mark.parametrize("path", sorted(
    p.relative_to(ROOT).as_posix()
    for folder in ("gauge", "train", "eval") for p in (ROOT / folder).rglob("*.py")))
def test_every_source_file_starts_with_the_copyright_header(path):
    assert (ROOT / path).read_text().startswith(HEADER), f"{path} is missing the header"
