"""Slow tests: load the real stand-in model. Run with `pytest -m slow`.

Assertions are loose (answer only) because exact scores depend on the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from yn.model import Decider

pytestmark = pytest.mark.slow

YN_MCP = Path(sys.executable).parent / "yn-mcp"


@pytest.fixture(scope="module")
def real_decider():
    d = Decider()
    d.load()
    return d


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_obvious_spam_is_spam(real_decider):
    r = real_decider.check("Congratulations!! You won a free iPhone, click here to claim",
                           "This email is spam.")
    assert r.answer == "true"


def test_normal_email_is_not_spam(real_decider):
    r = real_decider.check("Hi, attached are the Q3 invoices you asked for. Thanks, Mei",
                           "This email is spam.")
    assert r.answer == "false"


def test_shipping_question_picks_shipping(real_decider):
    options = [
        "The customer has a billing problem.",
        "The customer is asking about a delivery.",
        "The customer has a technical problem.",
    ]
    r = real_decider.decide("Where is my package? It's been 2 weeks", options)
    assert r.answer == options[1]


@pytest.mark.anyio
async def test_mcp_stdio_round_trip(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    assert YN_MCP.exists(), f"{YN_MCP} missing; run `pip install -e .`"
    with open(tmp_path / "server-stderr.txt", "w") as errlog:
        params = StdioServerParameters(command=str(YN_MCP))
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == {
                    "yn_check", "yn_decide", "yn_check_batch", "yn_decide_batch"}
                result = await session.call_tool("yn_check", {
                    "input": "URGENT: production database is down, customers can't pay",
                    "claim": "This message is urgent.",
                })
                assert result.is_error is False
                assert json.loads(result.content[0].text)["answer"] == "true"


def test_long_non_latin_claim_is_clear_input_error(real_decider):
    from yn.model import MAX_STATEMENT_CHARS, InputError
    claim = "这封邮件非常紧急需要马上处理" * 70  # under the character limit, over 400 tokens
    assert len(claim) < MAX_STATEMENT_CHARS
    with pytest.raises(InputError, match="tokens"):
        real_decider.check("服务器宕机了", claim)
    # The model still works afterwards.
    assert real_decider.check("You won a free iPhone, click here!", "This email is spam.").answer == "true"
