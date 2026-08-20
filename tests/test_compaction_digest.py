"""_summarize_research's structured digest (mcp_agent/compaction.py) — two
separate concerns:

1. _format_digest renders _COMPACT_SYSTEM_PROMPT's JSON contract
   (files_read/actions_taken/key_facts/current_state) into the digest text
   that actually replaces raw history, extracting concrete facts instead of
   free prose, and skipping any section the model left empty.
2. _summarize_research must pass config=INTERNAL_JUDGE_CONFIG (an explicit
   empty callbacks list) to judge_model.ainvoke — otherwise LangChain's
   ensure_config() inherits whatever RunnableConfig is already ambient on
   the current asyncio task via contextvars, which inside
   _CompactResearchMiddleware.awrap_model_call is the SAME callback tree
   LangGraph wired up for the outer agent.astream(stream_mode=["messages"])
   call this middleware runs inside of — without cutting that inheritance,
   the judge's own raw JSON response rides the same "messages" stream as
   the real model's answer and can render in the UI as if it were the
   model's own reply (see agent.py:_stream_round, which only resets its
   buffer on a new message id, never checks which runnable produced a
   chunk)."""
import pytest
from langchain_core.messages import HumanMessage

from mcp_agent.compaction import _format_digest, _summarize_research
from mcp_agent.message_utils import INTERNAL_JUDGE_CONFIG


def test_format_digest_renders_every_section():
    data = {
        "files_read": [{"path": "a.go", "note": "defines handleRequest"}],
        "actions_taken": ["added STATUS_CANCELLED=50 to a.go"],
        "key_facts": ["OLLAMA_NUM_CTX=65536"],
        "current_state": "fix applied, verification pending",
    }
    result = _format_digest(data)
    assert "a.go: defines handleRequest" in result
    assert "added STATUS_CANCELLED=50 to a.go" in result
    assert "OLLAMA_NUM_CTX=65536" in result
    assert "fix applied, verification pending" in result


def test_format_digest_skips_empty_sections():
    result = _format_digest({"files_read": [], "actions_taken": [], "key_facts": [], "current_state": ""})
    assert result == ""


def test_format_digest_tolerates_plain_string_files_read():
    result = _format_digest({"files_read": ["a.go"], "actions_taken": [], "key_facts": [], "current_state": ""})
    assert "a.go" in result


class _FakeJudgeModel:
    def __init__(self, content):
        self._content = content
        self.calls = []

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls.append({"messages": messages, "config": config, "kwargs": kwargs})
        class _Resp:
            content = self._content
        return _Resp()


@pytest.mark.asyncio
async def test_summarize_research_cuts_ambient_callback_inheritance():
    judge = _FakeJudgeModel('{"files_read": [], "actions_taken": ["did X"], "key_facts": [], "current_state": ""}')
    prefix = [HumanMessage(content="task")]

    digest = await _summarize_research(judge, prefix)

    assert "did X" in digest
    assert judge.calls[0]["config"] == INTERNAL_JUDGE_CONFIG
    assert judge.calls[0]["config"]["callbacks"] == []
