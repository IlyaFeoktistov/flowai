"""_AskUserFinalizeNumPredictMiddleware (mcp_agent/ask_user_tool.py) — caps
the model's budget for the ONE turn right after a real ask_user answer, so
a model that re-derives instead of restating its plan (live incident,
2026-08-19: ~1900 tokens/~2min against the full OLLAMA_NUM_PREDICT budget)
gets cut off far sooner. No real LangChain agent/model involved — a fake
request/handler is enough to exercise the branching logic."""
from types import SimpleNamespace

import pytest
from langchain_ollama import ChatOllama

from conftest import tool_message
from mcp_agent.ask_user_tool import _AskUserFinalizeNumPredictMiddleware
from mcp_agent.model_config import ASK_USER_FINALIZE_NUM_PREDICT


class _FakeOtherModel:
    """Stands in for the expert-streaming backend's ChatOpenAI instance —
    only needs to NOT be a ChatOllama."""


def _make_request(messages, model):
    overridden = {}

    def _override(**kwargs):
        overridden.update(kwargs)
        return SimpleNamespace(messages=messages, model=model, model_settings=kwargs)

    request = SimpleNamespace(messages=messages, model=model, override=_override)
    return request, overridden


async def _handler(request):
    return request  # echoes back whatever the middleware passed through


@pytest.mark.asyncio
async def test_passthrough_before_ask_user_answered():
    """No ask_user ToolMessage yet — budget must stay untouched (Planner's
    investigation/plan-drafting/ask_user-call turns all legitimately need
    the full OLLAMA_NUM_PREDICT, e.g. a long `options` argument)."""
    request, overridden = _make_request([tool_message("read_file", content="...")], ChatOllama(model="x"))
    result = await _AskUserFinalizeNumPredictMiddleware().awrap_model_call(request, _handler)
    assert result is request
    assert overridden == {}


@pytest.mark.asyncio
async def test_passthrough_when_ask_user_errored():
    """A failed ask_user call (bad args -> ToolInvocationError) never
    reached the user — must not count as "answered", same guard as
    _AskUserFinalizeMiddleware's own status != "error" check."""
    request, overridden = _make_request(
        [tool_message("ask_user", content="bad args", status="error")], ChatOllama(model="x"),
    )
    result = await _AskUserFinalizeNumPredictMiddleware().awrap_model_call(request, _handler)
    assert result is request
    assert overridden == {}


@pytest.mark.asyncio
async def test_caps_budget_for_ollama_after_real_answer():
    messages = [tool_message("ask_user", content="Только Go (удалить C-код)")]
    request, overridden = _make_request(messages, ChatOllama(model="x"))
    await _AskUserFinalizeNumPredictMiddleware().awrap_model_call(request, _handler)
    assert overridden == {
        "model_settings": {"options": {"num_ctx": overridden["model_settings"]["options"]["num_ctx"], "num_predict": ASK_USER_FINALIZE_NUM_PREDICT}},
    }
    assert overridden["model_settings"]["options"]["num_predict"] == ASK_USER_FINALIZE_NUM_PREDICT


@pytest.mark.asyncio
async def test_caps_budget_for_non_ollama_after_real_answer():
    """expert-streaming's ChatOpenAI backend takes max_tokens directly, no
    options{}/num_ctx (see the middleware's own docstring)."""
    messages = [tool_message("ask_user", content="Только Go (удалить C-код)")]
    request, overridden = _make_request(messages, _FakeOtherModel())
    await _AskUserFinalizeNumPredictMiddleware().awrap_model_call(request, _handler)
    assert overridden == {"model_settings": {"max_tokens": ASK_USER_FINALIZE_NUM_PREDICT}}
