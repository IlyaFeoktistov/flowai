"""_base_agent_middleware's pre_hitl ordering (mcp_agent/agent_builder.py) —
langchain composes middleware "first = outermost" (langchain.agents.
factory._chain_tool_call_wrappers's own docstring), so a mechanical
rejector meant to short-circuit BEFORE the human approval prompt must sit
earlier in the list than hitl_middleware, not later. A live run: Verifier's
`cat > file <<EOF` self-fix attempt sat through a real 41-second approval
wait, then got denied anyway regardless of the answer."""
from mcp_agent.agent_builder import _base_agent_middleware


class _DummyMiddleware:
    pass


def test_pre_hitl_middlewares_come_before_hitl_middleware():
    hitl = _DummyMiddleware()
    pre = _DummyMiddleware()
    result = _base_agent_middleware("/tmp/repo", hitl, [pre])
    assert result.index(pre) < result.index(hitl)


def test_no_pre_hitl_still_works():
    hitl = _DummyMiddleware()
    result = _base_agent_middleware("/tmp/repo", hitl)
    assert hitl in result


def test_multiple_pre_hitl_all_come_before_hitl_middleware_in_order():
    hitl = _DummyMiddleware()
    a, b = _DummyMiddleware(), _DummyMiddleware()
    result = _base_agent_middleware("/tmp/repo", hitl, [a, b])
    assert result.index(a) < result.index(b) < result.index(hitl)
