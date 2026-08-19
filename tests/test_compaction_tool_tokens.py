"""mcp_agent/compaction.py:_estimate_tool_tokens — memoizes the schema
serialization per bound tool object (perf audit finding: _needs_compaction
runs before EVERY model call in a stage's ReAct loop, but the bound tool
list never changes within one role-agent's lifetime, so re-deriving each
tool's json_schema() every round was pure repeated work)."""
from mcp_agent.compaction import _estimate_tool_tokens, _tool_token_cache


class _FakeSchema:
    def __init__(self):
        self.calls = 0

    def model_json_schema(self):
        self.calls += 1
        return {"type": "object", "properties": {"path": {"type": "string"}}}


class _FakeTool:
    def __init__(self, name="read_file", description="Read a file"):
        self.name = name
        self.description = description
        self.args_schema = _FakeSchema()


def test_second_call_for_the_same_tool_object_does_not_reserialize_schema():
    tool = _FakeTool()
    first = _estimate_tool_tokens(tool)
    second = _estimate_tool_tokens(tool)
    assert first == second
    assert tool.args_schema.calls == 1


def test_different_tool_objects_are_estimated_independently():
    a, b = _FakeTool(name="a", description="short"), _FakeTool(name="bbbbbbbbbb", description="a much longer description here")
    assert _estimate_tool_tokens(a) != _estimate_tool_tokens(b)
    assert a.args_schema.calls == 1
    assert b.args_schema.calls == 1


def test_dict_shaped_tools_are_never_cached():
    before = len(_tool_token_cache)
    _estimate_tool_tokens({"name": "x", "parameters": {}})
    assert len(_tool_token_cache) == before
