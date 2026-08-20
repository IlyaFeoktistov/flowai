"""mcp_agent/plugin_hooks.py:PluginHookMiddleware — post_file_edit runs
AFTER a successful write_file/edit_file (best-effort, a hook raising never
fails the edit); pre_commit runs BEFORE a `git commit` bash call and can
block it outright by returning a reason string. Also covers the built-in
(not plugin-declared) auto-reindex-on-file-touch: read_file/write_file/
edit_file success schedules a fire-and-forget background task that keeps
the semantic code index fresh for that ONE file (rag/index_code.py)."""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from mcp_agent import plugin_hooks
# Captured before the autouse _stub_auto_reindex fixture below replaces
# plugin_hooks._auto_reindex_file with a no-op for every OTHER test — the
# two tests that exercise _auto_reindex_file itself need the real
# function, not whatever the module attribute currently points to.
from mcp_agent.plugin_hooks import _auto_reindex_file as _real_auto_reindex_file


def _request(name, args, tool_call_id="1"):
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": tool_call_id})


async def _passthrough_handler(request):
    return ToolMessage(content="ok", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])


@pytest.fixture(autouse=True)
def _stub_auto_reindex(monkeypatch):
    """Every test in this file except the ones that specifically exercise
    auto-reindex gets a harmless no-op here — without this, the tests
    above (fake paths like "/repo"/"a.py") would schedule a REAL
    background task that tries to resolve those paths on disk and, absent
    _resolve_targets's "path doesn't exist" early-out, could reach all the
    way to a live Ollama embedding call. Tests that DO want to verify the
    scheduling itself override this again locally."""
    async def _noop(path, repo_path):
        pass
    monkeypatch.setattr(plugin_hooks, "_auto_reindex_file", _noop)


@pytest.mark.asyncio
async def test_post_file_edit_hook_runs_after_a_successful_write(monkeypatch):
    calls = []
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda path, repo: calls.append((kind, path, repo))] if kind == "post_file_edit" else [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    result = await middleware.awrap_tool_call(_request("write_file", {"path": "a.py"}), _passthrough_handler)

    assert result.content == "ok"
    assert calls == [("post_file_edit", "a.py", "/repo")]


@pytest.mark.asyncio
async def test_post_file_edit_hook_does_not_run_after_a_failed_write(monkeypatch):
    calls = []
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda path, repo: calls.append(path)])

    async def _failing_handler(request):
        return ToolMessage(content="boom", name="write_file", tool_call_id="1", status="error")

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await middleware.awrap_tool_call(_request("write_file", {"path": "a.py"}), _failing_handler)

    assert calls == []


@pytest.mark.asyncio
async def test_post_file_edit_hook_raising_does_not_break_the_tool_result(monkeypatch):
    def _boom(path, repo):
        raise RuntimeError("plugin bug")

    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [_boom])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    result = await middleware.awrap_tool_call(_request("edit_file", {"path": "a.py"}), _passthrough_handler)

    assert result.content == "ok"  # the edit itself still succeeded


@pytest.mark.asyncio
async def test_non_edit_tool_calls_are_not_touched_by_post_file_edit_hooks(monkeypatch):
    calls = []
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda *a: calls.append(a)])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await middleware.awrap_tool_call(_request("read_file", {"path": "a.py"}), _passthrough_handler)

    assert calls == []


@pytest.mark.asyncio
async def test_pre_commit_hook_blocks_when_it_returns_a_reason(monkeypatch):
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda cmd, repo: "secrets detected"] if kind == "pre_commit" else [])
    handler_called = []

    async def _handler(request):
        handler_called.append(True)
        return ToolMessage(content="committed", name="bash", tool_call_id="1")

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    result = await middleware.awrap_tool_call(_request("bash", {"command": "git commit -m x"}), _handler)

    assert handler_called == []
    assert "secrets detected" in result.content
    assert result.status != "error"  # a deliberate block, not a tool failure


@pytest.mark.asyncio
async def test_pre_commit_hook_allows_when_it_returns_nothing(monkeypatch):
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda cmd, repo: None] if kind == "pre_commit" else [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    result = await middleware.awrap_tool_call(_request("bash", {"command": "git commit -m x"}), _passthrough_handler)

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_pre_commit_hook_not_triggered_by_unrelated_git_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda *a: calls.append(a)])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await middleware.awrap_tool_call(_request("bash", {"command": "git status"}), _passthrough_handler)

    assert calls == []


@pytest.mark.asyncio
async def test_pre_commit_hook_ignores_git_commit_hidden_later_in_a_chain(monkeypatch):
    """Only the first pipeline segment is checked — a hook can't vet a
    commit it was never shown, and there's no reliable way to tell "this
    chain WILL run git commit" from "this string merely mentions it"."""
    calls = []
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [lambda *a: calls.append(a)])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await middleware.awrap_tool_call(_request("bash", {"command": "echo hi && git commit -m x"}), _passthrough_handler)

    assert calls == []


@pytest.mark.asyncio
async def test_pre_commit_hook_raising_does_not_block_the_commit(monkeypatch):
    """A broken plugin hook is a plugin bug, not a reason to make every
    future commit impossible — see the middleware's own console warning."""
    def _boom(cmd, repo):
        raise RuntimeError("plugin bug")

    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [_boom] if kind == "pre_commit" else [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    result = await middleware.awrap_tool_call(_request("bash", {"command": "git commit -m x"}), _passthrough_handler)

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_async_hooks_are_awaited(monkeypatch):
    calls = []

    async def _async_hook(path, repo):
        calls.append((path, repo))

    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda kind, repo=None: [_async_hook] if kind == "post_file_edit" else [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await middleware.awrap_tool_call(_request("write_file", {"path": "a.py"}), _passthrough_handler)

    assert calls == [("a.py", "/repo")]


async def _run_and_let_background_tasks_run(middleware, request, handler=_passthrough_handler):
    result = await middleware.awrap_tool_call(request, handler)
    await asyncio.sleep(0)  # let the scheduled asyncio.create_task actually execute
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file"])
async def test_auto_reindex_scheduled_after_successful_read_or_edit(monkeypatch, tool_name):
    calls = []

    async def _fake_auto_reindex(path, repo_path):
        calls.append((path, repo_path))
    monkeypatch.setattr(plugin_hooks, "_auto_reindex_file", _fake_auto_reindex)
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda *a, **k: [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await _run_and_let_background_tasks_run(middleware, _request(tool_name, {"path": "a.py"}))

    assert calls == [("a.py", "/repo")]


@pytest.mark.asyncio
async def test_auto_reindex_not_scheduled_for_unrelated_tools(monkeypatch):
    calls = []

    async def _fake_auto_reindex(path, repo_path):
        calls.append((path, repo_path))
    monkeypatch.setattr(plugin_hooks, "_auto_reindex_file", _fake_auto_reindex)
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda *a, **k: [])

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await _run_and_let_background_tasks_run(middleware, _request("bash", {"command": "ls"}))

    assert calls == []


@pytest.mark.asyncio
async def test_auto_reindex_not_scheduled_after_a_failed_write(monkeypatch):
    calls = []

    async def _fake_auto_reindex(path, repo_path):
        calls.append((path, repo_path))
    monkeypatch.setattr(plugin_hooks, "_auto_reindex_file", _fake_auto_reindex)
    monkeypatch.setattr(plugin_hooks, "load_hooks", lambda *a, **k: [])

    async def _failing_handler(request):
        return ToolMessage(content="boom", name="write_file", tool_call_id="1", status="error")

    middleware = plugin_hooks.PluginHookMiddleware("/repo")
    await _run_and_let_background_tasks_run(middleware, _request("write_file", {"path": "a.py"}), _failing_handler)

    assert calls == []


@pytest.mark.asyncio
async def test_auto_reindex_file_delegates_to_reindex_code_from_disk(monkeypatch):
    calls = []

    async def _fake_reindex_from_disk(repo_path, targets=None):
        calls.append((repo_path, targets))
        return {"chunks": 1}
    monkeypatch.setattr(plugin_hooks, "reindex_code_from_disk", _fake_reindex_from_disk)

    await _real_auto_reindex_file("a.py", "/repo")

    assert calls == [("/repo", ["a.py"])]


@pytest.mark.asyncio
async def test_auto_reindex_file_failure_is_logged_not_raised(monkeypatch):
    logged = []

    async def _boom(repo_path, targets=None):
        raise ConnectionError("ollama unreachable")
    monkeypatch.setattr(plugin_hooks, "reindex_code_from_disk", _boom)
    monkeypatch.setattr(plugin_hooks, "log_event", lambda event, **kw: logged.append((event, kw)))

    await _real_auto_reindex_file("a.py", "/repo")  # must not raise

    assert logged[0][0] == "auto_reindex_failed"
    assert logged[0][1]["path"] == "a.py"
