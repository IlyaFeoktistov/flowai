"""
PluginHookMiddleware — runs plugin-declared hooks (mcp_agent/plugins.py)
at two points every role's tool loop already passes through:

- post_file_edit: after a successful write_file/edit_file call — a plugin
  can react to a file changing (e.g. run a formatter, update an index).
  Best-effort and after the fact: a hook raising never undoes the write or
  fails the tool call, it only gets reported to the console — a plugin's
  own bug shouldn't make an unrelated edit look like it failed.
- pre_commit: before a bash call whose command is (or starts with, once
  chained with && / ; / |) `git commit` — a plugin can block the commit
  outright (e.g. a lint/secret-scan gate) by returning a non-empty reason
  string. Unlike post_file_edit this runs BEFORE the real tool call, so a
  hook's decision actually has an effect: returning a reason skips running
  git entirely and hands the model back an ordinary (non-error) ToolMessage
  explaining why, the same shape every other approval-style rejection in
  this codebase already uses (see ask_user_tool.py's guard middlewares).

Both hook kinds accept a plain function or an async one — this middleware
already runs in an async context, so no code should need to know which
kind a given plugin/skill author chose to write.

Hooks come from two sources, both surfaced through mcp_agent/plugins.py's
load_hooks(hook_name, repo_path): global plugins (<repo_root>/plugins/)
and, since repo_path is always known here, the current project's own
<repo_path>/.flowai/hooks/*.py files — see that module's docstring for
the full story on why the two are loaded so differently.
"""
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from mcp_agent.plugins import load_hooks

_EDIT_PATH_ARG_NAMES = {"write_file": "path", "edit_file": "path"}

# First pipeline segment only (before &&/;/|) — a hook can't meaningfully
# vet a commit hidden behind other commands it was never shown, and a
# command that merely MENTIONS "git commit" later in a chain (e.g. inside
# a string argument to echo) isn't actually running one.
_GIT_COMMIT_RE = re.compile(r"^\s*git\s+commit\b")


async def _call_hook(func, *args):
    result = func(*args)
    if hasattr(result, "__await__"):
        result = await result
    return result


class PluginHookMiddleware(AgentMiddleware):
    def __init__(self, repo_path: str):
        self._repo_path = repo_path

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call.get("args") or {}

        if name == "bash":
            command = args.get("command") or ""
            first_segment = re.split(r"&&|;|\|", command, maxsplit=1)[0]
            if _GIT_COMMIT_RE.match(first_segment):
                for hook in load_hooks("pre_commit", self._repo_path):
                    try:
                        reason = await _call_hook(hook, command, self._repo_path)
                    except Exception as e:
                        reason = None
                        from ui.console import console
                        console.print(f"[yellow]⚠ pre_commit хук упал ({e}) — коммит пропущен без блокировки[/]")
                    if reason:
                        return ToolMessage(
                            content=f"Commit blocked by plugin hook: {reason}",
                            name=name, tool_call_id=request.tool_call["id"], status="success",
                        )

        result = await handler(request)

        arg_name = _EDIT_PATH_ARG_NAMES.get(name)
        if arg_name and isinstance(result, ToolMessage) and getattr(result, "status", "success") != "error":
            path = args.get(arg_name)
            if path:
                for hook in load_hooks("post_file_edit", self._repo_path):
                    try:
                        await _call_hook(hook, path, self._repo_path)
                    except Exception as e:
                        from ui.console import console
                        console.print(f"[yellow]⚠ post_file_edit хук упал на {path!r}: {e}[/]")
        return result
