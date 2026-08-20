"""
PluginHookMiddleware — runs plugin-declared hooks (mcp_agent/plugins.py)
at two points every role's tool loop already passes through, PLUS one
built-in (not plugin-declared) behavior at the same first point:

- auto-reindex: after any successful read_file/write_file/edit_file call,
  fire-and-forget refresh that ONE file's chunks in the semantic code
  index (rag/index_code.py) in the background — keeps the index "alive"
  incrementally as files actually get touched, instead of only updating
  on a manual, whole-project /reindex. Builds the index up from nothing
  if none exists yet (no separate "only if already built" gate) — see
  _auto_reindex_file below. Never awaited inline (asyncio.create_task),
  so it can never add latency to the read/write/edit call that triggered
  it, and a failure there (Ollama down, embedding model not pulled, ...)
  can never make an otherwise-successful file operation look like it
  failed — see debug_log's "auto_reindex_failed" event for diagnosing a
  silently-not-updating index instead.
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

Both hook kinds (post_file_edit/pre_commit) accept a plain function or an
async one — this middleware already runs in an async context, so no code
should need to know which kind a given plugin/skill author chose to write.

Hooks come from two sources, both surfaced through mcp_agent/plugins.py's
load_hooks(hook_name, repo_path): global plugins (<repo_root>/plugins/)
and, since repo_path is always known here, the current project's own
<repo_path>/.flowai/hooks/*.py files — see that module's docstring for
the full story on why the two are loaded so differently.
"""
import asyncio
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from mcp_agent.debug_log import log_event
from mcp_agent.plugins import load_hooks
from rag.index_code import reindex_code_from_disk

_EDIT_PATH_ARG_NAMES = {"write_file": "path", "edit_file": "path"}
# Same set PLUS read_file — auto-reindex is triggered by reads too (a file
# nobody's edited yet still deserves to be IN the index, not just files
# that happen to get written), post_file_edit plugin hooks stay
# write/edit-only (that's their own documented contract, unrelated to
# this built-in feature).
_INDEXABLE_PATH_ARG_NAMES = {**_EDIT_PATH_ARG_NAMES, "read_file": "path"}

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


async def _auto_reindex_file(path: str, repo_path: str) -> None:
    """Fire-and-forget — see module docstring's "auto-reindex" section.
    reindex_code_from_disk (not a raw VectorStore.load+reindex_code here)
    — it owns its own per-repo_path lock, so this doesn't race a manual
    /reindex or another auto-reindex task from a DIFFERENT file touched
    moments earlier in the same turn."""
    try:
        await reindex_code_from_disk(repo_path, targets=[path])
    except Exception as e:
        log_event("auto_reindex_failed", path=path, error=str(e))


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
        succeeded = isinstance(result, ToolMessage) and getattr(result, "status", "success") != "error"

        indexable_arg_name = _INDEXABLE_PATH_ARG_NAMES.get(name)
        if succeeded and indexable_arg_name:
            path = args.get(indexable_arg_name)
            if path:
                asyncio.create_task(_auto_reindex_file(path, self._repo_path))

        arg_name = _EDIT_PATH_ARG_NAMES.get(name)
        if succeeded and arg_name:
            path = args.get(arg_name)
            if path:
                for hook in load_hooks("post_file_edit", self._repo_path):
                    try:
                        await _call_hook(hook, path, self._repo_path)
                    except Exception as e:
                        from ui.console import console
                        console.print(f"[yellow]⚠ post_file_edit хук упал на {path!r}: {e}[/]")
        return result
