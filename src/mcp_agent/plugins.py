"""
Two independent layers of user-provided extension, both loaded by this
module and exposed to cli.py (commands), mcp_agent/config.py (MCP
servers), and mcp_agent/plugin_hooks.py (hooks):

1. Global plugins — <repo_root>/plugins/<name>/plugin.json, one
   directory per plugin, git-ignored (see .gitignore's "/plugins/"
   entry) so a user's own plugins never end up in flowai's own git
   history. Co-located with the actual checkout rather than
   storage.data_dir() (~/.local/share/flowai/) on purpose: unlike
   memory/settings/usage (genuinely per-machine XDG state), a plugin is
   closer to "part of this particular flowai installation" — someone
   running more than one checkout (different machines, a dev copy vs a
   stable one) would want each to keep its own plugin set right next to
   it, not all of them sharing one hidden global directory. See the
   manifest shape below.

2. Per-project skills/hooks — <repo_path>/.flowai/skills/*.py and
   <repo_path>/.flowai/hooks/*.py, inside whatever project the user has
   open (repo_path — see mcp_agent/config.py's own docstring on where
   that comes from). Deliberately NO manifest, unlike global plugins:
   these are one-off extensions someone writes for the project they're
   sitting in right now, not something meant to be shared/versioned/
   distributed the way a real plugin is — a manifest would be pure
   ceremony for that case. A skill is just a .py file whose filename
   (minus ".py") becomes the command name; a hook file is scanned for
   well-known function names (post_file_edit/pre_commit) and whichever
   it defines get registered. Not cached across calls (unlike global
   plugin discovery below) — repo_path changes between projects/sessions
   within the same flowai process in a way the global plugins directory
   never does, and rescanning a couple of small directories is cheap
   enough that a repo_path-keyed cache isn't worth the complexity.

Global plugin manifest shape:
    {
      "name": "example",                       (required, must match the directory name)
      "version": "0.1.0",                       (required, informational only for now)
      "description": "...",                     (optional)
      "commands": {
        "hello": {"module": "hello.py", "function": "run", "help": "say hello"}
      },
      "mcp_servers": {
        "example": {"command": "python3", "args": ["server.py"]}
      },
      "hooks": {
        "post_file_edit": ["hooks.py:on_edit"],
        "pre_commit": ["hooks.py:on_commit"]
      }
    }
All of commands/mcp_servers/hooks are optional — a plugin can provide just
one of the three. `module`/`args` entries that name a file existing inside
the plugin's own directory are resolved to an absolute path; anything else
(an installed console script, a bare module name) is passed through
as-is, same convention as mcp_agent/config.py's own server commands.

Command function signature — `def run(args: str, console) -> None | str |
SkillTask` (sync or async, cli.py awaits it only if it returns an
awaitable): `args` is the raw text after "/hello ", `console` is
ui/console.py's Rich console (the same one every other command prints
through — printing anywhere else won't render inside the TUI, see
ui/console.py's own docstring). A global-plugin or project-skill command
can never SHADOW a built-in one — cli.py only checks these after every
"/xxx" it already knows about; a project skill CAN shadow a global
plugin's command of the same name (checked first — it's the more specific
of the two).

Returning `None` (or nothing) keeps the original behavior — a pure
side-effect command that stops right there. Returning a plain `str`, or a
`SkillTask` (see its own docstring below), hands the text off to a REAL
agent turn instead — cli.py feeds it into the exact same pipeline a
manually typed message goes through (stream_chat), so a skill can pre-fetch
real data (read a file directly, hit an API — whatever it needs) and hand
back a fully-formed task instead of hoping the model asks for the right
tool calls itself.

Hook function signatures — mcp_agent/plugin_hooks.py calls these, see its
own docstring for exactly when and with what arguments.

Every load function is best-effort per plugin/skill/hook file: one broken
manifest or module must not take down flowai startup or stop every OTHER
one from loading — each failure is caught, reported via console, and
skipped.

Global plugin discovery is cached for the process lifetime (like
agent_builder.py's other caches) — plugins are an install-time concept,
not something that changes mid-session; a settings-style live-reload
isn't worth the complexity here.
"""
import importlib.util
import json
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ui.console import console

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLUGINS_DIR_NAME = "plugins"

_manifests_cache: list[dict] | None = None
_global_commands_cache: dict[str, dict] | None = None
_global_hooks_cache: dict[str, list[Callable]] = {}


# ---------------------------------------------------------------------------
# SkillTask — a skill's `run()` returns one of these (or a plain str) to hand
# its task off to a real agent turn instead of just printing and stopping.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillTask:
    """Richer alternative to returning a plain `str` from a skill's `run()`.
    A bare `str` return is exactly `SkillTask(task=that_string)` — these
    two extra fields are opt-in.

    allowed_tools — restricts the tool set for THIS turn only, to the given
    tool NAMES (e.g. {"read_file", "grep_search", "glob_search"} for a
    read-only investigation skill). Enforced by SkillToolRestrictionMiddleware
    below (a disallowed tool call is mechanically rejected, not just asked
    politely not to happen) — None means no restriction, same as today.
    This does NOT shrink what the model's schema shows it (that would touch
    agent_builder.py's cached per-(role, tool_names) agent construction,
    which lives well outside a single turn) — the model still SEES every
    tool it normally would, it just gets told no if it calls one outside
    this set.

    prefer_delegate — soft nudge (prepended text), not an enforced routing
    rule: flowai has no way to hand a skill's task straight to delegate()
    without going through the model's own tool-calling decision (delegate
    only exists in the main agent's tool set to begin with, see
    delegate_tool.py — nothing to route to at all in pipeline_mode). Same
    class of mechanism as delegate_tool.py's own _DelegateNudgeMiddleware
    (prompt text, not a hard bypass) — harmless if delegate isn't bound for
    this session, the sentence just goes unused."""

    task: str
    allowed_tools: frozenset[str] | None = None
    prefer_delegate: bool = False

    def render(self) -> str:
        if not self.prefer_delegate:
            return self.task
        return (
            "(This looks like a broad, read-only investigation spanning "
            "many files — if delegate() is available, prefer calling it "
            "for this right away instead of investigating step by step "
            "yourself.)\n\n" + self.task
        )


# Set once per turn (cli.py, right before handing a skill's task into the
# normal message pipeline), read by SkillToolRestrictionMiddleware below on
# every tool call during that same turn. None (the default) means no
# restriction — reset to None at the top of EVERY _handle_input call in
# cli.py, not just cleared after a restricted one, so a stray leftover
# restriction can survive at most one turn even in a bug, never longer.
current_skill_restriction: ContextVar[frozenset[str] | None] = ContextVar(
    "skill_tool_restriction", default=None
)


class SkillToolRestrictionMiddleware(AgentMiddleware):
    """Unconditional in the base middleware stack (agent_builder.py:
    _base_agent_middleware), like PluginHookMiddleware — current_skill_restriction
    being unset (the overwhelming common case: no skill in play, or a skill
    that didn't ask for a restriction) makes every check here a no-op.
    Placed before HumanInTheLoopMiddleware in the list (see
    _base_agent_middleware's own docstring on why pre_hitl rejectors go
    first) — a tool call that's going to be mechanically refused must never
    make the user sit through approving it first."""

    async def awrap_tool_call(self, request, handler):
        allowed = current_skill_restriction.get()
        name = request.tool_call["name"]
        if allowed is None or name in allowed:
            return await handler(request)
        return ToolMessage(
            content=(
                f"Tool '{name}' is not allowed for this task — it was "
                "started by a skill restricted to: " + ", ".join(sorted(allowed)) +
                ". Work within that set; if you genuinely need something "
                "outside it, say so in your final answer instead of calling it."
            ),
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def plugins_dir() -> Path:
    path = _REPO_ROOT / _PLUGINS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(base_dir: Path, rel_or_name: str) -> str:
    candidate = base_dir / rel_or_name
    return str(candidate) if candidate.is_file() else rel_or_name


def _resolve_command(command: str) -> str:
    """"python3"/"python" -> sys.executable — a bare "python3" resolves
    through PATH to whatever system interpreter happens to be first there,
    almost never the one flowai itself runs under (and the only one with
    the `mcp` package, or anything else in requirements.txt, actually
    installed). mcp_agent/config.py's own built-in servers avoid this by
    using sys.executable directly; plugin authors writing a manifest by
    hand have no reason to know that trap exists, so it's handled here
    instead of documented as a gotcha to remember."""
    if command in ("python3", "python"):
        return sys.executable
    return command


def _import_from_file(namespace: str, file_path: Path):
    """Each module gets its own sys.modules entry namespaced by caller
    (flowai_plugin.<namespace>.<stem>) — two plugins, or a plugin and a
    project's own .flowai/ files, are free to both ship a file called
    hooks.py without colliding in the shared import cache."""
    module_name = f"flowai_plugin.{namespace}.{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Global plugins — <repo_root>/plugins/<name>/plugin.json
# ---------------------------------------------------------------------------


def discover_plugins() -> list[dict]:
    """One dict per enabled plugin, each carrying the parsed manifest plus
    "_dir" (the plugin's own directory, for resolving relative paths).
    Skips (with a console warning, not an exception) any directory whose
    plugin.json is missing/malformed, whose declared "name" doesn't match
    its own directory name, or that contains a ".disabled" marker file."""
    global _manifests_cache
    if _manifests_cache is not None:
        return _manifests_cache

    manifests = []
    for entry in sorted(plugins_dir().iterdir()):
        if not entry.is_dir() or (entry / ".disabled").exists():
            continue
        manifest_path = entry / "plugin.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[yellow]⚠ Плагин '{entry.name}': не удалось прочитать plugin.json ({e})[/]")
            continue
        if manifest.get("name") != entry.name:
            console.print(
                f"[yellow]⚠ Плагин '{entry.name}': поле \"name\" в plugin.json "
                f"({manifest.get('name')!r}) не совпадает с именем папки — пропущен[/]"
            )
            continue
        manifest["_dir"] = entry
        manifests.append(manifest)

    _manifests_cache = manifests
    return manifests


def _load_global_commands() -> dict[str, dict]:
    global _global_commands_cache
    if _global_commands_cache is not None:
        return _global_commands_cache

    commands: dict[str, dict] = {}
    for manifest in discover_plugins():
        plugin_dir: Path = manifest["_dir"]
        for name, spec in (manifest.get("commands") or {}).items():
            try:
                module_path = plugin_dir / spec["module"]
                module = _import_from_file(manifest["name"], module_path)
                func = getattr(module, spec["function"])
            except Exception as e:
                console.print(f"[yellow]⚠ Плагин '{manifest['name']}': команда /{name} не загрузилась ({e})[/]")
                continue
            if name in commands:
                console.print(
                    f"[yellow]⚠ Плагин '{manifest['name']}': команда /{name} уже "
                    f"объявлена плагином '{commands[name]['plugin']}' — вторая копия пропущена[/]"
                )
                continue
            commands[name] = {"func": func, "help": spec.get("help", ""), "plugin": manifest["name"]}

    _global_commands_cache = commands
    return commands


def load_mcp_servers() -> dict[str, tuple[str, list[str]]]:
    """{server_name: (command, args)} — same shape build_mcp_connections()
    (mcp_agent/config.py) already uses for its own built-in servers, so
    it can just update() this in. Global plugins only — no per-project
    MCP servers (.flowai/ only covers skills/hooks, see module docstring).
    Server names collide the same way command names do — first plugin
    wins, rest are skipped with a warning (a silently dropped MCP server
    is a much quieter failure than an exception mid-turn, so it's caught
    here rather than left to whatever error langchain_mcp_adapters would
    raise on a duplicate key)."""
    servers: dict[str, tuple[str, list[str]]] = {}
    owner: dict[str, str] = {}
    for manifest in discover_plugins():
        plugin_dir: Path = manifest["_dir"]
        for name, spec in (manifest.get("mcp_servers") or {}).items():
            if name in servers:
                console.print(
                    f"[yellow]⚠ Плагин '{manifest['name']}': MCP-сервер '{name}' уже "
                    f"объявлен плагином '{owner[name]}' — вторая копия пропущена[/]"
                )
                continue
            command = _resolve_command(spec["command"])
            args = [_resolve(plugin_dir, a) for a in spec.get("args", [])]
            servers[name] = (command, args)
            owner[name] = manifest["name"]
    return servers


def _load_global_hooks(hook_name: str) -> list[Callable]:
    if hook_name in _global_hooks_cache:
        return _global_hooks_cache[hook_name]

    funcs: list[Callable] = []
    for manifest in discover_plugins():
        plugin_dir: Path = manifest["_dir"]
        for entry in (manifest.get("hooks") or {}).get(hook_name, []):
            try:
                module_file, func_name = entry.split(":", 1)
                module = _import_from_file(manifest["name"], plugin_dir / module_file)
                funcs.append(getattr(module, func_name))
            except Exception as e:
                console.print(f"[yellow]⚠ Плагин '{manifest['name']}': хук {hook_name} ({entry}) не загрузился ({e})[/]")

    _global_hooks_cache[hook_name] = funcs
    return funcs


def invalidate_cache() -> None:
    """No automatic file-watching — plugins are meant to be dropped in
    before flowai starts, not hot-reloaded mid-session. Exists for /plugin
    reload and for tests, not called anywhere during normal operation.
    Only global-plugin caches — per-project skills/hooks are never cached
    in the first place (see module docstring)."""
    global _manifests_cache, _global_commands_cache
    _manifests_cache = None
    _global_commands_cache = None
    _global_hooks_cache.clear()


# ---------------------------------------------------------------------------
# Per-project skills/hooks — <repo_path>/.flowai/{skills,hooks}/*.py
# ---------------------------------------------------------------------------


def _project_namespace(repo_path: str) -> str:
    """A stable, filesystem-path-safe namespace for _import_from_file —
    doesn't need to be reversible, only distinct per repo_path so two
    projects' same-named skill/hook files never collide in sys.modules."""
    return "project." + str(abs(hash(str(Path(repo_path).resolve()))))


def discover_project_skills(repo_path: str) -> dict[str, dict]:
    """{command_name: {"func": callable, "help": "", "plugin": "..."}}
    from <repo_path>/.flowai/skills/*.py — the filename (minus ".py") IS
    the command name; each file must define a module-level `run(args,
    console)` (see module docstring for the signature). No manifest, no
    caching — see module docstring for why."""
    skills_dir = Path(repo_path) / ".flowai" / "skills"
    result: dict[str, dict] = {}
    if not skills_dir.is_dir():
        return result
    namespace = _project_namespace(repo_path)
    for file in sorted(skills_dir.glob("*.py")):
        try:
            module = _import_from_file(namespace, file)
            func = module.run
        except Exception as e:
            console.print(f"[yellow]⚠ .flowai/skills/{file.name}: не загрузился ({e})[/]")
            continue
        result[file.stem] = {"func": func, "help": "", "plugin": f".flowai/skills/{file.name}"}
    return result


def discover_project_hooks(repo_path: str, hook_name: str) -> list[Callable]:
    """Every <repo_path>/.flowai/hooks/*.py file that defines a
    module-level function named `hook_name` (post_file_edit/pre_commit,
    see mcp_agent/plugin_hooks.py) contributes one hook — a single file
    is free to define both, or just the one it needs. No caching — see
    module docstring for why."""
    hooks_dir = Path(repo_path) / ".flowai" / "hooks"
    result: list[Callable] = []
    if not hooks_dir.is_dir():
        return result
    namespace = _project_namespace(repo_path)
    for file in sorted(hooks_dir.glob("*.py")):
        try:
            module = _import_from_file(namespace, file)
        except Exception as e:
            console.print(f"[yellow]⚠ .flowai/hooks/{file.name}: не загрузился ({e})[/]")
            continue
        func = getattr(module, hook_name, None)
        if func is not None:
            result.append(func)
    return result


# ---------------------------------------------------------------------------
# Combined public API — what cli.py / plugin_hooks.py actually call
# ---------------------------------------------------------------------------


def load_commands(repo_path: str | None = None) -> dict[str, dict]:
    """Project skills (more specific) checked first, global plugins fill
    in the rest — a project skill can shadow a global plugin's
    same-named command, neither can ever shadow a cli.py built-in (that
    check happens entirely in cli.py, before this is even consulted)."""
    commands = dict(_load_global_commands())
    if repo_path is not None:
        commands.update(discover_project_skills(repo_path))
    return commands


def load_hooks(hook_name: str, repo_path: str | None = None) -> list[Callable]:
    """Global plugin hooks, then project hooks — no "one winner" here
    (see _load_global_hooks' docstring), every hook of this kind runs."""
    hooks = list(_load_global_hooks(hook_name))
    if repo_path is not None:
        hooks.extend(discover_project_hooks(repo_path, hook_name))
    return hooks


def describe_installed(repo_path: str | None = None) -> str:
    """Human-readable summary for /plugin — what's installed (global
    plugins) and what's declared for the current project (.flowai/), not
    a raw manifest dump."""
    manifests = discover_plugins()
    lines = []
    if not manifests:
        lines.append(f"Глобальных плагинов не найдено. Положи папку с plugin.json в {plugins_dir()}.")
    for m in manifests:
        provides = []
        if m.get("commands"):
            provides.append("команды: " + ", ".join(f"/{n}" for n in m["commands"]))
        if m.get("mcp_servers"):
            provides.append("MCP-серверы: " + ", ".join(m["mcp_servers"]))
        if m.get("hooks"):
            provides.append("хуки: " + ", ".join(m["hooks"]))
        lines.append(f"[bold]{m['name']}[/] v{m.get('version', '?')} — {m.get('description', '')}\n  " + ("; ".join(provides) or "ничего не объявлено"))

    if repo_path is not None:
        skills = discover_project_skills(repo_path)
        hook_names = [h for h in ("post_file_edit", "pre_commit") if discover_project_hooks(repo_path, h)]
        if skills or hook_names:
            lines.append("")
            lines.append(f"[bold]Этот проект[/] ({repo_path}/.flowai/):")
            if skills:
                lines.append("  скилы: " + ", ".join(f"/{n}" for n in skills))
            if hook_names:
                lines.append("  хуки: " + ", ".join(hook_names))

    return "\n".join(lines)
