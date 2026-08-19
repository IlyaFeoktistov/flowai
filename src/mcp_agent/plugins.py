"""
Plugin loader — discovers user-installed plugins under
storage.data_dir()/plugins/<name>/plugin.json and exposes three things a
plugin can declare: slash commands (cli.py), MCP servers (mcp_agent/
config.py:build_mcp_connections), and hooks (mcp_agent/plugin_hooks.py).

A plugin is just a directory with a plugin.json manifest — no install
step, no registry, no build: drop a folder in, it's live on next launch;
add a ".disabled" file inside it to turn it off without deleting it.

Manifest shape:
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

Command function signature — `def run(args: str, console) -> None` (sync
or async, cli.py awaits it only if it returns an awaitable): `args` is the
raw text after "/hello ", `console` is ui/console.py's Rich console (the
same one every other command prints through — printing anywhere else
won't render inside the TUI, see ui/console.py's own docstring). A plugin
command can never SHADOW a built-in one — cli.py only checks plugin
commands after every "/xxx" it already knows about.

Hook function signatures — mcp_agent/plugin_hooks.py calls these, see its
own docstring for exactly when and with what arguments.

Every load function is best-effort per plugin: one broken manifest, one
plugin whose module fails to import, must not take down flowai startup or
stop every OTHER plugin from loading — each failure is caught, reported
via console, and skipped.

Discovery result is cached for the process lifetime (like agent_builder.py's
other caches) — plugins are a install-time concept, not something that
changes mid-session; a settings-style live-reload isn't worth the
complexity here.
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import storage
from ui.console import console

_PLUGINS_DIR_NAME = "plugins"

_manifests_cache: list[dict] | None = None
_commands_cache: dict[str, dict] | None = None
_hooks_cache: dict[str, list[Callable]] = {}


def plugins_dir() -> Path:
    path = storage.data_dir() / _PLUGINS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(plugin_dir: Path, rel_or_name: str) -> str:
    candidate = plugin_dir / rel_or_name
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


def _import_from_file(plugin_name: str, file_path: Path):
    """Each plugin module gets its own sys.modules entry namespaced by
    plugin name (flowai_plugin.<name>.<stem>) — two plugins are free to
    both ship a file called hooks.py without colliding in the shared
    import cache."""
    module_name = f"flowai_plugin.{plugin_name}.{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_commands() -> dict[str, dict]:
    """{command_name (no leading "/"): {"func": callable, "help": str,
    "plugin": plugin_name}} — cli.py checks this AFTER every built-in
    command, so a plugin can't shadow one of flowai's own."""
    global _commands_cache
    if _commands_cache is not None:
        return _commands_cache

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

    _commands_cache = commands
    return commands


def load_mcp_servers() -> dict[str, tuple[str, list[str]]]:
    """{server_name: (command, args)} — same shape build_mcp_connections()
    (mcp_agent/config.py) already uses for its own built-in servers, so
    it can just update() this in. Server names collide the same way
    command names do — first plugin wins, rest are skipped with a warning
    (a silently dropped MCP server is a much quieter failure than an
    exception mid-turn, so it's caught here rather than left to whatever
    error langchain_mcp_adapters would raise on a duplicate key)."""
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


def load_hooks(hook_name: str) -> list[Callable]:
    """Every plugin's hooks[hook_name] entries ("module.py:function"),
    concatenated across plugins in discovery order — unlike commands/MCP
    servers, hooks have no name to collide on and no reason only one
    plugin's hook of a given kind should run; mcp_agent/plugin_hooks.py
    calls every one of them."""
    if hook_name in _hooks_cache:
        return _hooks_cache[hook_name]

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

    _hooks_cache[hook_name] = funcs
    return funcs


def invalidate_cache() -> None:
    """No automatic file-watching — plugins are meant to be dropped in
    before flowai starts, not hot-reloaded mid-session. Exists for /plugin
    reload and for tests, not called anywhere during normal operation."""
    global _manifests_cache, _commands_cache
    _manifests_cache = None
    _commands_cache = None
    _hooks_cache.clear()


def describe_installed() -> str:
    """Human-readable summary for /plugin — what's installed and what
    each one provides, not a raw manifest dump."""
    manifests = discover_plugins()
    if not manifests:
        return f"Плагинов не найдено. Положи папку с plugin.json в {plugins_dir()}."
    lines = []
    for m in manifests:
        provides = []
        if m.get("commands"):
            provides.append("команды: " + ", ".join(f"/{n}" for n in m["commands"]))
        if m.get("mcp_servers"):
            provides.append("MCP-серверы: " + ", ".join(m["mcp_servers"]))
        if m.get("hooks"):
            provides.append("хуки: " + ", ".join(m["hooks"]))
        lines.append(f"[bold]{m['name']}[/] v{m.get('version', '?')} — {m.get('description', '')}\n  " + ("; ".join(provides) or "ничего не объявлено"))
    return "\n".join(lines)
