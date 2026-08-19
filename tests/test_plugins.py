"""mcp_agent/plugins.py — a plugin is a directory under
storage.data_dir()/plugins/<name>/ with a plugin.json manifest declaring
some mix of slash commands, MCP servers, and hooks. Isolated from the
real ~/.local/share/flowai/ by monkeypatching storage.data_dir(), same
pattern as tests/test_clean.py."""
import json
import sys

import pytest

import storage
from mcp_agent import plugins


@pytest.fixture(autouse=True)
def isolated_plugins_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path)
    plugins.invalidate_cache()
    yield tmp_path
    plugins.invalidate_cache()


def _write_plugin(base, name, manifest, files=None):
    plugin_dir = base / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    for filename, content in (files or {}).items():
        (plugin_dir / filename).write_text(content)
    return plugin_dir


def test_discovers_a_well_formed_plugin(isolated_plugins_dir):
    _write_plugin(isolated_plugins_dir, "hello", {"name": "hello", "version": "0.1.0"})
    manifests = plugins.discover_plugins()
    assert [m["name"] for m in manifests] == ["hello"]


def test_skips_directory_without_a_manifest(isolated_plugins_dir):
    (isolated_plugins_dir / "plugins" / "empty").mkdir(parents=True)
    assert plugins.discover_plugins() == []


def test_skips_malformed_json_without_raising(isolated_plugins_dir):
    plugin_dir = isolated_plugins_dir / "plugins" / "broken"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text("{not valid json")
    assert plugins.discover_plugins() == []


def test_skips_name_directory_mismatch(isolated_plugins_dir):
    _write_plugin(isolated_plugins_dir, "actual-dir-name", {"name": "different-name", "version": "1.0"})
    assert plugins.discover_plugins() == []


def test_disabled_marker_file_hides_the_plugin(isolated_plugins_dir):
    plugin_dir = _write_plugin(isolated_plugins_dir, "hello", {"name": "hello", "version": "0.1.0"})
    (plugin_dir / ".disabled").touch()
    assert plugins.discover_plugins() == []


def test_loads_a_command_and_calls_it(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "hello",
        {"name": "hello", "version": "0.1.0", "commands": {"hello": {"module": "cmd.py", "function": "run", "help": "say hi"}}},
        files={"cmd.py": "def run(args, console):\n    return f'hi {args}'\n"},
    )
    commands = plugins.load_commands()
    assert set(commands) == {"hello"}
    assert commands["hello"]["help"] == "say hi"
    assert commands["hello"]["func"]("world", None) == "hi world"


def test_two_plugins_declaring_the_same_command_first_one_wins(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "a-plugin",
        {"name": "a-plugin", "version": "1.0", "commands": {"hello": {"module": "cmd.py", "function": "run"}}},
        files={"cmd.py": "def run(args, console):\n    return 'from a'\n"},
    )
    _write_plugin(
        isolated_plugins_dir, "b-plugin",
        {"name": "b-plugin", "version": "1.0", "commands": {"hello": {"module": "cmd.py", "function": "run"}}},
        files={"cmd.py": "def run(args, console):\n    return 'from b'\n"},
    )
    commands = plugins.load_commands()
    assert len(commands) == 1
    assert commands["hello"]["plugin"] == "a-plugin"


def test_broken_command_module_does_not_crash_discovery(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "broken-cmd",
        {"name": "broken-cmd", "version": "1.0", "commands": {"oops": {"module": "cmd.py", "function": "run"}}},
        files={"cmd.py": "this is not valid python(((\n"},
    )
    assert plugins.load_commands() == {}


def test_loads_mcp_server_and_resolves_local_script_path(isolated_plugins_dir):
    plugin_dir = _write_plugin(
        isolated_plugins_dir, "my-server",
        {"name": "my-server", "version": "1.0", "mcp_servers": {"myserver": {"command": "python3", "args": ["server.py", "--flag"]}}},
        files={"server.py": "# not actually run in this test\n"},
    )
    servers = plugins.load_mcp_servers()
    assert set(servers) == {"myserver"}
    command, args = servers["myserver"]
    assert command == sys.executable  # "python3" resolves to flowai's OWN interpreter, not whatever's on PATH
    assert args[0] == str(plugin_dir / "server.py")  # resolved to absolute
    assert args[1] == "--flag"  # not a local file, passed through


def test_mcp_server_command_other_than_python_is_passed_through_unchanged(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "npm-server",
        {"name": "npm-server", "version": "1.0", "mcp_servers": {"myserver": {"command": "some-installed-cli", "args": []}}},
    )
    command, _args = plugins.load_mcp_servers()["myserver"]
    assert command == "some-installed-cli"


def test_two_plugins_declaring_the_same_mcp_server_first_one_wins(isolated_plugins_dir):
    _write_plugin(isolated_plugins_dir, "a-plugin", {"name": "a-plugin", "version": "1.0", "mcp_servers": {"srv": {"command": "a", "args": []}}})
    _write_plugin(isolated_plugins_dir, "b-plugin", {"name": "b-plugin", "version": "1.0", "mcp_servers": {"srv": {"command": "b", "args": []}}})
    servers = plugins.load_mcp_servers()
    assert servers["srv"][0] == "a"


def test_loads_hooks_from_multiple_plugins_concatenated(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "plugin-a",
        {"name": "plugin-a", "version": "1.0", "hooks": {"post_file_edit": ["hooks.py:on_edit"]}},
        files={"hooks.py": "def on_edit(path, repo_path):\n    return 'a'\n"},
    )
    _write_plugin(
        isolated_plugins_dir, "plugin-b",
        {"name": "plugin-b", "version": "1.0", "hooks": {"post_file_edit": ["hooks.py:on_edit"]}},
        files={"hooks.py": "def on_edit(path, repo_path):\n    return 'b'\n"},
    )
    hooks = plugins.load_hooks("post_file_edit")
    assert [h("x", "/repo") for h in hooks] == ["a", "b"]


def test_no_hooks_declared_returns_empty_list(isolated_plugins_dir):
    _write_plugin(isolated_plugins_dir, "no-hooks", {"name": "no-hooks", "version": "1.0"})
    assert plugins.load_hooks("pre_commit") == []


def test_describe_installed_lists_what_each_plugin_provides(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "full",
        {
            "name": "full", "version": "1.0", "description": "does things",
            "commands": {"hello": {"module": "c.py", "function": "run"}},
            "mcp_servers": {"srv": {"command": "x", "args": []}},
            "hooks": {"pre_commit": ["h.py:f"]},
        },
    )
    report = plugins.describe_installed()
    assert "full" in report
    assert "/hello" in report
    assert "srv" in report
    assert "pre_commit" in report


def test_describe_installed_with_no_plugins():
    report = plugins.describe_installed()
    assert "не найдено" in report
