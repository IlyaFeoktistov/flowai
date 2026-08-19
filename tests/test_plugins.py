"""mcp_agent/plugins.py — two independent layers:

- Global plugins: <repo_root>/plugins/<name>/plugin.json, isolated here
  by monkeypatching plugins._REPO_ROOT to a tmp_path (plugins_dir()
  derives from it) rather than the real flowAI checkout.
- Per-project skills/hooks: <repo_path>/.flowai/{skills,hooks}/*.py, no
  manifest — isolated by using a separate tmp_path as "the project" and
  passing it explicitly as repo_path, never touching the real cwd.
"""
import json
import sys

import pytest

from mcp_agent import plugins


@pytest.fixture(autouse=True)
def isolated_plugins_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins, "_REPO_ROOT", tmp_path)
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


# ---------------------------------------------------------------------------
# Per-project skills/hooks — <repo_path>/.flowai/{skills,hooks}/*.py
# ---------------------------------------------------------------------------


def test_discovers_a_project_skill_by_filename(tmp_path):
    project = tmp_path / "some-project"
    skills_dir = project / ".flowai" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "greet.py").write_text("def run(args, console):\n    return f'hi {args}'\n")

    skills = plugins.discover_project_skills(str(project))
    assert set(skills) == {"greet"}
    assert skills["greet"]["func"]("world", None) == "hi world"


def test_project_with_no_flowai_dir_has_no_skills(tmp_path):
    assert plugins.discover_project_skills(str(tmp_path)) == {}


def test_broken_project_skill_does_not_raise(tmp_path):
    skills_dir = tmp_path / ".flowai" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "broken.py").write_text("this is not valid python(((\n")
    assert plugins.discover_project_skills(str(tmp_path)) == {}


def test_discovers_project_hooks_by_well_known_function_name(tmp_path):
    hooks_dir = tmp_path / ".flowai" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "checks.py").write_text(
        "def post_file_edit(path, repo_path):\n    return 'edited'\n"
        "def pre_commit(command, repo_path):\n    return None\n"
    )
    edit_hooks = plugins.discover_project_hooks(str(tmp_path), "post_file_edit")
    commit_hooks = plugins.discover_project_hooks(str(tmp_path), "pre_commit")
    assert len(edit_hooks) == 1 and edit_hooks[0]("f.py", str(tmp_path)) == "edited"
    assert len(commit_hooks) == 1 and commit_hooks[0]("git commit", str(tmp_path)) is None


def test_project_hook_file_without_the_requested_function_contributes_nothing(tmp_path):
    hooks_dir = tmp_path / ".flowai" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "checks.py").write_text("def pre_commit(command, repo_path):\n    return None\n")
    assert plugins.discover_project_hooks(str(tmp_path), "post_file_edit") == []


def test_two_projects_with_same_named_skill_files_do_not_collide(tmp_path):
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    for project, word in ((project_a, "a"), (project_b, "b")):
        skills_dir = project / ".flowai" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "greet.py").write_text(f"def run(args, console):\n    return '{word}'\n")

    skills_a = plugins.discover_project_skills(str(project_a))
    skills_b = plugins.discover_project_skills(str(project_b))
    assert skills_a["greet"]["func"](None, None) == "a"
    assert skills_b["greet"]["func"](None, None) == "b"


def test_load_commands_merges_global_plugins_and_project_skills(isolated_plugins_dir, tmp_path):
    _write_plugin(
        isolated_plugins_dir, "global-plugin",
        {"name": "global-plugin", "version": "1.0", "commands": {"fromglobal": {"module": "c.py", "function": "run"}}},
        files={"c.py": "def run(args, console):\n    return 'global'\n"},
    )
    project = tmp_path / "project"
    skills_dir = project / ".flowai" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "fromproject.py").write_text("def run(args, console):\n    return 'project'\n")

    commands = plugins.load_commands(str(project))
    assert set(commands) == {"fromglobal", "fromproject"}


def test_project_skill_shadows_a_global_plugin_command_of_the_same_name(isolated_plugins_dir, tmp_path):
    _write_plugin(
        isolated_plugins_dir, "global-plugin",
        {"name": "global-plugin", "version": "1.0", "commands": {"hello": {"module": "c.py", "function": "run"}}},
        files={"c.py": "def run(args, console):\n    return 'global'\n"},
    )
    project = tmp_path / "project"
    skills_dir = project / ".flowai" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "hello.py").write_text("def run(args, console):\n    return 'project'\n")

    commands = plugins.load_commands(str(project))
    assert commands["hello"]["func"](None, None) == "project"


def test_load_hooks_merges_global_plugins_and_project_hooks(isolated_plugins_dir, tmp_path):
    _write_plugin(
        isolated_plugins_dir, "global-plugin",
        {"name": "global-plugin", "version": "1.0", "hooks": {"post_file_edit": ["h.py:on_edit"]}},
        files={"h.py": "def on_edit(path, repo_path):\n    return 'global'\n"},
    )
    project = tmp_path / "project"
    hooks_dir = project / ".flowai" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "h.py").write_text("def post_file_edit(path, repo_path):\n    return 'project'\n")

    results = [h("f.py", str(project)) for h in plugins.load_hooks("post_file_edit", str(project))]
    assert results == ["global", "project"]


def test_load_commands_without_repo_path_is_global_only(isolated_plugins_dir):
    _write_plugin(
        isolated_plugins_dir, "global-plugin",
        {"name": "global-plugin", "version": "1.0", "commands": {"fromglobal": {"module": "c.py", "function": "run"}}},
        files={"c.py": "def run(args, console):\n    return 'global'\n"},
    )
    assert set(plugins.load_commands()) == {"fromglobal"}


def test_describe_installed_mentions_project_skills_and_hooks(tmp_path):
    project = tmp_path / "project"
    skills_dir = project / ".flowai" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "greet.py").write_text("def run(args, console):\n    pass\n")

    report = plugins.describe_installed(str(project))
    assert "/greet" in report
