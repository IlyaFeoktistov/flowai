"""Analyzer/Planner's bash allowlist (mcp_agent/agent_builder.py:
_is_read_only_bash_command) — the "run to reproduce" tier added on top of
the original read-only-diagnostics allowlist, so a debugging investigation
can actually run the program under investigation, without reopening the
original incident this allowlist exists to prevent (installs/builds/
writes instead of reporting)."""
import pytest

from mcp_agent.agent_builder import _is_read_only_bash_command


@pytest.mark.parametrize("command", [
    "go run platformer.go",
    "go run platformer.go --level 2",
    "python3 script.py",
    "python3 script.py --dry-run",
    "node app.js",
    "ruby script.rb",
    "php index.php",
    "npm test",
    "npm test -- --watch=false",
    "pytest",
    "pytest -k foo",
    "pytest tests/test_x.py",
    "go test",
    "go test ./...",
    "go test -run TestFoo -v",
    "make test",
    "cargo test",
    "cargo test test_name",
])
def test_execute_to_reproduce_allowed(command):
    assert _is_read_only_bash_command(command) is True


@pytest.mark.parametrize("command", [
    "go run -exec sudo platformer.go",   # interpreter-position flag
    "go install",
    "go build",
    "python3 -m pip install requests",   # module-runner escape hatch
    "python3 -c \"import os\"",
    "node -e \"1\"",
    "npm install",
    "npm run build",
    "make",                              # bare make — could build/install
    "make install",
    "cargo build",
    "cargo install foo",
    "yarn install",
])
def test_mutating_or_install_commands_still_denied(command):
    assert _is_read_only_bash_command(command) is False


@pytest.mark.parametrize("command", [
    "git status",
    "git log",
    "cat foo.txt",
    "grep -r foo .",
])
def test_original_read_only_diagnostics_still_allowed(command):
    assert _is_read_only_bash_command(command) is True


@pytest.mark.parametrize("command", [
    "rm -rf /",
    "git commit -m x",
    "apt-get install -y foo",
    "cat foo; rm -rf bar",
    "go run x.go | sh",
])
def test_original_mutation_paths_still_denied(command):
    assert _is_read_only_bash_command(command) is False
