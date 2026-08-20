"""grep_search's plain-`grep` fallback (mcp_agent/servers/file_ops_server.py,
used when ripgrep isn't installed) — live-caught bug: a pattern using
alternation (`A|B`) silently matched nothing (GNU grep's BRE default
treats `|` as a literal character, not ripgrep's native alternation), and
a "**/*.ext"-style glob silently matched nothing too (--include does a
plain basename fnmatch, no concept of ripgrep's recursive "**"). Both
returned an honest-looking "No matches for ..." instead of erroring,
indistinguishable from a real empty result."""
import pytest

import mcp_agent.servers.file_ops_server as file_ops_server
from mcp_agent.servers.file_ops_server import grep_search


@pytest.fixture(autouse=True)
def _force_plain_grep_fallback(monkeypatch):
    monkeypatch.setattr(file_ops_server, "_HAS_RG", False)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "UserController.php").write_text("<?php\nclass UserController {}\n")
    (tmp_path / "app" / "not_php.txt").write_text("Controller\n")
    return tmp_path


@pytest.mark.asyncio
async def test_alternation_pattern_matches(project):
    result = await grep_search(pattern="Controller|controller", path=str(project))
    assert "UserController.php" in result


@pytest.mark.asyncio
async def test_recursive_glob_prefix_matches(project):
    result = await grep_search(pattern="Controller", path=str(project), glob="**/*.php")
    assert "UserController.php" in result
    assert "not_php.txt" not in result
