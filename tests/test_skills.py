"""mcp_agent/skills.py — SKILL.md discovery, frontmatter parsing, the
`skill` tool and `/name` expansion."""
from mcp_agent import skills


def _write(root, name, text):
    d = root / ".flowai" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_discovery_and_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    _write(tmp_path, "deploy", "---\nname: deploy\ndescription: Ship it.\nallowed-tools: bash, read_file\n---\nRun make deploy.\n")
    _write(tmp_path, "nofront", "Just do the thing.\n")
    found = skills.discover_skills(str(tmp_path))
    assert found["deploy"].description == "Ship it."
    assert found["deploy"].allowed_tools == frozenset({"bash", "read_file"})
    assert found["deploy"].body == "Run make deploy."
    assert found["nofront"].allowed_tools is None


def test_flat_md_file_is_a_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    d = tmp_path / "home" / ".flowai" / "skills"
    d.mkdir(parents=True)
    (d / "review.md").write_text("---\ndescription: flat\n---\nDo the review.\n")
    found = skills.discover_skills(str(tmp_path / "proj"))
    assert found["review"].description == "flat" and found["review"].source == "user"


def test_project_skill_shadows_user_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    user_dir = tmp_path / "home" / ".flowai" / "skills" / "x"
    user_dir.mkdir(parents=True)
    (user_dir / "SKILL.md").write_text("---\ndescription: user\n---\nuser body\n")
    _write(tmp_path, "x", "---\ndescription: project\n---\nproject body\n")
    assert skills.discover_skills(str(tmp_path / "other"))["x"].source == "user"
    assert skills.discover_skills(str(tmp_path))["x"].source == "project"


def test_arguments_placeholder_and_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    _write(tmp_path, "notes", "---\ndescription: d\n---\nFocus on $ARGUMENTS.\n")
    out = skills.build_skill_tool(str(tmp_path)).invoke({"name": "notes", "arguments": "the API"})
    assert "Focus on the API." in out
    assert "Error" in skills.build_skill_tool(str(tmp_path)).invoke({"name": "missing"})


def test_expand_slash_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    _write(tmp_path, "notes", "---\ndescription: d\nallowed-tools: [read_file]\n---\nBody.\n")
    text, allowed = skills.expand_slash_command("/notes extra words", str(tmp_path))
    assert "Body." in text and "Arguments: extra words" in text
    assert allowed == frozenset({"read_file"})
    assert skills.expand_slash_command("/unknown", str(tmp_path)) is None
    assert skills.expand_slash_command("plain text", str(tmp_path)) is None
