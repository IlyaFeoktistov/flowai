"""
Markdown skills in the Claude Code format: a directory with a SKILL.md
whose YAML frontmatter names and describes it, and whose body is the
instructions the model follows when the skill is used.

    <dir>/<name>/SKILL.md
    ---
    name: release-notes            (optional, defaults to the directory name)
    description: Draft release notes from the git log since the last tag.
    allowed-tools: git_log, read_file   (optional: comma list or YAML list)
    ---
    ...instructions...

Two ways a skill gets used:
- The model picks it: the system prompt lists every skill's name +
  description (skills_prompt_block), and the `skill(name, arguments)` tool
  (build_skill_tool) returns the full body on demand — so only the
  one-line descriptions cost context until a skill is actually needed.
- The user invokes it: `/<name> args` (cli.py via plugins.load_commands,
  main.py via expand_slash_command) puts the body straight into the turn.

Search order, first match per name wins (most specific first):
  1. <open project>/.flowai/skills/<name>/SKILL.md
  2. ~/.flowai/skills/<name>/SKILL.md          (user-wide, every project)
  3. ~/.config/flowai/skills/<name>/SKILL.md   (same, XDG location)
  4. <flowai checkout>/plugins/<plugin>/skills/<name>/SKILL.md

Files next to SKILL.md (scripts, templates, reference docs) are the
skill's own resources — the body refers to them by relative path, and the
tool result tells the model the skill's base directory to resolve them.

Rescanned on every call (a few small directories) — skills are edited
while flowai runs, and a stale cache would silently hide a new one.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from langchain_core.tools import tool

_FLOWAI_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str
    allowed_tools: frozenset[str] | None
    source: str  # "project" | "user" | "plugin:<name>" — for /plugin

    @property
    def base_dir(self) -> Path:
        return self.path.parent


def user_skills_dir() -> Path:
    """Primary user-wide location, in the home folder like ~/.claude/skills."""
    return Path(os.path.expanduser("~")) / ".flowai" / "skills"


def _xdg_user_skills_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "flowai" / "skills"


def _search_roots(repo_path: str | None) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    if repo_path:
        roots.append((Path(repo_path) / ".flowai" / "skills", "project"))
    roots.append((user_skills_dir(), "user"))
    roots.append((_xdg_user_skills_dir(), "user"))
    plugins_root = _FLOWAI_ROOT / "plugins"
    if plugins_root.is_dir():
        for plugin_dir in sorted(p for p in plugins_root.iterdir() if p.is_dir()):
            roots.append((plugin_dir / "skills", f"plugin:{plugin_dir.name}"))
    return roots


def _parse(path: Path, source: str) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, _, rest = text.partition("\n")
        front, sep, after = rest.partition("\n---")
        if sep:
            try:
                meta = yaml.safe_load(front) or {}
            except yaml.YAMLError:
                meta = {}
            body = after.partition("\n")[2]
    if not isinstance(meta, dict):
        meta = {}
    name = str(meta.get("name") or path.parent.name).strip()
    description = " ".join(str(meta.get("description") or "").split())
    tools = meta.get("allowed-tools") or meta.get("allowed_tools")
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",")]
    allowed = frozenset(t for t in tools if t) if isinstance(tools, list) else None
    if not name or not body.strip():
        return None
    return Skill(name=name, description=description, path=path, body=body.strip(), allowed_tools=allowed, source=source)


def discover_skills(repo_path: str | None = None) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for root, source in _search_roots(repo_path):
        if not root.is_dir():
            continue
        for skill_md in sorted(root.glob("*/SKILL.md")):
            skill = _parse(skill_md, source)
            if skill is not None and skill.name not in found:
                found[skill.name] = skill
    return found


def fingerprint(repo_path: str | None = None) -> tuple:
    """Changes whenever a skill is added/removed/edited — part of the agent
    cache key (agent_builder.py), since the skill list is baked into the
    cached system prompt."""
    items = []
    for skill in discover_skills(repo_path).values():
        try:
            mtime = skill.path.stat().st_mtime
        except OSError:
            mtime = 0
        items.append((skill.name, str(skill.path), mtime))
    return tuple(sorted(items))


def render_skill(skill: Skill, args: str = "") -> str:
    """What the model reads when a skill is used — model-facing, English.
    $ARGUMENTS in the body is replaced with the invocation's arguments
    (Claude Code convention); without the placeholder they're appended."""
    body = skill.body
    args = (args or "").strip()
    if "$ARGUMENTS" in body:
        body = body.replace("$ARGUMENTS", args)
        args = ""
    parts = [
        f'Skill "{skill.name}" loaded. Follow these instructions for the current task. '
        f"Paths in them are relative to the skill's base directory: {skill.base_dir}",
        "",
        body,
    ]
    if args:
        parts += ["", f"Arguments: {args}"]
    return "\n".join(parts)


def skills_prompt_block(repo_path: str | None) -> str:
    skills = discover_skills(repo_path)
    if not skills:
        return ""
    lines = [
        "\n\n## Skills",
        "The skills below are ready-made instructions for specific kinds of "
        "tasks in THIS working directory. If the user's request matches a "
        "skill's description, your FIRST action must be calling the `skill` "
        "tool with that name (pass any details from the request as "
        "`arguments`), then do exactly what the returned instructions say, "
        "using your tools. Never ask the user for something a skill already "
        "tells you how to find. If no skill matches, ignore this section.",
        "Available skills:",
    ]
    for skill in skills.values():
        lines.append(f"- {skill.name}: {skill.description or '(no description)'}")
    return "\n".join(lines)


def build_skill_tool(repo_path: str | None):
    @tool
    def skill(name: str, arguments: str = "") -> str:
        """Load a skill's full instructions by name (the list of available
        skills and what each is for is in the system prompt). Call this
        before starting a task that matches a skill's description, then
        follow the returned instructions. `arguments` — optional arguments for the
        skill (e.g. a file path or topic from the user's request)."""
        skills = discover_skills(repo_path)
        found = skills.get(name.strip())
        if found is None:
            available = ", ".join(skills) or "none"
            return f"Error: no skill named {name!r}. Available skills: {available}."
        return render_skill(found, arguments)

    return skill


def expand_slash_command(text: str, repo_path: str | None) -> tuple[str, frozenset[str] | None] | None:
    """`/<name> args` for a markdown skill -> (task text for the model,
    allowed_tools). None if the text isn't a known skill invocation."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    head, _, args = stripped[1:].partition(" ")
    found = discover_skills(repo_path).get(head)
    if found is None:
        return None
    return render_skill(found, args), found.allowed_tools
