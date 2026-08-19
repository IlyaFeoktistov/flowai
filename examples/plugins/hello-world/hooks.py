"""post_file_edit/pre_commit hooks — see plugin.json's "hooks" entry and
mcp_agent/plugin_hooks.py for exactly when each runs and what a return
value means. Both may be sync or async."""
from ui.console import console


def on_edit(path, repo_path):
    """Runs after every successful write_file/edit_file — return value is
    ignored, this is a notification, not a gate."""
    console.print(f"[dim]  (hello-world plugin: noticed an edit to {path})[/]")


def on_commit(command, repo_path):
    """Runs before every `git commit ...` bash call. Returning a non-empty
    string BLOCKS the commit and shows that string to the model as the
    reason; returning None/"" lets it through. This example never blocks —
    replace the body with a real check (e.g. grep the diff for secrets)."""
    return None
