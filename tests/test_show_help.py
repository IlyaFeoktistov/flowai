"""cli.py's /help panel — used to be hand-padded strings inside a Panel,
which only aligned at one specific terminal width and broke badly at any
other (long descriptions wrapped with no indent under their own bullet,
mid-word truncation on unbreakable slash/pipe-joined tokens like the old
"logs|trash|snapshots|projects|all"). Now a Table.grid so Rich handles
column alignment/wrapping itself.

cli.py rewires sys.stdout/sys.stderr as a load-bearing side effect at
import time (see its own top-of-file comment) — importing it in-process
here would trip that up (and pytest's own output capture, see the
ValueError this hit when first tried in-process), the same reason no
other test in this suite imports cli.py directly. Rendering _show_help()
in a subprocess sidesteps it entirely — ONE subprocess for everything this
file needs (not one per width checked), since `import cli` alone is slow
enough (TUI/model-related imports) that doing it repeatedly would make
this file dominate the whole suite's runtime."""
import json
import subprocess
import sys

import pytest

_WIDTHS = (50, 70, 80, 100, 120, 160)

_DUMP_SCRIPT = f"""
import io, json
from rich.console import Console
import cli
renders = {{}}
for w in {_WIDTHS!r}:
    buf = io.StringIO()
    cli.console = Console(file=buf, width=w, force_terminal=True, no_color=True)
    cli._show_help()
    renders[w] = buf.getvalue()
print(json.dumps({{"renders": renders, "commands": [cmd for cmd, _, _ in cli._HELP_ROWS]}}))
"""


@pytest.fixture(scope="module")
def _dump():
    result = subprocess.run(
        [sys.executable, "-c", _DUMP_SCRIPT],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_help_rows_cover_every_command_in_the_completer_list(_dump):
    from ui.app import COMMANDS, _DND_ONLY_CMDS
    help_cmds = set(_dump["commands"])
    for cmd, _ in COMMANDS:
        if cmd in _DND_ONLY_CMDS:
            continue
        assert cmd in help_cmds, f"{cmd} is in the completer popup but missing from /help"


def test_help_panel_has_no_ellipsis_truncation_at_common_widths(_dump):
    renders = _dump["renders"]
    for width in _WIDTHS:
        assert "…" not in renders[str(width)], f"truncated at width {width}"
