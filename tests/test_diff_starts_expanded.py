"""StreamDisplay._fill_tool_result's start_expanded (ui/stream.py) — a
write_file/edit_file diff must be visible immediately, not one extra click
away, since it's the actual code change and the thing most worth seeing
without being asked."""
from ui.app import _OutputControl
from ui.stream import StreamDisplay


class _FakeApp:
    def __init__(self):
        self._output = _OutputControl()
        self._output.connect_app(self)

    def invalidate(self):
        pass


def test_diff_result_starts_expanded():
    app = _FakeApp()
    sd = StreamDisplay(session_stats={"tools_called": 0}, app=app)

    app._output.append("\n")
    trigger_line = len(app._output._lines) - 1
    fold = app._output.reserve_fold(trigger_line)
    app._output._lines[trigger_line] = "  ● обновляю файл x.py"

    sd._fill_tool_result(
        fold, "  ● обновляю файл x.py",
        ["[green]  1 +new line[/]"],
        start_expanded=True,
    )

    assert fold.is_expanded is True
    assert app._output._lines[trigger_line].endswith("▾")
    assert len(app._output._lines) > trigger_line + 1  # content already inserted, not hidden


def test_plain_result_stays_collapsed_by_default():
    app = _FakeApp()
    sd = StreamDisplay(session_stats={"tools_called": 0}, app=app)

    app._output.append("\n")
    trigger_line = len(app._output._lines) - 1
    fold = app._output.reserve_fold(trigger_line)
    app._output._lines[trigger_line] = "  ● читаю файл x.py"

    sd._fill_tool_result(fold, "  ● читаю файл x.py", ["[bright_black]     ↳ ok[/]"])

    assert fold.is_expanded is False
    assert app._output._lines[trigger_line].endswith("▸")
