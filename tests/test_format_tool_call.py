from ui.stream import _format_tool_call


def test_read_file_with_range():
    assert _format_tool_call("read_file", {"path": "a/b.py", "offset": 49, "limit": 51}) == \
        "читаю файл a/b.py (50-100)"


def test_read_file_with_offset_only():
    assert _format_tool_call("read_file", {"path": "a/b.py", "offset": 10}) == \
        "читаю файл a/b.py (с 11)"


def test_read_file_whole_file():
    assert _format_tool_call("read_file", {"path": "a/b.py"}) == "читаю файл a/b.py"


def test_write_and_edit_file():
    assert _format_tool_call("write_file", {"path": "x.py"}) == "обновляю файл x.py"
    assert _format_tool_call("edit_file", {"path": "x.py"}) == "обновляю файл x.py"


def test_bash():
    assert _format_tool_call("bash", {"command": "git status"}) == "выполняю команду $ git status"


def test_bash_bg():
    assert _format_tool_call("bash_bg", {"command": "sleep 100"}) == \
        "выполняю команду в фоне $ sleep 100"


def test_unknown_tool_falls_back_to_name_and_args():
    result = _format_tool_call("some_new_tool", {"foo": "bar"})
    assert result == "some_new_tool(foo=bar)"


def test_unknown_tool_no_args_falls_back_to_bare_name():
    assert _format_tool_call("flowai_guide", {}) == "flowai_guide"
