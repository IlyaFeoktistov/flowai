"""ui/error_reporting.py:install_background_exception_handler — an orphaned
background task (e.g. an MCP server's stdio reader task, which nobody
directly awaits) can raise an exception — such as a pydantic
ValidationError on a malformed JSON-RPC frame — and asyncio's default
handler for exceptions in tasks nobody retrieves the result of just dumps
a raw traceback to real stderr, invisible to (and overwritten by) the
Rich/prompt_toolkit TUI, so the error flashes and vanishes with no trace.
This installs a handler that routes it through the app's own console
(scrollback-visible) and log_event (permanent record) instead.

Deliberately does NOT import cli.py — that module rewires sys.stdout/
sys.stderr at import time, which breaks pytest's own capture teardown."""
import asyncio
import gc

import pytest

import ui.error_reporting as error_reporting


@pytest.mark.asyncio
async def test_orphaned_task_exception_is_routed_to_console_not_lost(monkeypatch):
    printed = []
    logged = []
    monkeypatch.setattr(error_reporting.console, "print", lambda *a, **k: printed.append(a[0] if a else ""))
    monkeypatch.setattr(error_reporting, "log_event", lambda kind, **fields: logged.append((kind, fields)))

    error_reporting.install_background_exception_handler()

    async def _boom():
        raise ValueError("found 0 vulnerabilities")

    # No reference kept — asyncio only reports "exception never retrieved"
    # once the Task object itself is garbage collected (its __del__ checks
    # whether .result()/.exception() was ever called); mirrors production,
    # where nothing in the app holds onto an MCP connection's internal
    # reader task either.
    asyncio.create_task(_boom())
    await asyncio.sleep(0.05)  # let the orphaned task run and raise
    gc.collect()
    await asyncio.sleep(0)  # let the loop process the exception callback

    assert len(printed) == 1
    assert "found 0 vulnerabilities" in printed[0]
    assert len(logged) == 1
    assert logged[0][0] == "background_exception"
    assert "ValueError: found 0 vulnerabilities" in logged[0][1]["detail"]


@pytest.mark.asyncio
async def test_survives_a_message_only_context_with_no_exception_object(monkeypatch):
    """asyncio can call the handler with just a message, no exception
    instance (e.g. "Task was destroyed but it is pending") — must not
    itself raise trying to format a None traceback."""
    printed = []
    logged = []
    monkeypatch.setattr(error_reporting.console, "print", lambda *a, **k: printed.append(a[0] if a else ""))
    monkeypatch.setattr(error_reporting, "log_event", lambda kind, **fields: logged.append((kind, fields)))

    error_reporting.install_background_exception_handler()
    loop = asyncio.get_running_loop()
    loop.call_exception_handler({"message": "Task was destroyed but it is pending"})

    assert logged[0][1]["detail"] == "Task was destroyed but it is pending"
    assert "Task was destroyed but it is pending" in printed[0]
