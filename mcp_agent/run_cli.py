"""
Раннер для сравнения нового MCP+LangGraph агента со старым пайплайном —
аналог scenarios/common.py, но для mcp_agent/agent.py.

Запуск:
    source .venv/bin/activate
    python3 mcp_agent/run_cli.py "проведи аудит незакоммиченных изменений"
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.console import console  # noqa: E402
from mcp_agent.agent import stream_chat, clear_session_file_snapshots  # noqa: E402


async def main(task: str) -> None:
    events = []

    async def on_event(ev: dict) -> None:
        events.append(ev)
        if ev.get("type") in ("tool_start", "tool_end"):
            line = f"  [{ev['type']}] {ev.get('name', '')} {ev.get('args', ev.get('result', ''))}"[:200]
            console.print(line)

    console.print(f"Задача: {task}\n")
    t0 = time.monotonic()
    result = None
    try:
        async for chunk in stream_chat([{"role": "user", "content": task}], on_event=on_event):
            result = chunk
    finally:
        clear_session_file_snapshots()
    elapsed = time.monotonic() - t0

    console.print(f"\n--- Ответ ({elapsed:.1f}s) ---")
    console.print(result)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "какие файлы изменены и что в них?"
    asyncio.run(main(task))
