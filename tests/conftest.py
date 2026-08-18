"""Small shared helpers for building synthetic LangChain messages in tests
— every test file that exercises verdict/guidance functions or tool-call
lookups needs the same two constructors (an AIMessage with tool_calls, a
matching ToolMessage), so they live here once instead of copy-pasted per
file."""
from langchain_core.messages import AIMessage, ToolMessage


def ai_message(tool_calls=None, content=""):
    return AIMessage(content=content, tool_calls=tool_calls or [])


def tool_message(name, content="ok", status="success", tool_call_id="1"):
    return ToolMessage(name=name, content=content, tool_call_id=tool_call_id, status=status)


def write_round(path="x.py", write_ok=True, bash_ok=None, final_text="done"):
    """A round that wrote a file and (optionally) ran bash to verify it —
    covers the common (write, then check) shape most verdict tests need.
    bash_ok=None means no bash call was made at all."""
    msgs = [
        ai_message([{"id": "w1", "name": "write_file", "args": {"path": path}}]),
        tool_message("write_file", content="ok" if write_ok else "Error: bad", status="success" if write_ok else "error", tool_call_id="w1"),
    ]
    if bash_ok is not None:
        msgs += [
            ai_message([{"id": "b1", "name": "bash", "args": {"command": f"python {path}"}}]),
            tool_message("bash", content="ok" if bash_ok else "Error (exit 1): boom", tool_call_id="b1"),
        ]
    msgs.append(AIMessage(content=final_text))
    return msgs
