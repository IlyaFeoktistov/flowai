"""Ghost-text suggestion for the input box after the AI ends its turn on a
question ("...сделать это?") — accept with Tab, dismissed for free the
moment the user types anything.

This module only decides WHEN a suggestion makes sense (looks_like_question)
and generates its text (suggest_reply). Display (Buffer.suggestion, the
"auto-suggestion" style) and the Tab-to-accept keybinding live in ui/app.py.
"""
import re

_TRAILING_PUNCT_RE = re.compile(r"[)\"'»›』」\]]*\s*$")


def looks_like_question(text: str) -> bool:
    """True if *text* ends with '?', ignoring trailing closing punctuation
    (e.g. 'сделать это?)' still counts)."""
    tail = _TRAILING_PUNCT_RE.sub("", text.rstrip())
    return tail.endswith("?")


_SUGGEST_SYSTEM_PROMPT = (
    "The assistant just asked the user a question at the end of its turn. "
    "Write the user's likely short affirmative reply — casual, a few words, "
    "like 'да, делай' or 'да, исправь' — in the same language the question "
    "was asked in. Output ONLY that reply, nothing else, no quotes."
)


async def suggest_reply(ai_text: str) -> str | None:
    """One-shot completion — deliberately NOT mcp_agent.agent.stream_chat's
    full agentic loop (tools, self-heal retries, episodic history writes):
    that loop exists to actually work the user's task and is way too slow/
    heavy just to guess a one-line reply.

    Goes through agent_builder._build_chat_model, not a raw ChatOllama
    pointed at the default Ollama host — same reasoning as
    router.py:_get_classify_model/_get_casual_agent: with
    expert_streaming_enabled ON, the main chat model's weights live in the
    separate llama-server fork (port 8090), and expert_streaming.py's
    ensure_running explicitly UNLOADS Ollama's own copy before starting
    that process. A raw ChatOllama here would find nothing resident to
    reuse and make Ollama load a second full copy of the same model from
    scratch — two resident instances of the same multi-GB model at once.
    _build_chat_model is the one place that already knows how to pick the
    right backend. Returns None on any failure — a missing suggestion
    is a no-op for the caller, never worth surfacing as an error."""
    try:
        import settings
        from mcp_agent.agent_builder import _build_chat_model
        model = _build_chat_model(
            model_tag=settings.get("chat_model"),
            num_predict=20,
            reasoning=False,
            num_keep=4,
            has_tools=False,
        )
        resp = await model.ainvoke([
            {"role": "system", "content": _SUGGEST_SYSTEM_PROMPT},
            {"role": "user", "content": ai_text[-1000:]},
        ])
        text = str(resp.content).strip().strip('"').strip()
        return text or None
    except Exception:
        return None
