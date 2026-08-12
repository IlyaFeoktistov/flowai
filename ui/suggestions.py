"""Ghost-text suggestion for the input box after the AI ends its turn on a
question ("...сделать это?") — accept with Tab, dismissed for free the
moment the user types anything.

This module only decides WHEN a suggestion makes sense (looks_like_question)
and generates its text (suggest_reply). Display (Buffer.suggestion, the
"auto-suggestion" style) and the Tab-to-accept keybinding live in ui/app.py.
"""
import os
import re

import settings
from mcp_agent.model_config import OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX, MODEL_TEMPERATURE

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
    heavy just to guess a one-line reply. Same model tag as the main chat
    model so Ollama reuses the already-resident weights instead of evicting
    them for a second pair (see mcp_agent/agent_builder.py's judge_model for
    the same reasoning). Returns None on any failure — a missing suggestion
    is a no-op for the caller, never worth surfacing as an error."""
    try:
        from langchain_ollama import ChatOllama
        model = ChatOllama(
            model=settings.get("chat_model"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            keep_alive=OLLAMA_KEEP_ALIVE,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=20,
            reasoning=False,
            temperature=MODEL_TEMPERATURE,
        )
        resp = await model.ainvoke([
            {"role": "system", "content": _SUGGEST_SYSTEM_PROMPT},
            {"role": "user", "content": ai_text[-1000:]},
        ])
        text = str(resp.content).strip().strip('"').strip()
        return text or None
    except Exception:
        return None
