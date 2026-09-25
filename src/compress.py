import re
from datetime import datetime
from typing import Callable

from memory import DEFAULT_USER, get_store
import settings

SUMMARY_PROMPT = (
    "Summarize the following conversation between a user and an AI assistant. "
    "Respond with exactly two parts, separated by a line containing only "
    "'---':\n"
    "1. DETAILED SUMMARY — concrete facts, decisions, file paths, and "
    "unresolved questions; drop small talk and step-by-step narration. A "
    "few sentences to a short paragraph. This replaces the compressed "
    "history for the assistant, so keep what it needs to stay consistent.\n"
    "2. ONE-SENTENCE RECAP — a single short sentence naming what the "
    "conversation was about, for a UI footer line (not the model, a human "
    "glancing at it). Must be ONE sentence, no more.\n"
    "Write both in the same language as the conversation. Plain text, no "
    "headers, no numbering, no extra commentary."
)


def _render(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("content"))


def _split_summary(raw: str) -> tuple[str, str]:
    """(detailed_summary, one_sentence_recap). Модель не всегда честно ставит
    разделитель '---' — если не нашли, детальная версия = весь ответ, а
    recap для футера получаем truncate'ом первого предложения, а не молча
    отдаём весь абзац в однострочный UI-футер (см. жалобу: recap был
    длинным, а не одним предложением)."""
    parts = re.split(r"\n\s*-{3,}\s*\n", raw, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    detailed = raw.strip()
    first_sentence = re.split(r"(?<=[.!?])\s", detailed, maxsplit=1)[0]
    short = first_sentence if len(first_sentence) <= 140 else first_sentence[:140].rstrip() + "…"
    return detailed, short


async def _summarize(text: str) -> str:
    """Through agent_builder._build_chat_model, not a bare ollama client:
    with expert_streaming_enabled the chat model lives in llama-server, and
    a direct Ollama call would load a SECOND full copy of it (and the
    default glm-4.7-flash can't run on plain Ollama at all)."""
    from mcp_agent.agent_builder import _build_chat_model
    model = _build_chat_model(
        model_tag=settings.get("chat_model"), num_predict=1024, reasoning=False,
        num_keep=4, has_tools=False,
    )
    resp = await model.ainvoke([
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": text},
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def should_compress(context_tokens: int | None) -> bool:
    """Between turns: compress once the LAST model call's real window fill
    (the "context" event, mcp_agent/agent.py:_stream_round) crosses
    compress_at of num_ctx. Not the turn's summed tokens_in — that re-counts
    the whole history on every call and says nothing about how full the
    window actually is."""
    if not context_tokens:
        return False
    return context_tokens > int(settings.get("num_ctx") * settings.get("compress_at"))


async def summarize_messages(messages: list[dict], max_chars: int) -> str:
    """Detailed summary of `messages` (role/content dicts) — the in-turn
    overflow recovery (mcp_agent/stage_runner.py) uses it to fold the older
    conversation into one message. Input is capped at max_chars, keeping the
    most recent part: the summarization call must fit the same window that
    just overflowed."""
    text = _render(messages)
    if len(text) > max_chars:
        text = "…" + text[-max_chars:]
    return _split_summary(await _summarize(text))[0]


async def compress_history(
    messages: list[dict],
    on_notify: Callable[[int, int], None] | None = None,
    keep_last: int = 6,
) -> list[dict]:
    """Сжимает старую часть истории в одно summary-сообщение, оставляя
    последние keep_last сообщений как есть. Пишет короткий (одно
    предложение) recap в data["recap"] (memory-store) — единственное место в
    проекте, которое реально заполняет это поле; до этого оно только
    читалось для футера UI. Детальная версия (для самой истории диалога) и
    recap (для футера) — РАЗНЫЕ тексты: раньше это было одно и то же значение
    "пара предложений — абзац", из-за чего однострочный футер разрастался на
    несколько строк."""
    if len(messages) <= keep_last:
        return messages

    old = messages[:-keep_last]
    kept = messages[-keep_last:]

    text = _render(old)
    # Same cap as summarize_messages: the old part alone can exceed the window.
    max_chars = settings.get("num_ctx") * 2
    if len(text) > max_chars:
        text = "…" + text[-max_chars:]
    detailed_summary, short_recap = _split_summary(await _summarize(text))

    store = get_store()
    data = await store.load(DEFAULT_USER)
    data["recap"] = short_recap
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    await store.save(DEFAULT_USER, data)

    if on_notify:
        on_notify(len(old), len(detailed_summary.split()))

    return [{"role": "system", "content": f"[Резюме предыдущего диалога]\n{detailed_summary}"}] + kept
