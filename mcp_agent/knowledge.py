"""
Общая логика knowledge-хранилища — вынесена сюда из
mcp_agent/servers/knowledge_server.py, чтобы agent.py тоже мог читать и
писать knowledge НАПРЯМУЮ (без похода через MCP-подпроцесс), не дублируя
формат данных (category -> key -> value) в двух местах:

- format_knowledge/load_knowledge — auto-inject в agent.py:stream_chat
  (каждый ход, дёшево: прямой SQLiteMemoryStore.load()), избавляет модель
  от необходимости САМОЙ вспомнить вызвать тул get_knowledge (живой прогон:
  за всю историю проекта вызван 2 раза, несмотря на явную инструкцию в
  system prompt).
- save_knowledge_entry — то же, что делает тул update_knowledge.
- save_auto_note — программная запись авто-факта, минующая approval-гейт
  (см. mcp_agent/config.py:TOOLS_REQUIRING_APPROVAL) и саму модель: решение
  сохранить факт принимает детерминированный код в agent.py:stream_chat
  после хода, а не модель через tool-call.
"""
import os
from datetime import datetime

from memory import get_store
from mcp_agent.debug_log import log_event
from utils.parsing import parse_json_loose

AUTO_CATEGORY = "auto"
_AUTO_MAX_ENTRIES = 20


def project_key(repo_path: str) -> str:
    return f"project:{os.path.abspath(repo_path)}"


async def load_knowledge(repo_path: str) -> dict:
    store = get_store()
    data = await store.load(project_key(repo_path))
    return data.get("knowledge", {})


def format_knowledge(knowledge: dict, category: str = "") -> str:
    """Тот же текстовый формат, что тул get_knowledge отдаёт модели —
    единственное место, где он определён."""
    if not knowledge:
        return "No knowledge recorded yet"
    if category:
        entries = knowledge.get(category, {})
        if not entries:
            return f"No knowledge recorded under category '{category}'"
        return f"[{category}]\n" + "\n".join(f"- {k}: {v}" for k, v in entries.items())
    lines = []
    for cat, entries in knowledge.items():
        lines.append(f"[{cat}]")
        lines.extend(f"- {k}: {v}" for k, v in entries.items())
    return "\n".join(lines)


async def save_knowledge_entry(repo_path: str, category: str, key: str, value: str) -> None:
    store = get_store()
    pkey = project_key(repo_path)
    data = await store.load(pkey)
    knowledge: dict = data.get("knowledge", {})
    knowledge.setdefault(category, {})[key] = value
    data["knowledge"] = knowledge
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    await store.save(pkey, data)


async def save_auto_note(repo_path: str, text: str) -> None:
    """FIFO-ограничение на _AUTO_MAX_ENTRIES: это накапливается без участия
    модели/пользователя на каждый достаточно нетривиальный ход (см.
    agent.py:stream_chat), так что без потолка список рос бы неограниченно
    и раздувал бы то, что format_knowledge целиком подмешивает в контекст
    каждого следующего хода. Ручные записи (любая другая category) этим
    потолком не ограничены — это осознанные решения, не автосбор."""
    store = get_store()
    pkey = project_key(repo_path)
    data = await store.load(pkey)
    knowledge: dict = data.get("knowledge", {})
    auto: dict = knowledge.setdefault(AUTO_CATEGORY, {})
    # Микросекунды, не только секунды — две записи в один и тот же вызов
    # секунды (наблюдалось в тестовом прогоне с частыми вызовами подряд)
    # иначе схлопывались бы в один ключ и тихо перезатирали друг друга
    # вместо накопления отдельных записей.
    auto[datetime.now().strftime("%Y%m%d-%H%M%S%f")] = text
    while len(auto) > _AUTO_MAX_ENTRIES:
        del auto[min(auto)]
    knowledge[AUTO_CATEGORY] = auto
    data["knowledge"] = knowledge
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    await store.save(pkey, data)


async def maybe_auto_capture(
    judge_model, repo_path: str, task_text: str, investigated_items: set[str], final_text: str,
) -> None:
    """Общая auto-capture логика для legacy mcp_agent/agent.py:stream_chat и
    mcp_agent/pipeline.py — вызывающий код уже решил, что стоит попробовать
    (обычно: >=4 разных мест разведки, ничего не сохранено вручную тулом,
    ход завершился успехом); эта функция делает сам judge-вызов
    (синтезирует факт или NONE) и запись через save_auto_note, единая для
    обоих путей, чтобы не разъезжались при следующей правке."""
    digest = "\n".join(f"- {item}" for item in sorted(investigated_items))
    note_text = "NONE"
    try:
        note_resp = await judge_model.ainvoke([
            {"role": "system", "content": (
                "You just watched a coding agent investigate a project "
                "across several files/searches, then answer a task. "
                "Respond with ONLY a JSON object {\"note\": <string or "
                "null>}. If — and only if — this investigation surfaced a "
                "durable architecture/decision/convention fact about THIS "
                "project worth remembering for a FUTURE, UNRELATED "
                "session, set note to 1-3 short concrete factual "
                "sentences citing real file paths. Do NOT summarize this "
                "task or its answer — a fact only this task cares about "
                "is not durable. If there's no such fact, set note to null."
            )},
            {"role": "user", "content": (
                f"Task: {task_text}\n\nExplored:\n{digest}\n\n"
                f"Final answer: {final_text[:1000]}"
            )},
        ])
        data = parse_json_loose(note_resp.content) or {}
        note = data.get("note")
        if isinstance(note, str) and note.strip():
            note_text = note.strip()
    except Exception as e:
        log_event("auto_knowledge_failed", error=str(e))
        return
    if note_text != "NONE":
        await save_auto_note(repo_path, note_text)
        log_event("auto_knowledge_saved", text=note_text)
