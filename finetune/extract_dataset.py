"""
Извлекает из реальной истории flowAI (~/.local/share/flowai/flowai.db,
таблица episodic_messages) обучающие примеры для LoRA-дообучения:
(история диалога → следующий вызов тула).

Структура одной сессии (см. episodic/writer.py):
  user
  answer_start
  answer_end {"had_tool_calls": true/false}
  tool_start {"name": ..., "args": ...}   <- есть, только если had_tool_calls
  tool_end   {"name": ..., "result": ...}
  answer_start
  ...
  self_heal_reject {"reason": ...}        <- относится к РАУНДУ, который
                                              только что закончился (к
                                              последнему answer_end перед ним)

Раунд, за которым сразу следует self_heal_reject, ИСКЛЮЧАЕТСЯ из позитивных
примеров — цель на этом этапе учить студента на том, что система приняла
как хорошее, а не на бракованных раундах вперемешку с хорошими.

Выход: JSONL, одна строка — один обучающий пример:
  {"session_id": ..., "seq": ..., "messages": [...], "target_tool_call": {...}}
"messages" — история В ФОРМАТЕ ролей user/assistant/tool ДО целевого вызова
(не включая его) — то, что подаётся модели на вход. "target_tool_call" —
то, что модель должна научиться сгенерировать (loss считается только на нём,
как и в нашем gpt_cli.py --qa-format).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # чтобы импортировать storage.py из корня flowAI
import storage  # noqa: E402
from mcp_agent.roles import (  # noqa: E402
    investigator_tools, planner_tools, executor_tools, coder_tools, verifier_tools,
)

# stage (из episodic role="stage_changed") -> точный набор ИМЁН тулов,
# которые роль реально видела в этот момент — единый источник правды
# mcp_agent/roles.py, а не наше приближение. needs_project=True жёстко для
# всех: pipeline.py форсит его всегда, когда needs_change=true (правка
# ЭТОГО проекта не бывает без его чтения — см. docstring executor_tools),
# а вся реальная история в episodic_messages — это работа над проектом.
_STAGE_TOOL_SETS = {
    "analyzer": lambda: investigator_tools(needs_project=True),
    "planner": lambda: planner_tools(needs_project=True),
    "quick_fix": lambda: executor_tools(needs_project=True),
    "coder": lambda: coder_tools(),
    "verifier": lambda: verifier_tools(),
    "casual": lambda: set(),  # Router: casual-ответ вообще без тулов (create_agent(tools=[]))
}


# Первый коммит, где self_heal_reject вообще стал отдельным логируемым
# событием (35f544a, "new pipeline", 2026-07-06) — до этой даты раунды в
# episodic_messages физически не могли получить self_heal_reject, ЧТО БЫ
# модель ни ответила: отсутствие reject'а там означает "проверка не
# существовала", а не "проверка прошла". live-разбор (2026-08-11): 27 из 128
# сессий в живой базе старше этой даты — без явного отсечения
# _round_was_rejected молча засчитывала все их раунды как принятые.
_SELF_HEAL_TRACKING_SINCE = "2026-07-06"


def load_sessions(db_path=None):
    conn = storage.connect() if db_path is None else __import__("sqlite3").connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT session_id FROM episodic_messages ORDER BY session_id")
    session_ids = [row[0] for row in cur.fetchall()]

    sessions = {}
    for sid in session_ids:
        cur.execute(
            "SELECT seq, role, content, ts FROM episodic_messages WHERE session_id=? ORDER BY seq",
            (sid,),
        )
        sessions[sid] = cur.fetchall()
    return sessions


def extract_examples(rows):
    """
    rows: список (seq, role, content, ts) одной сессии, по возрастанию seq.
    Возвращает список примеров-словарей (см. докстринг модуля).

    "verified" в каждом примере — False, если ts раунда раньше
    _SELF_HEAL_TRACKING_SINCE (см. её докстринг) — в этом случае "rejected"
    ничего не значит (self_heal_reject физически не мог появиться), и такой
    пример НЕ должен считаться подтверждённо хорошим только потому, что
    reject не найден."""
    examples = []
    history = []  # накопленные messages в формате user/assistant/tool
    pending_tool_call = None  # (seq, name, args) уже отправленного вызова, ждём его tool_end
    round_had_tool_call_seq = None  # seq последнего answer_end had_tool_calls=true раунда
    current_stage = None  # последний виденный role="stage_changed"; None — легаси/неизвестно (см. модуль docstring)

    for i, (seq, role, content, ts) in enumerate(rows):
        if role == "stage_changed":
            current_stage = json.loads(content).get("stage")

        elif role == "user":
            history.append({"role": "user", "content": content})

        elif role == "tool_start":
            data = json.loads(content)
            pending_tool_call = (seq, data.get("name"), data.get("args"))

        elif role == "tool_end":
            if pending_tool_call is None:
                continue
            call_seq, name, args = pending_tool_call
            if not history:
                # tool_start оказался самой первой строкой сессии — обрывок
                # восстановленного/незалогированного начала диалога, без
                # контекста учить студента нечему (и apply_chat_template не
                # умеет рендерить пустую историю)
                pending_tool_call = None
                continue

            # смотрим вперёд: отклонён ли раунд, породивший этот вызов —
            # self_heal_reject идёт СРАЗУ после следующего answer_end,
            # ищем его в ближайших последующих строках той же сессии
            rejected = _round_was_rejected(rows, i)

            tool_scope = None
            if current_stage in _STAGE_TOOL_SETS:
                tool_scope = sorted(_STAGE_TOOL_SETS[current_stage]())
                if name not in tool_scope:
                    # Роль формально не должна была звать этот тул (либо наша
                    # реконструкция needs_project/стадии неточна, либо это
                    # реальная аномалия) — не тащим противоречивый пример:
                    # либо давать неверный (без цели) tool_scope, либо
                    # рисковать, что модель решит, будто это "нормально".
                    # Откатываемся на None (легаси-путь ниже по пайплайну).
                    tool_scope = None

            example = {
                "session_id": None,  # заполним снаружи
                "seq": call_seq,
                "messages": list(history),  # копия истории ДО этого вызова
                "target_tool_call": {"name": name, "args": args},
                "rejected": rejected,
                # ts < _SELF_HEAL_TRACKING_SINCE -> self_heal_reject не мог
                # физически появиться в этой сессии, "rejected=False" здесь
                # значит "не проверялось", а не "проверено и принято".
                "verified": ts >= _SELF_HEAL_TRACKING_SINCE,
                "tool_scope": tool_scope,  # точный список имён тулов роли, или None (легаси/неизвестно)
            }
            examples.append(example)

            # добавляем сам вызов и его результат в историю для СЛЕДУЮЩИХ примеров
            data_end = json.loads(content)
            history.append({"role": "assistant", "tool_calls": [{"name": name, "args": args}]})
            history.append({"role": "tool", "name": name, "content": data_end.get("result")})
            pending_tool_call = None

        elif role == "assistant":
            # финальный текстовый ответ (без вызова тула) — тоже кладём в
            # историю, чтобы следующие примеры видели полный контекст
            history.append({"role": "assistant", "content": content})

    return examples


def _round_was_rejected(rows, tool_end_index):
    """Ищем self_heal_reject, который относится к раунду, содержащему
    tool_end на позиции tool_end_index — он идёт после СЛЕДУЮЩЕГО answer_end,
    до следующего answer_start/user."""
    for seq, role, content, ts in rows[tool_end_index + 1:]:
        if role == "self_heal_reject":
            return True
        if role in ("answer_start", "user"):
            # прошли мимо конца раунда, reject'а не было — раунд принят
            if role == "answer_start":
                continue  # answer_end внутри этого же раунда мог быть чуть позже
            return False
        if role == "tool_start":
            return False  # начался следующий раунд с новым вызовом — этот принят
    return False


def main():
    parser = argparse.ArgumentParser(description="Извлечь датасет тул-вызовов из истории flowAI")
    parser.add_argument("--db", default=None, help="путь к flowai.db (по умолчанию — как у самого flowAI)")
    parser.add_argument("--out", default="finetune/dataset.jsonl")
    parser.add_argument("--include-rejected", action="store_true",
                         help="включить в выход и отклонённые раунды (с пометкой rejected=true) — "
                              "по умолчанию они выбрасываются из позитивного датасета")
    parser.add_argument("--include-unverified", action="store_true",
                         help="включить раунды старше появления self_heal_reject в логах "
                              f"({_SELF_HEAL_TRACKING_SINCE}, verified=false) — по умолчанию "
                              "выбрасываются, т.к. для них rejected=False значит 'не проверялось', "
                              "а не 'проверено и принято'")
    args = parser.parse_args()

    sessions = load_sessions(args.db)
    print(f"Сессий в базе: {len(sessions)}")

    all_examples = []
    for sid, rows in sessions.items():
        examples = extract_examples(rows)
        for ex in examples:
            ex["session_id"] = sid
        all_examples.extend(examples)

    total = len(all_examples)
    rejected = sum(1 for e in all_examples if e["rejected"])
    unverified = sum(1 for e in all_examples if not e["verified"])
    print(f"Всего примеров вызовов тулов: {total} (отклонённых раундов: {rejected}, "
          f"без self-heal вообще/до {_SELF_HEAL_TRACKING_SINCE}: {unverified})")

    if not args.include_rejected:
        all_examples = [e for e in all_examples if not e["rejected"]]
    if not args.include_unverified:
        all_examples = [e for e in all_examples if e["verified"]]
    if not args.include_rejected or not args.include_unverified:
        print(f"После фильтрации отклонённых/непроверенных: {len(all_examples)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Сохранено в {out_path}")

    # немного статистики по частоте инструментов — полезно понять перекос датасета
    from collections import Counter
    tool_counts = Counter(e["target_tool_call"]["name"] for e in all_examples)
    print("\nЧастота тулов в датасете:")
    for name, count in tool_counts.most_common():
        print(f"  {name:30s} {count}")


if __name__ == "__main__":
    main()
