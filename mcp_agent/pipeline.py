"""
Точка входа нового пайплайна Router->Analyzer->Planner->Coder->Verifier
(.claude/plans/, реализация по фазам). ТОТ ЖЕ контракт stream_chat(messages,
on_event=None), что у mcp_agent/agent.py — async-generator, on_event
получает тот же набор типов событий — чтобы при решении о cutover cli.py
было достаточно поменять один импорт (см. agent.py:1-51 про то, как оно
само когда-то так заменило agent/orchestrator.py; здесь тот же манёвр).

Текущая фаза реализации: все 5 стадий собраны — Router -> Analyzer ->
Planner -> Coder <-> Verifier (последняя пара — ограниченный цикл, см.
CODER_VERIFIER_MAX_ROUNDS). После подтверждения плана эмитится "plan_steps"
(см. _parse_numbered_plan) — ui/app.py рисует чек-лист над футером
(ui/stream.py:"plan_steps"/"plan_step_done"); Coder отмечает шаги
выполненными по своему отчёту (см. stream_chat ниже, после coder-раунда).
Verifier'а реальный fail — ВАЛИДНЫЙ исход её собственного раунда (см.
mcp_agent/stages/verifier.py) — pipeline.py решает по kind="execution_failure",
возвращать ли правки Coder'у на новый круг; если круги кончились — откат
через mcp_agent/snapshots.py:_revert_turn_paths, тот же механизм, что уже
был в legacy mcp_agent/agent.py.

Ветвление здесь идёт НЕ по одному kind-enum'у, а по 4 независимым булевым
флагам от mcp_agent/router.py:classify_intent (needs_project/needs_shell/
needs_change/change_is_ambiguous) — mcp_agent/roles.py даёт функции-
компоновщики (investigator_tools/planner_tools/executor_tools/coder_tools/
verifier_tools), которые stream_chat вызывает с текущими флагами, чтобы
получить КОНКРЕТНЫЙ набор тулов для каждой стадии. Новый, не совпадающий с
прежними случай — это просто другая комбинация тех же 4 флагов, а не повод
заводить пятую хардкоженную ветку (см. router.py про живой пример: вопрос
про квантизацию локальной модели).

needs_change=true и change_is_ambiguous=false (старое kind="quick_fix") —
короткая ветка Router -> executor(investigate+write в ОДНОЙ роли) <->
Verifier: отдельная investigator/planner-стадия целиком пропускается, роль
"quick_fix" сама читает, что нужно, и правит в том же раунде. Нет ни
plan_steps-чек-листа (нет плана, который можно было бы пронумеровать), ни
ask_user — эта ветка существует именно для задач, где Planner-этап
согласования плана был бы чистым оверхедом (живой инцидент: 8 циклов
"готов ли я...?" на правке в проекте из 3 файлов, ни один так и не привёл
к реальной правке).

needs_change=false (старое kind="explain", включая его расширение на
needs_project=false+needs_shell=true — см. router.py) — короткая ветка
Router -> investigator, без Planner/Coder/Verifier: read-only вопрос (что
изменилось, что делает этот код, ревью диффа, локальное окружение), где не
подразумевается правка файла — саммари инвестигатора сразу становится
финальным ответом (см. stream_chat, ветка сразу после инвестигатора).
Существует для задач, где Planner заведомо не может ничего согласовать —
нет предполагаемой правки, только вопрос (живой инцидент, см. docstring
router.py: Planner на чисто информационном запросе заблудился в
несуществующем пути к файлу и застрял на бессмысленном уточнении).
"""
import asyncio
import os
import re
import time
from datetime import datetime

from langchain_core.messages import HumanMessage

from mcp_agent.agent import _investigation_signals  # общий подсчёт "мест разведки" для auto-capture, см. его докстринг
from mcp_agent.agent_builder import _get_role_agent
from mcp_agent.debug_log import log_event
from mcp_agent.knowledge import format_knowledge, load_knowledge, maybe_auto_capture
from mcp_agent.message_utils import _to_lc_messages
from mcp_agent.model_config import CODER_VERIFIER_MAX_ROUNDS
from mcp_agent.roles import (
    ROLE_MAX_ATTEMPTS,
    ROLE_RECURSION_LIMIT,
    coder_tools,
    executor_tools,
    investigator_tools,
    planner_tools,
    verifier_tools,
)
from mcp_agent.router import answer_casual, classify_intent
from mcp_agent.self_heal import _written_paths
from mcp_agent.snapshots import _revert_turn_paths
from mcp_agent.stage_runner import run_stage
from mcp_agent.stages.analyzer import analyzer_guidance, analyzer_verdict
from mcp_agent.stages.coder import coder_guidance, coder_verdict
from mcp_agent.stages.planner import planner_guidance, planner_verdict
from mcp_agent.stages.quick_fix import quick_fix_guidance, quick_fix_verdict
from mcp_agent.stages.verifier import verifier_guidance, verifier_verdict


def _seed_stage_payload(original_messages: list, stage_digests: list[tuple[str, str]]) -> dict:
    """Дайджест между СТАДИЯМИ пайплайна — тот же принцип, что
    mcp_agent/stage_runner.py:_seed_retry уже применяет между ПОПЫТКАМИ
    одной стадии, просто уровнем выше: дайджест — это финальный текст
    предыдущей стадии целиком (саммари Analyzer'а, план Planner'а...), а не
    список путей/команд."""
    digest_block = "\n\n".join(f"{name} findings:\n{text}" for name, text in stage_digests)
    return {"messages": [*original_messages, HumanMessage(content=digest_block)]}


# Строки вида "1. В файле X сделай Y" (mcp_agent/prompts.py:
# _planner_system_prompt требует именно такой формат) — ui/app.py:
# set_plan_steps/mark_plan_step_done рисует их как чек-лист над футером
# (см. ui/stream.py:"plan_steps"). Не пытаемся распарсить вложенные
# подпункты — один уровень нумерации, как и требует промпт Planner'а.
_NUMBERED_STEP_RE = re.compile(r"^\s*(\d+)[.\)]\s+(.+)$")


def _parse_numbered_plan(text: str) -> list[str]:
    """Live bug (67fac007f4c34ba88d899fbb175f7313): Planner's final plan
    sometimes balloons past what was actually approved in ask_user — one
    step trailed off on a dangling ':' with no content (a cut-off
    sub-list), and two other steps were exact-duplicate wording. Coder
    executes each numbered step literally and never merges/skips one, so a
    dangling or duplicate step becomes a real, separate (and inconsistent)
    edit. This is a mechanical backstop for the prompt instruction in
    _planner_system_prompt (mcp_agent/prompts.py) — drop what slips
    through rather than trust wording alone."""
    steps = []
    seen = set()
    for line in text.splitlines():
        m = _NUMBERED_STEP_RE.match(line)
        if not m:
            continue
        step = m.group(2).strip()
        if not step or step.endswith(":"):
            continue
        key = step.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        steps.append(step)
    return steps


# Отслеживаем реально тронутые файлы В РЕАЛЬНОМ ВРЕМЕНИ, по tool_start, а
# не постфактум по StageResult.all_round_msgs (см. _written_paths) — если
# отмена (Ctrl+C) прилетает СЕРЕДИНЕ run_stage, до того как он успеет
# вернуть результат, all_round_msgs теряется вместе с ним, и touched_paths
# остался бы пустым в except-ветке ниже ровно тогда, когда предупреждение
# нужнее всего. copy_lines мутирует dest_path, не path.
_WRITE_TOOL_PATH_KEY = {
    "write_file": "path", "edit_file": "path", "replace_lines": "path",
    "insert_lines": "path", "copy_lines": "dest_path",
}


def _track_touched_paths(on_event, touched_paths: set[str]):
    async def wrapped(event: dict) -> None:
        if event.get("type") == "tool_start":
            key = _WRITE_TOOL_PATH_KEY.get(event.get("name"))
            if key:
                path = (event.get("args") or {}).get(key)
                if path:
                    touched_paths.add(path)
        if on_event:
            await on_event(event)
    return wrapped


def _inject_knowledge(original_messages: list, knowledge_text: str | None) -> list:
    """Вставляет knowledge отдельным сообщением прямо перед ТЕКУЩИМ вопросом
    пользователя — не в самое начало истории (иначе на десятом ходу той же
    сессии это читалось бы как "сказано до первой реплики"). Та же логика,
    что раньше жила в mcp_agent/agent.py:stream_chat, перенесена сюда и
    вызывается ОДИН раз на весь пайплайн (не на каждую стадию)."""
    if not knowledge_text or not original_messages:
        return original_messages
    return [
        *original_messages[:-1],
        HumanMessage(content=(
            "(Persistent project knowledge saved by earlier sessions — use "
            "it instead of re-deriving the same thing by re-reading files it "
            "already covers; it may be incomplete or stale, verify against "
            "real files if something looks off.)\n\n" + knowledge_text
        )),
        original_messages[-1],
    ]


def _inject_note_before_last(messages: list, note: str) -> list:
    if not messages:
        return messages
    return [*messages[:-1], HumanMessage(content=note), messages[-1]]


def _investigator_scope_note(is_final_answer: bool, is_followup: bool = False) -> str | None:
    """Extra per-call guidance for the investigator ("analyzer") stage,
    covering things its STATIC system prompt (prompts.py:
    _analyzer_system_prompt) can't know ahead of time — whether its summary
    is the final user-facing answer or feed for a Planner, and whether this
    is a follow-up in an ongoing conversation that already did related
    investigation. Injected as a per-call message instead of baked into the
    cached system prompt — same pattern _seed_stage_payload already uses
    for cross-stage digests — so is_final_answer/is_followup don't need to
    become extra dimensions of the agent cache key (agent_builder.py:
    _get_role_agent).

    Used to also carry a needs_project=false branch ("this round you do NOT
    have project tools, don't bother trying") — removed once project-read
    tools became unconditional in roles.py:investigator_tools, same class
    of fix as _SHELL_TOOLS earlier: Router's needs_project classification
    (the same quantized chat model, fallible) was plainly wrong on "what
    parameters does this project's model have" (a project-scoped question
    if there ever was one), and this note then actively told the model to
    refuse instead of trying — a live incident (2026-08-14, glm-4.7-flash)
    reproduced exactly that: the model obediently answered "I can't access
    project files" to a question that needed nothing more than
    `cat mcp_agent/model_config.py` (bash_exec, which it always had
    regardless of needs_project).

    Live incidents this still fixes: (1) final answers to pure questions
    came back with Planner-style "inventory" framing and unwanted
    meta-commentary ("mission accomplished") because the prompt always
    assumes a Planner reads this next; (2) "проведи анализ на дыры в
    безопасности" followed a turn later by "попробуй исправить самое
    простое" — the user's own earlier message list already contained this
    stage's OWN full report (files, line numbers, code) from the first
    turn, but this stage re-ran the same searches/reads from zero anyway,
    burning ~2.5 minutes before Planner even started on what should have
    been a two-line fix."""
    parts = []
    if is_final_answer:
        parts.append(
            "No Planner/Coder stage runs after you this time — your final "
            "summary below goes DIRECTLY to the user as the actual answer "
            "to their question. Answer it plainly and completely: do NOT "
            "write a Planner-style file/symbol inventory, and don't add "
            "meta-commentary about your own process or completion status "
            "('mission accomplished' framing) — just answer the question."
        )
    if is_followup:
        parts.append(
            "This is a follow-up in an ongoing conversation, shown below — "
            "check it FIRST. If an earlier message here (including one of "
            "your own past reports) already named the specific file(s)/"
            "line(s)/code relevant to THIS request, don't re-run the same "
            "broad search-the-whole-project investigation from scratch: "
            "read only that already-named spot to confirm it still looks "
            "the same, and reuse the rest of the earlier finding as-is. "
            "Only fall back to a full fresh investigation for parts that "
            "genuinely weren't covered before, or that later messages in "
            "this same conversation show were since changed."
        )
    if not parts:
        return None
    return "(" + " ".join(parts) + ")"


async def stream_chat(messages: list[dict], on_event=None):
    turn_start = time.monotonic()
    # Якорь ДО первой возможной правки этого хода — тот же принцип, что
    # turn_start_wall в legacy mcp_agent/agent.py:stream_chat: auto-revert
    # (_revert_turn_paths) должен откатывать только правки ИЗ ЭТОГО хода.
    turn_start_wall = datetime.now().isoformat(timespec="seconds")
    if not messages:
        if on_event:
            await on_event({"type": "done"})
        yield "⚠️ Нет сообщений для обработки."
        return

    flags = await classify_intent(messages)
    log_event("pipeline_route", **flags)

    # Правка ЭТОГО проекта не бывает без его чтения — форсим здесь, а не
    # просим классификатор держать эту связь в голове сам (см. router.py:
    # промпт уже говорит "лишь один флаг влияет на другой" один раз для
    # change_is_ambiguous, второй такой связи туда не добавляем).
    needs_project = flags["needs_project"] or flags["needs_change"]
    needs_shell = flags["needs_shell"]
    needs_change = flags["needs_change"]
    change_is_ambiguous = flags["change_is_ambiguous"] and needs_change

    if not (needs_project or needs_shell or needs_change):
        text = await answer_casual(messages, on_event=on_event)
        if on_event:
            await on_event({
                "type": "stats", "tokens_in": 0, "tokens_out": 0, "tokens_in_content": 0,
                "duration_ms": int((time.monotonic() - turn_start) * 1000),
            })
            await on_event({"type": "done"})
        yield text
        return

    repo_path = os.getcwd()
    knowledge = await load_knowledge(repo_path)
    knowledge_text = format_knowledge(knowledge) if knowledge else None
    original_messages = _inject_knowledge(_to_lc_messages(messages), knowledge_text)

    tokens_in = tokens_out = 0
    investigated_items: set = set()
    saved_knowledge_this_turn = False
    judge_model = None

    async def _finish(text: str):
        if on_event:
            await on_event({
                "type": "stats", "tokens_in": tokens_in, "tokens_out": tokens_out,
                "tokens_in_content": tokens_in,
                "duration_ms": int((time.monotonic() - turn_start) * 1000),
            })
            await on_event({"type": "done"})
        return text

    async def _capture_if_useful(note_context: str) -> None:
        # Разведка не должна теряться, если пайплайн НЕ дошёл до успешного
        # конца (Planner/Coder/Verifier исчерпали бюджет, Verifier так и не
        # подтвердил успех и т.д.) — живой прогон: Analyzer честно прочитал
        # 5+ файлов, Planner дважды упёрся в лимит, пользователь остановил
        # ход сам, и вся эта работа пропала без следа. Вызывается на КАЖДОМ
        # пути завершения после того, как Analyzer уже что-то дал, не только
        # на happy path в самом конце.
        if not saved_knowledge_this_turn and len(investigated_items) >= 4:
            await maybe_auto_capture(judge_model, repo_path, messages[-1].get("content", ""), investigated_items, note_context)

    if needs_change and not change_is_ambiguous:
        # Роутер уже решил, что задача узкая и однозначная (см.
        # mcp_agent/router.py) — отдельная investigator/planner-стадия
        # целиком пропускается, роль "quick_fix" (executor_tools) сама
        # читает, что нужно, и правит в том же раунде, без плана и без
        # ask_user.
        plan_steps: list[str] = []
        planner_text = ""
        plan_context_text = (
            "Classified as a narrow, unambiguous change — no separate "
            "investigation/planning stage, this stage reads whatever "
            "files it needs and applies the edit directly."
        )
        base_digest = [("Router", plan_context_text)]
        coder_role = "quick_fix"
        exec_tool_names = frozenset(executor_tools(needs_project))
    else:
        if on_event:
            await on_event({"type": "stage_changed", "stage": "analyzer"})

        is_final_answer = not needs_change
        investigator_tool_names = frozenset(investigator_tools())
        agent, model, judge_model, tools_by_name, read_history, compact_research, _tok_est = (
            await _get_role_agent("analyzer", investigator_tool_names, repo_path)
        )

        scope_note = _investigator_scope_note(is_final_answer, is_followup=len(messages) > 1)
        analyzer_messages = (
            _inject_note_before_last(original_messages, scope_note)
            if scope_note else original_messages
        )
        analyzer_result = await run_stage(
            agent, {"messages": analyzer_messages}, on_event,
            judge_model=judge_model, tools_by_name=tools_by_name, read_history=read_history,
            verdict_fn=analyzer_verdict, guidance_fn=analyzer_guidance,
            max_attempts=ROLE_MAX_ATTEMPTS["analyzer"], recursion_limit=ROLE_RECURSION_LIMIT["analyzer"],
            stage_name="analyzer",
        )
        tokens_in += analyzer_result.tokens_in
        tokens_out += analyzer_result.tokens_out
        analyzer_text = analyzer_result.final_text.strip()
        items, saved = _investigation_signals(analyzer_result.all_round_msgs)
        investigated_items |= items
        saved_knowledge_this_turn = saved_knowledge_this_turn or saved

        if analyzer_result.hit_recursion_limit:
            await _capture_if_useful(analyzer_text or "(no summary — ran out of budget)")
            yield await _finish(
                f"⚠️ Analyzer не уложился в {ROLE_RECURSION_LIMIT['analyzer']} шагов — "
                "задача требует более узкой формулировки."
            )
            return
        if analyzer_result.hit_context_overflow:
            # Живой прогон #8 (compaction.py's module docstring, 20260814):
            # запрос разросся больше, чем помещается в контекст модели, и
            # был отклонён бэкендом ещё до генерации — run_stage уже дал
            # Analyzer'у шанс переретраить с дайджестом (stage_runner.py),
            # это финальный, ПОСЛЕДНИЙ провал.
            await _capture_if_useful(analyzer_text or "(no summary — request kept overflowing the context window)")
            yield await _finish(
                "⚠️ Расследование каждый раз разрасталось больше, чем помещается в "
                "контекст модели, и запрос отклонялся ещё до ответа. Попробуй сузить "
                "задачу (точный файл/диапазон времени вместо общего описания) или "
                "поднять num_ctx в /settings."
            )
            return
        if not analyzer_text:
            yield await _finish("⚠️ Не удалось получить саммари исследования — попробуй переформулировать задачу.")
            return

        if is_final_answer:
            # Read-only ветка (mcp_agent/router.py про инцидент с
            # "остановись") — саммари инвестигатора УЖЕ полный ответ на
            # чисто информационный запрос, Planner/Coder/Verifier здесь не
            # нужны: нет правки, которую нужно согласовывать/применять/
            # проверять.
            await _capture_if_useful(analyzer_text)
            yield await _finish(analyzer_text)
            return

        if on_event:
            await on_event({"type": "stage_changed", "stage": "planner"})

        planner_payload = _seed_stage_payload(original_messages, [("Analyzer", analyzer_text)])
        planner_tool_names = frozenset(planner_tools())
        planner_agent, _model, planner_judge, planner_tools_by_name, planner_read_history, _cr, _tok = (
            await _get_role_agent("planner", planner_tool_names, repo_path)
        )
        planner_result = await run_stage(
            planner_agent, planner_payload, on_event,
            judge_model=planner_judge, tools_by_name=planner_tools_by_name, read_history=planner_read_history,
            verdict_fn=planner_verdict, guidance_fn=planner_guidance,
            max_attempts=ROLE_MAX_ATTEMPTS["planner"], recursion_limit=ROLE_RECURSION_LIMIT["planner"],
            stage_name="planner",
        )
        tokens_in += planner_result.tokens_in
        tokens_out += planner_result.tokens_out
        planner_text = planner_result.final_text.strip()

        if planner_result.hit_recursion_limit:
            await _capture_if_useful(analyzer_text)
            yield await _finish(
                f"⚠️ Planner не уложился в {ROLE_RECURSION_LIMIT['planner']} шагов — "
                "задача требует более узкой формулировки.\n\nСаммари Analyzer'а:\n\n" + analyzer_text
            )
            return
        if planner_result.hit_context_overflow:
            await _capture_if_useful(analyzer_text)
            yield await _finish(
                "⚠️ Планирование каждый раз разрасталось больше, чем помещается в "
                "контекст модели, и запрос отклонялся ещё до ответа.\n\nСаммари "
                "Analyzer'а:\n\n" + analyzer_text
            )
            return
        if not planner_text:
            await _capture_if_useful(analyzer_text)
            yield await _finish("⚠️ Не удалось получить план — попробуй переформулировать задачу.")
            return

        plan_steps = _parse_numbered_plan(planner_text)
        if plan_steps and on_event:
            # Чек-лист над футером (ui/app.py:set_plan_steps) — появляется сразу
            # после подтверждения плана, Coder отмечает шаги по ходу выполнения
            # (см. ниже, после его раунда).
            await on_event({"type": "plan_steps", "steps": plan_steps})

        plan_context_text = planner_text
        base_digest = [("Analyzer", analyzer_text), ("Planner", planner_text)]
        coder_role = "coder"
        exec_tool_names = frozenset(coder_tools())

    coder_agent, _m1, coder_judge, coder_tools_by_name, coder_read_history, _cr1, _t1 = (
        await _get_role_agent(coder_role, exec_tool_names, repo_path)
    )
    verifier_agent, _m2, verifier_judge, verifier_tools_by_name, verifier_read_history, _cr2, _t2 = (
        await _get_role_agent("verifier", frozenset(verifier_tools()), repo_path)
    )
    if judge_model is None:
        # quick_fix пропустил Analyzer — берём judge той же роли, что
        # реально запускалась (coder_judge для quick_fix — просто
        # _ChatOllamaWithNumKeep, собранная под роль "quick_fix", см.
        # agent_builder.py:_build_role_agent).
        judge_model = coder_judge
    exec_verdict_fn = coder_verdict if coder_role == "coder" else quick_fix_verdict
    exec_guidance_fn = coder_guidance if coder_role == "coder" else quick_fix_guidance

    touched_paths: set[str] = set()
    verifier_feedback: str | None = None
    coder_text = ""
    verifier_text = ""

    try:
        for _round_n in range(CODER_VERIFIER_MAX_ROUNDS):
            if on_event:
                await on_event({"type": "stage_changed", "stage": coder_role})

            coder_digest = list(base_digest)
            if verifier_feedback is not None:
                coder_digest.append(("Verifier", verifier_feedback))
            coder_payload = _seed_stage_payload(original_messages, coder_digest)
            coder_result = await run_stage(
                coder_agent, coder_payload, _track_touched_paths(on_event, touched_paths),
                judge_model=coder_judge, tools_by_name=coder_tools_by_name, read_history=coder_read_history,
                verdict_fn=exec_verdict_fn, guidance_fn=exec_guidance_fn,
                max_attempts=ROLE_MAX_ATTEMPTS[coder_role], recursion_limit=ROLE_RECURSION_LIMIT[coder_role],
                stage_name=coder_role,
            )
            tokens_in += coder_result.tokens_in
            tokens_out += coder_result.tokens_out
            touched_paths |= _written_paths(coder_result.all_round_msgs)
            c_items, c_saved = _investigation_signals(coder_result.all_round_msgs)
            investigated_items |= c_items
            saved_knowledge_this_turn = saved_knowledge_this_turn or c_saved
            coder_text = coder_result.final_text.strip()

            if coder_result.hit_recursion_limit or coder_result.hit_context_overflow or not coder_text:
                reverted = _revert_turn_paths(touched_paths, turn_start_wall) if touched_paths else []
                stage_label = "QuickFix" if coder_role == "quick_fix" else "Coder"
                msg = (
                    f"⚠️ {stage_label} не справился с задачей"
                    + (f" (исчерпан бюджет {ROLE_RECURSION_LIMIT[coder_role]} шагов)" if coder_result.hit_recursion_limit else "")
                    + (" (запрос разросся больше контекста модели)" if coder_result.hit_context_overflow else "")
                    + "."
                )
                if reverted:
                    msg += "\n\nПравки отменены:\n" + "\n".join(f"- {r}" for r in reverted)
                await _capture_if_useful(plan_context_text)
                yield await _finish(msg + "\n\nКонтекст:\n\n" + plan_context_text)
                return

            if plan_steps and on_event:
                # Coder обязан отчитаться "numbered 1:1 with the plan" (см.
                # mcp_agent/prompts.py:_coder_system_prompt) — сколько шагов
                # он сам перечислил в отчёте, столько и отмечаем
                # выполненными. Не идеальная гранулярность (не по каждому
                # tool-call отдельно), но честная: основана на том, что
                # Coder реально заявил.
                reported = _parse_numbered_plan(coder_text)
                for i in range(min(len(reported), len(plan_steps))):
                    await on_event({"type": "plan_step_done", "index": i})

            if on_event:
                await on_event({"type": "stage_changed", "stage": "verifier"})

            verifier_digest = coder_digest + [("Coder", coder_text)]
            verifier_payload = _seed_stage_payload(original_messages, verifier_digest)
            verifier_result = await run_stage(
                verifier_agent, verifier_payload, on_event,
                judge_model=verifier_judge, tools_by_name=verifier_tools_by_name, read_history=verifier_read_history,
                verdict_fn=verifier_verdict, guidance_fn=verifier_guidance,
                max_attempts=ROLE_MAX_ATTEMPTS["verifier"], recursion_limit=ROLE_RECURSION_LIMIT["verifier"],
                stage_name="verifier",
            )
            tokens_in += verifier_result.tokens_in
            tokens_out += verifier_result.tokens_out
            v_items, v_saved = _investigation_signals(verifier_result.all_round_msgs)
            investigated_items |= v_items
            saved_knowledge_this_turn = saved_knowledge_this_turn or v_saved
            verifier_text = verifier_result.final_text.strip()

            if verifier_result.hit_recursion_limit or verifier_result.hit_context_overflow or not verifier_text:
                await _capture_if_useful(plan_context_text + "\n\n" + coder_text)
                reason = (
                    f"исчерпан бюджет {ROLE_RECURSION_LIMIT['verifier']} шагов"
                    if verifier_result.hit_recursion_limit
                    else "запрос разросся больше контекста модели"
                    if verifier_result.hit_context_overflow
                    else "пустой ответ"
                )
                yield await _finish(
                    f"⚠️ Verifier не смог завершить проверку ({reason}).\n\nОтчёт: " + coder_text
                )
                return

            if verifier_result.verdict and verifier_result.verdict.get("kind") == "execution_failure":
                verifier_feedback = verifier_text
                continue

            break
        else:
            reverted = _revert_turn_paths(touched_paths, turn_start_wall) if touched_paths else []
            msg = f"⚠️ После {CODER_VERIFIER_MAX_ROUNDS} кругов проверка так и не прошла."
            if reverted:
                msg += "\n\nПравки отменены:\n" + "\n".join(f"- {r}" for r in reverted)
            await _capture_if_useful(plan_context_text + "\n\n" + coder_text)
            yield await _finish(msg + "\n\nПоследний отчёт Verifier'а:\n\n" + (verifier_feedback or verifier_text))
            return
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Живой инцидент: пользователь остановил ход РУЧНО посреди Coder, ДО
        # того как дело дошло до Verifier — auto-revert выше срабатывает
        # только когда Verifier САМ провалил проверку внутри пайплайна, не
        # при внешнем прерывании. Файл остался в частично применённом,
        # невалидном состоянии (реальный случай: сигнатура метода была
        # стёрта на середине правки) без единого предупреждения — заметили
        # только руками через git diff. Не тихий auto-revert здесь
        # (прервали могли по причине, не связанной с качеством правки) —
        # громкое предупреждение с точным списком тронутых файлов, чтобы
        # это не пришлось искать вручную.
        if touched_paths and on_event:
            await on_event({"type": "answer_start"})
            await on_event({"type": "answer_chunk", "text": (
                "\n\n⚠️ Ход прерван до завершения проверки — возможно "
                "НЕЗАВЕРШЁННЫЕ/невалидные правки в: "
                + ", ".join(sorted(touched_paths))
                + ". Auto-revert не сработал (он срабатывает только когда "
                "Verifier сам провалил проверку, не при ручном прерывании) "
                "— проверь `git diff` перед тем как продолжать."
            )})
            await on_event({"type": "answer_end", "had_tool_calls": False})
        raise

    if coder_role == "quick_fix":
        final_report = (
            "✅ Готово (быстрая правка, без отдельного плана).\n\nЧто сделано:\n\n" + coder_text
            + "\n\nПроверка:\n\n" + verifier_text
        )
    else:
        final_report = (
            "✅ Готово.\n\nПлан:\n\n" + planner_text
            + "\n\nЧто сделано:\n\n" + coder_text
            + "\n\nПроверка:\n\n" + verifier_text
        )
    await _capture_if_useful(final_report)

    yield await _finish(final_report)
