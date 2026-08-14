"""
Router — стадия 0 пайплайна Router->Analyzer->Planner->Coder->Verifier
(mcp_agent/pipeline.py, план в .claude/plans/). Дешёвая классификация
входящего сообщения ДО того, как спавнить Analyzer — иначе "привет" или
"напиши bubble sort" запускали бы полноценное исследование проекта с
подъёмом MCP-тулов, что и было исходной жалобой пользователя.

Две независимые дешёвые операции, обе НЕ трогают mcp_agent/agent_builder.py:
_get_tools() (никаких MCP-подпроцессов не поднимается на этом пути вообще):
- classify_intent — один короткий judge-вызов (format="json", по образцу
  self_heal.py:_semantic_check, но короче) — возвращает НЕ один kind-string,
  а 4 независимых булевых флага (needs_project/needs_shell/needs_change/
  change_is_ambiguous). mcp_agent/pipeline.py САМ собирает из них, какие
  стадии запускать и какой набор тулов дать каждой (mcp_agent/roles.py:
  investigator_tools/planner_tools/executor_tools/...) — раньше здесь была
  фиксированная категория (casual/snippet/quick_fix/explain/project_task),
  и каждый новый несовпадающий с существующими случай требовал новой
  категории + нового захардкоженного маршрута в pipeline.py. Живой пример:
  вопрос "какая квантизация у локальной модели" не подходил ни под "casual"
  (нужна настоящая команда, не то, что модель просто знает), ни под
  "explain" по-хорошему (речь не про ЭТОТ проект вообще) — набор флагов
  (needs_project=false, needs_shell=true, needs_change=false) описывает
  его точно БЕЗ пятой категории; следующий похожий, но не идентичный
  случай — это просто другая комбинация тех же 4 флагов, а не повод
  добавлять шестую.
- answer_casual — прямой ответ БЕЗ единого тула (create_agent с tools=[],
  тот же рецепт, что agent_builder.py уже применяет для voice_mode), через
  mcp_agent/stage_runner.py:run_stage — чтобы сохранить обычный потоковый
  UI (answer_start/answer_chunk/...), а не отдать пользователю финальный
  текст одним куском. Используется, когда ВСЕ 4 флага false — casual/
  snippet в старой терминологии.

change_is_ambiguous (только когда needs_change=true) — узкая, однозначная
правка не нуждается в отдельном согласовании плана (старое kind="quick_fix"):
pipeline.py заходит прямо в объединённую investigate+write стадию, минуя
Planner целиком — живой инцидент, который это обходит: Planner застрял на
8 циклах "готов ли я...?" при правке в проекте из 3 файлов, где само
согласование плана было накладными расходами, не страховкой.
classify_intent остаётся fail-open в сторону true при любой неуверенности —
change_is_ambiguous=false должно ловить только случаи, где риск неверной
трактовки объёма/подхода реально низкий.

needs_change=false с needs_project/needs_shell=true (старое kind="explain")
— read-only вопрос о проекте/окружении: идёт только в investigator-стадию,
а её саммари сразу становится финальным ответом, Planner/Coder/Verifier не
запускаются вообще (см. stream_chat в pipeline.py). Живой инцидент, который
это обходит: вопрос вида "посмотри незакоммиченные правки и объясни"
раньше не имел отдельной категории — Planner, получив чисто информационный
запрос без реального "что поправить", попытался прочитать файл по
неверному пути, застрял на бессмысленном уточняющем вопросе и не
остановился, даже когда пользователь явно попросил (см. tools/confirm.py:
_is_stop_intent — отдельный, независимый фикс той же сессии).
classify_intent просит fail-open в сторону needs_change=true при любой
неуверенности — ошибочное false тише проглатывает реально нужную правку
(Coder до неё вообще не доходит), чем ошибочное true на чистом вопросе
просто тратит немного лишнего времени на Planner."""
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import settings
from mcp_agent.agent_builder import _build_chat_model
from mcp_agent.debug_log import log_event
from mcp_agent.message_utils import _to_lc_messages
from mcp_agent.model_config import JUDGE_NUM_PREDICT, OLLAMA_NUM_PREDICT
from mcp_agent.stage_runner import run_stage
from utils.parsing import parse_json_loose

_FLAG_KEYS = ("needs_project", "needs_shell", "needs_change", "change_is_ambiguous")

# Fail-open — та же asymmetry, что была у старого kind="project_task"
# дефолта: при парсинге/классификации, упавшей начисто, безопасный дефолт —
# "расследовать и, возможно, править с согласованием", не "пропустить
# мимо". Все True здесь означает: full investigation + Planner
# confirmation — самый дорогой, но и самый безопасный маршрут.
_FAIL_OPEN_FLAGS = {
    "needs_project": True, "needs_shell": True,
    "needs_change": True, "change_is_ambiguous": True,
}

# Каждый флаг — отдельная, независимая да/нет ось вместо одной категории
# (см. докстринг модуля про то, почему) — намеренно ПРОСИМ модель отвечать
# на каждый по отдельности, а не выбирать один enum-вариант, иначе она
# скатывается в те же 4-5 заученных категорий, что и раньше.
_ROUTER_CLASSIFY_PROMPT = (
    "Classify the user's message along four INDEPENDENT yes/no axes — "
    "answer EACH one on its own merits, don't pick a single category. "
    "Respond with ONLY a JSON object {\"needs_project\": bool, "
    "\"needs_shell\": bool, \"needs_change\": bool, "
    "\"change_is_ambiguous\": bool}.\n\n"
    "- \"needs_project\": true if answering requires looking at THIS "
    "project's files, git history, or code (a status question like 'what "
    "changed', an investigation like 'find bugs'/'explain how X works', "
    "any fix/feature request about this codebase). false for casual "
    "conversation, a standalone code/algorithm snippet with NO reference "
    "to this project ('write bubble sort'), or a question about the "
    "assistant's own local setup/environment/tools rather than this "
    "project's code — e.g. 'what quantization is my local model', 'how "
    "much disk space is free', 'what node version is installed'.\n"
    "- \"needs_shell\": true if answering genuinely requires RUNNING a "
    "real command or a real lookup — an installed tool/model's own "
    "version/metadata, running tests, observing actual runtime behavior, "
    "OR a real-world fact that changes over time and isn't in the "
    "model's frozen training data (current weather, news, prices, sports "
    "scores, today's exact date-dependent info). Live bug: a weather "
    "question got answered from memory as if it were small talk — the "
    "model has NO way to know tomorrow's actual forecast without a real "
    "lookup (web search or an API call), so this must be true, never "
    "casual, for ANY question whose correct answer can change day to "
    "day. false only if reading local files/git history is enough.\n"
    "- \"needs_change\": true if the user wants a file/config actually "
    "modified, not just information/explanation returned. false for a "
    "pure question, status check, explanation, or review request where "
    "nothing should be written. If there's ANY realistic chance a fix/"
    "change is wanted once you look — not just a description — answer "
    "true: false here skips planning/coding entirely, so a wrong guess "
    "silently drops a real fix request.\n"
    "- \"change_is_ambiguous\": only meaningful when needs_change is true "
    "(answer false otherwise). true UNLESS the user already names or "
    "clearly implies ONE specific, small-scope symptom to fix (a visual "
    "glitch, a wrong value, a typo, a small clearly-bounded function, "
    "RENAMING a specific named variable/parameter/function, changing a "
    "specific constant/config value) where there's realistically only one "
    "reasonable way to do it once you look — only that narrow case gets "
    "false, skipping the separate planning/confirmation step and going "
    "straight to reading+editing. Live bug: 'rename these two variables' "
    "got classified ambiguous=true and went through a full Planner/"
    "ask_user round for a rename that had exactly one reasonable "
    "interpretation — renaming a NAMED thing the user already identified "
    "is the definition of unambiguous here, even though it touches more "
    "than one line/spot. When genuinely unsure, answer true.\n\n"
    "Also lean needs_project/needs_change true if the message is a short "
    "follow-up to an ongoing project conversation shown below, even if it "
    "looks casual in isolation (e.g. 'и ещё вот это посмотри' after a "
    "project discussion).\n\n"
    "When genuinely unsure whether needs_project or needs_change should be "
    "true AT ALL, answer true for it — investigating and finding nothing "
    "relevant costs little; answering false when real project work was "
    "actually needed means it silently never happens. When genuinely "
    "unsure about change_is_ambiguous, also answer true — same reasoning: "
    "an unnecessary confirmation step costs a little time, skipping it on "
    "something that needed real planning risks an incomplete or wrong "
    "edit nobody reviewed first."
)

_CASUAL_CHAT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. This message doesn't need project "
    "investigation — either it's casual conversation, or a standalone code "
    "request with no reference to this project's files. Answer directly: "
    "for a standalone snippet/algorithm, just write the code in your "
    "response, in the language asked for — no tools, no project context "
    "needed. For casual conversation, respond naturally and briefly.\n\n"
    "Respond in the same language the user wrote in, "
    "addressing them directly."
)

_classify_model_cache = None
_classify_model_cache_key: str | None = None
_casual_agent_cache: tuple | None = None
_casual_agent_cache_key: str | None = None


def _get_classify_model():
    """format="json" здесь безопасен по той же причине, что у judge_model в
    agent_builder.py:_build_agent — этот объект используется ИСКЛЮЧИТЕЛЬНО
    для classify_intent, ответ всегда обязан быть чистым JSON.

    Через _build_chat_model (не голый _ChatOllamaWithNumKeep с зашитым
    base_url на дефолтный Ollama-хост) — живой баг (2026-08-13): пока
    основная чат-модель шла через expert-streaming backend (порт 8090,
    settings.expert_streaming_enabled), classify_intent на каждый ход всё
    равно стучался в обычный Ollama API, который тихо поднимал СВОЙ,
    отдельный полный процесс модели (~20 ГБ весов) с другим num_ctx —
    VRAM ушла с ~4 ГБ занятых до 29 МиБ свободных, два резидентных
    инстанса одной и той же модели одновременно. _build_chat_model — единая
    точка сборки, которая уже умеет выбирать backend правильно."""
    global _classify_model_cache, _classify_model_cache_key
    current_model = settings.get("chat_model")
    if _classify_model_cache is not None and _classify_model_cache_key == current_model:
        return _classify_model_cache
    _classify_model_cache = _build_chat_model(
        model_tag=current_model,
        num_predict=JUDGE_NUM_PREDICT,
        reasoning=False,
        num_keep=4,
        format="json",
    )
    _classify_model_cache_key = current_model
    return _classify_model_cache


def _coerce_flags(data) -> dict | None:
    """None если форма невалидна (не dict, отсутствующий или не-bool ключ)
    — вызывающий код тогда fail-open'ится на _FAIL_OPEN_FLAGS, а не молча
    доверяет частично распарсенным/угаданным значениям."""
    if not isinstance(data, dict):
        return None
    out = {}
    for key in _FLAG_KEYS:
        val = data.get(key)
        if not isinstance(val, bool):
            return None
        out[key] = val
    return out


async def classify_intent(messages: list[dict]) -> dict:
    """messages — вся текущая история хода (тот же список, что приходит в
    stream_chat), не только последнее сообщение — короткий пользовательский
    ответ вида "и второе:" в разгаре проектного разговора не должен
    читаться как casual в изоляции от контекста. Fail-open на
    _FAIL_OPEN_FLAGS — обратный принцип self_heal.py:_semantic_check (тот
    fail-open в сторону "не блокировать"), здесь безопасный дефолт —
    "расследовать", не "пропустить мимо". Возвращает dict с 4 булевыми
    флагами (_FLAG_KEYS) — mcp_agent/pipeline.py собирает из них конкретный
    прогон (какие стадии, какие тулы у каждой), см. docstring модуля."""
    model = _get_classify_model()
    recent = messages[-2:] if len(messages) >= 2 else messages
    convo = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent)
    try:
        resp = await model.ainvoke([
            {"role": "system", "content": _ROUTER_CLASSIFY_PROMPT},
            {"role": "user", "content": convo},
        ])
        data = parse_json_loose(resp.content) or {}
        flags = _coerce_flags(data)
        if flags is not None:
            log_event("router_classified", **flags)
            return flags
        log_event("router_classify_invalid", raw=str(data))
    except Exception as e:
        log_event("router_classify_failed", error=str(e))
    return dict(_FAIL_OPEN_FLAGS)


async def _get_casual_agent():
    """Ноль тулов, ноль MCP-подпроцессов — НЕ вызывает agent_builder.py:
    _get_tools() вообще, в отличие от ролей пайплайна (mcp_agent/roles.py):
    самый дешёвый путь ответа для casual/snippet, тот же рецепт, что
    agent_builder.py:_build_agent уже применяет для voice_mode
    (agent_tools=[] означает отсутствие 60-тульного оверхеда в промпте).

    Через _build_chat_model — см. docstring _get_classify_model выше про
    живой баг с двумя одновременными полными процессами модели, если
    строить ChatOllama здесь напрямую вместо единой точки сборки."""
    global _casual_agent_cache, _casual_agent_cache_key
    current_model = settings.get("chat_model")
    if _casual_agent_cache is not None and _casual_agent_cache_key == current_model:
        return _casual_agent_cache
    model = _build_chat_model(
        model_tag=current_model,
        num_predict=OLLAMA_NUM_PREDICT,
        reasoning=settings.get("show_thinking"),
        num_keep=4,
    )
    agent = create_agent(
        model, [], system_prompt=_CASUAL_CHAT_SYSTEM_PROMPT,
        middleware=[], checkpointer=InMemorySaver(),
    )
    _casual_agent_cache = (agent, model)
    _casual_agent_cache_key = current_model
    return _casual_agent_cache


def _casual_verdict(round_msgs, new_tool_msgs, round_final_text) -> dict:
    return {"relevant": True, "reason": "casual/snippet — no verification needed"}


def _casual_guidance(verdict, round_msgs, new_tool_msgs, round_final_text) -> str:
    return ""


async def answer_casual(messages: list[dict], on_event=None) -> str:
    """Прямой ответ на casual/snippet сообщение — вызывается, когда
    classify_intent вернула все 4 флага false (см. docstring модуля). Идёт через
    run_stage (не голый model.ainvoke), чтобы сохранить обычный потоковый
    UI (answer_start/answer_chunk/answer_end) — иначе пользователь увидел
    бы финальный текст одним куском вместо привычного стриминга."""
    if on_event:
        await on_event({"type": "stage_changed", "stage": "casual"})
    agent, model = await _get_casual_agent()
    payload = {"messages": _to_lc_messages(messages)}
    result = await run_stage(
        agent, payload, on_event,
        judge_model=model, tools_by_name={}, read_history={},
        verdict_fn=_casual_verdict, guidance_fn=_casual_guidance,
        max_attempts=1, recursion_limit=10, stage_name="casual",
    )
    return result.final_text
