"""
Новый агент: MCP-серверы (mcp_agent/config.py) + LangGraph (create_agent) +
Ollama, вместо agent/orchestrator.py + planner/executor/verifier/synthesizer
+ tools/registry.py.

Экспортирует stream_chat(messages, on_event=None) с ТЕМ ЖЕ контрактом, что
agent/orchestrator.py:stream_chat() — async-generator, on_event получает
{"type": "tool_start"|"tool_end"|"stats"|"done", ...} — чтобы при решении о
cutover cli.py достаточно было поменять один импорт.

Этот файл держит только _stream_round (низкоуровневый astream-стример,
общий и для этого легаси-пути, и для mcp_agent/pipeline.py — см.
mcp_agent/stage_runner.py's локальный импорт), _summarize_round/
_round_call_info/_investigation_signals (дайджест-хелперы, тоже
переиспользуемые stage_runner.py/pipeline.py) и сам stream_chat. Self-heal
retry-цикл (recursion-limit/context-overflow/ResponseError-восстановление,
разбор утёкшей tool-call разметки, punt-to-user rescue, дайджест-ретраи)
здесь БОЛЬШЕ НЕ дублируется — stream_chat тонкий вызыватель
mcp_agent/stage_runner.py:run_stage, тот же общий движок, что и у
mcp_agent/pipeline.py, с собственным verdict_fn/guidance_fn
(mcp_agent/stages/legacy.py — единственная роль с финальным LLM-судьёй
fallback, т.к. этот путь делает исследование+запись+проверку в ОДНОМ
раунде, без разделения на роли, см. её докстринг). До этого — до
2026-08-18 — здесь жила ВТОРАЯ, отдельно поддерживаемая копия того же
self-heal движка (~900 строк, разошедшаяся с извлечённой в run_stage
версией в паре мест — например, run_stage успела обзавестись bonus-
попыткой на транзиентный ResponseError на последней попытке, а эта копия
нет), собранная построчным sed-копированием при первоначальном разборе
файла на подмодули. Остальное вынесено в подмодули по смыслу:
  - model_config.py   — константы (DEBUG, MAX_ATTEMPTS, лимиты Ollama)
  - prompts.py         — системный промпт + FLOWAI.md-инъекция
  - self_heal.py        — детерминированные verdict-проверки, LLM-судья,
                          разбор утёкшей tool-call разметки
  - ask_user_tool.py    — ask_user тул, permission-мидлвари, HITL-декейд
  - message_utils.py    — дедуп одинаковых ToolMessage, конвертация сообщений
  - snapshots.py        — снимки файлов и auto-revert при провале проверки
  - tool_wrappers.py    — обёртки над MCP-тулами (обрезка вывода, дедуп
                          чтений, verify-напоминание, glob-предупреждение)
  - agent_builder.py    — сборка MCP-соединений, моделей Ollama и самого
                          LangGraph-агента (_build_agent/_get_agent)
  - stage_runner.py     — общий self-heal движок (recursion-limit/context-
                          overflow/ResponseError/leaked-tool-call/punt-to-
                          user/дайджест-ретраи), используемый и здесь, и
                          mcp_agent/pipeline.py, и mcp_agent/router.py
  - stages/legacy.py    — verdict_fn/guidance_fn ЭТОЙ роли (единственной,
                          кому вообще нужен LLM-судья fallback)

Permission-диалог переиспользует САМ tools/confirm.py:ask_permission() —
не копия, тот же код. Это значит: сессионный auto-approve, поштучное
одобрение bash-команд по первому слову и, если когда-нибудь этот модуль
будет подключён из cli.py (где уже вызывается connect_confirm_app(app)),
он автоматически покажет настоящий ui/app.py-диалог вместо терминального
Y/N-фоллбэка — никакого отдельного шага интеграции перед cutover не нужно.

Запуск для сравнения (из корня репозитория):
    source .venv/bin/activate
    python3 src/mcp_agent/run_cli.py "проведи аудит незакоммиченных изменений"
"""
import os
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

# One extra dirname — this file lives one level deeper under src/
# (src/mcp_agent/agent.py) now, but .env is a real repo-root file that
# never moved there.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.types import Command  # noqa: E402

import settings  # noqa: E402
from ui.console import console  # noqa: E402
from mcp_agent import prompts  # noqa: E402
from mcp_agent.agent_builder import _get_agent  # noqa: E402
from mcp_agent.ask_user_tool import _ask_decisions  # noqa: E402
from mcp_agent.compaction import is_context_overflow_error  # noqa: E402
from mcp_agent.debug_log import log_event  # noqa: E402
from mcp_agent.delegate_tool import current_on_event as _delegate_on_event  # noqa: E402
from mcp_agent.delegate_tool import _suppress_during_subagent_tools  # noqa: E402
from mcp_agent.knowledge import format_knowledge, load_knowledge, maybe_auto_capture  # noqa: E402
from mcp_agent.message_utils import _calls_by_id, _to_lc_messages, _tool_text  # noqa: E402
from mcp_agent.model_config import DEBUG, MAX_ATTEMPTS, RECURSION_LIMIT  # noqa: E402
from mcp_agent.self_heal import _LEAK_MARKER_START_RE, _LEAK_TAIL_MARGIN, _written_paths  # noqa: E402
from mcp_agent.snapshots import _revert_turn_paths, clear_session_file_snapshots  # noqa: E402,F401 (re-exported for cli.py/run_cli.py)
from mcp_agent.stage_runner import run_stage  # noqa: E402
from mcp_agent.stages.legacy import legacy_guidance, make_legacy_verdict  # noqa: E402


async def _stream_round(
    agent, current_input, config, on_event, emitted: int, mid_turn_queue=None,
) -> tuple[dict, int, int, int, int, bool, bool, int]:
    """Крутит agent.astream() с ДВУМЯ stream_mode одновременно вместо одного
    ainvoke():
    - "values" — полное состояние графа после каждого шага (как и раньше):
      отсюда берём завершённые tool_calls/ToolMessage/usage_metadata — эти
      вещи в принципе не существуют по частям, только целиком, когда шаг
      графа закончился.
    - "messages" — токен-дельты (AIMessageChunk) от LangChain СРАЗУ по мере
      генерации, а не по завершении всего сообщения. Раньше был только
      "values", и весь текст (и намерение-перед-тулом, и финальный ответ)
      прилетал ОДНИМ блоком, когда модель уже полностью его досочинила —
      живой прогон подтвердил, что ChatOllama.astream() отдаёт реальные
      суб-словные дельты ('2', ' +', ' 2', ' =', ' 4', ...), просто это
      никуда не прокидывалось дальше "values"-агрегации.

    Оба режима идут по одному и тому же message id: "messages" даёт текст
    непрерывно как только он появляется (эмитим как answer_start/
    answer_chunk/answer_end), "values" по завершении того же сообщения даёт
    tool_calls этого сообщения (если есть) — UI решает по answer_end,
    считать ли только что показанный текст намерением-перед-тулом или
    финальным ответом.

    Возвращает (последний чанк состояния, новый emitted, tokens_in, tokens_out,
    llm_calls, hit_recursion_limit, hit_context_overflow, gen_ms) — emitted
    передаётся насквозь, чтобы не переэмитить одни и те же сообщения
    повторно на следующей попытке/резюме. llm_calls — сколько раз модель
    реально дёргалась (см. _SYSTEM_PROMPT_TOKENS_ESTIMATE ниже, зачем это
    нужно).

    gen_ms — суммарное МИЛЛИСЕКУНДНОЕ время реальной генерации токенов (окна
    между первым и последним "messages"-чанком одного AIMessage), БЕЗ
    времени выполнения тулов между шагами графа (см. живой баг: duration_ms
    в stream_chat считает весь ход целиком, включая ожидание bash/
    tool-раундтрипов и judge-вызов self_heal — на медленном локальном
    железе это давало tok/s в разы ниже реальной скорости генерации
    модели). Недооценивает на шагах, где модель вызвала тул без единого
    текстового токена (msg_chunk.content всегда пусто) — приемлемо, речь
    именно о скорости печати ТЕКСТА, а не о времени шага графа целиком.

    mid_turn_queue — опциональная asyncio.Queue[str] (см. stream_chat), через
    которую cli.py подкладывает сообщение, пришедшее от пользователя, ПОКА
    этот ход уже идёт (не во время "жду первого токена" — тот случай стрим_chat
    решает раньше, амендом текущего запроса). Живой фиче-запрос: очередь
    должна попадать в контекст МЕЖДУ шагами графа, а не ждать конца всего
    хода — иначе пользователь, поправивший себя посреди длинного
    расследования/правки, увидел бы реакцию только после того, как модель
    уже довела до конца весь первоначальный (возможно, уже неверный) план.
    Проверяется ТОЛЬКО сразу после шага, где последнее
    добавленное сообщение — ToolMessage (см. ниже) — то есть после того, как
    ТЕКУЩИЙ вызов тула точно доведён до конца, и ПЕРЕД следующим вызовом
    модели, а не посреди генерации/выполнения тула. Инъекция технически —
    тот же приём, что уже обрабатывает "__interrupt__" ниже: выходим из
    async for, стартуем НОВЫЙ agent.astream() с тем же config (тот же
    thread_id, чекпоинтер продолжает состояние) и новым HumanMessage как
    current_input — LangGraph не отличает это от обычного следующего хода
    того же треда.

    GraphRecursionError ловится ЗДЕСЬ, а не пробрасывается наружу — до этого
    исключение убивало весь round без единого шанса на digest-ретрай (см.
    stream_chat: раньше `except GraphRecursionError: hit_recursion_limit =
    True; break` — конец хода целиком, даже если модель честно копала
    правильную цепочку файлов и не зациклилась в смысле повторов, а просто
    не уложилась в лимит). LangGraph поднимает это исключение НА СТАРТЕ шага,
    который превысил бы recursion_limit — то есть `chunk` уже содержит
    ПОЛНОЕ состояние (все messages/tool_calls) всех успешно завершённых
    шагов до этого момента, просто без итогового текстового ответа. Отдаём
    это наружу как обычный (усечённый) раунд — stream_chat решает, строить
    ли из него digest и ретраить, или сдаваться, если попытки кончились.

    Тот же приём — теперь и для реального 400 "exceeds the available
    context size" (см. compaction.py's is_context_overflow_error и его
    докстринг, live-run #8): _CompactResearchMiddleware's собственный
    проактивный гейт (compaction.py:_needs_compaction) — chars//4-эвристика,
    которая может занизить реальную токенизацию (лог-дампы токенизируются
    заметно хуже прозы/кода) и не сработать ДО того, как бэкенд реально
    отклонит запрос. Раньше это исключение убивало ВЕСЬ ход необработанным
    (всплывало до cli.py, который стирал даже сообщение пользователя) —
    теперь ловится точно так же, как GraphRecursionError: `chunk` уже
    содержит все успешно завершённые шаги до отказавшего вызова модели, так
    что вызывающий код может построить из этого обычный digest и
    ретраить с чистого, маленького payload вместо того, чтобы терять весь
    прогресс."""
    tokens_in = tokens_out = llm_calls = 0
    gen_ms = 0.0
    gen_window_start: float | None = None  # см. gen_ms в докстринге выше
    chunk = None
    streaming_msg_id = None  # id AIMessage, чей текст сейчас льётся через "messages"

    # Буфер утёкшей tool-call разметки (см. _LEAK_MARKER_START_RE) — весь
    # текст ТЕКУЩЕГО стримящегося сообщения, сколько из него уже показано
    # пользователю, и найдено ли начало маркера утечки. Живой прогон: сама
    # утечка не обязательно с первого символа сообщения — до неё может идти
    # настоящий текст-намерение, который надо показать как есть, а вот
    # дальше саму разметку — нет, дальше просто копим (и никогда больше не
    # показываем) для _parse_leaked_tool_calls в stream_chat.
    full_buffer = ""
    flushed_len = 0
    leak_detected = False
    shown_anything = False

    def _reset_stream_state():
        nonlocal full_buffer, flushed_len, leak_detected, shown_anything
        full_buffer = ""
        flushed_len = 0
        leak_detected = False
        shown_anything = False

    hit_recursion_limit = False
    hit_context_overflow = False
    injected_text: str | None = None  # см. mid_turn_queue в докстринге выше
    try:
        while True:
            async for mode, payload in agent.astream(
                current_input, config=config, stream_mode=["values", "messages"]
            ):
                if mode == "messages":
                    msg_chunk, _meta = payload
                    # Live bug (user report): the live token counter only
                    # ticked on msg_chunk.content (plain text) — a tool call
                    # being generated (write_file's whole new file content,
                    # replace_lines' new_content, ...) is real, often slow,
                    # generation too, but arrives as tool_call_chunks with
                    # content usually empty, so the counter looked frozen for
                    # the entire duration and only jumped once at tool_start
                    # (mcp_agent/agent.py below), well after the fact. Same
                    # AIMessageChunk can carry both fields — check
                    # independently of the content branch below, not elif.
                    if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.tool_call_chunks and on_event:
                        arg_text = "".join(tc.get("args") or "" for tc in msg_chunk.tool_call_chunks)
                        if arg_text:
                            await on_event({"type": "tool_arg_chunk", "text": arg_text})
                    if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                        if msg_chunk.id != streaming_msg_id:
                            streaming_msg_id = msg_chunk.id
                            _reset_stream_state()
                            gen_window_start = time.monotonic()

                        full_buffer += str(msg_chunk.content)

                        if not leak_detected:
                            unflushed = full_buffer[flushed_len:]
                            leak_match = _LEAK_MARKER_START_RE.search(unflushed)
                            if leak_match:
                                # Нашли начало утечки — показываем только то,
                                # что было ДО неё (настоящий текст-намерение),
                                # саму разметку и всё после неё больше не
                                # показываем вообще.
                                safe_part = unflushed[:leak_match.start()]
                                flushed_len += leak_match.start()
                                leak_detected = True
                            else:
                                # Придерживаем хвост длиной _LEAK_TAIL_MARGIN —
                                # он может оказаться началом ещё не полностью
                                # пришедшего маркера.
                                safe_len = max(0, len(unflushed) - _LEAK_TAIL_MARGIN)
                                safe_part = unflushed[:safe_len]
                                flushed_len += safe_len

                            if safe_part and on_event:
                                if not shown_anything:
                                    shown_anything = True
                                    await on_event({"type": "answer_start"})
                                await on_event({"type": "answer_chunk", "text": safe_part})
                    continue

                # mode == "values"
                chunk = payload
                if "__interrupt__" in chunk:
                    break
                msgs = chunk["messages"]
                new_msgs = msgs[emitted:]
                for m in new_msgs:
                    if isinstance(m, AIMessage):
                        # reasoning_content есть, только когда reasoning=True у
                        # модели (settings.show_thinking) — иначе Ollama его не
                        # присылает вообще, и этот блок просто не сработает. Не
                        # переведён на потоковую выдачу вместе с answer_chunk —
                        # выключен по умолчанию (см. OLLAMA_KEEP_ALIVE выше про
                        # reasoning=False), так что цена "целым блоком" тут не
                        # платится в типичном прогоне.
                        reasoning_content = m.additional_kwargs.get("reasoning_content")
                        if reasoning_content and on_event:
                            await on_event({"type": "thinking_start"})
                            await on_event({"type": "thinking_chunk", "text": reasoning_content})
                            await on_event({"type": "thinking_end"})
                        # Сообщение закончилось — если утечки не было, но
                        # остался непоказанный "хвост" (мы его придерживали на
                        # случай разбитого чанками маркера, см. выше), риска
                        # больше нет — текст сообщения окончательный, показываем
                        # остаток.
                        if not leak_detected:
                            tail = full_buffer[flushed_len:]
                            if tail and on_event:
                                if not shown_anything:
                                    shown_anything = True
                                    await on_event({"type": "answer_start"})
                                await on_event({"type": "answer_chunk", "text": tail})
                            flushed_len = len(full_buffer)
                        # Текст этого сообщения (намерение-перед-тулом или
                        # финальный ответ) уже показан по мере генерации через
                        # answer_start/answer_chunk выше — здесь только сообщаем
                        # UI, чем сообщение закончилось, чтобы решить дальнейшее
                        # оформление (см. answer_end в ui/stream.py). Пропускаем,
                        # если ничего и не показывали (весь текст был утечкой) —
                        # answer_end без answer_start UI ничем не поможет.
                        if m.content and on_event and shown_anything:
                            await on_event({"type": "answer_end", "had_tool_calls": bool(m.tool_calls)})
                        if gen_window_start is not None:
                            gen_ms += (time.monotonic() - gen_window_start) * 1000
                            gen_window_start = None
                        streaming_msg_id = None
                        _reset_stream_state()
                        for tc in (m.tool_calls or []):
                            # No console.print here on purpose — ui/stream.py's
                            # tool_start already renders this same call (name +
                            # args) as the "🔧 name {...}" box; a raw dict echo
                            # right above it duplicated that box using a
                            # DIFFERENT truncation limit than tool_end below,
                            # which on a live run (some-site, styles.css)
                            # produced a garbled half-line right before the
                            # correct, fully-rendered one. log_event still runs
                            # unconditionally (not under DEBUG) — it's the
                            # always-on side channel debug_log.py's docstring
                            # promises, and DEBUG gating it here defeated that.
                            log_event("tool_call", name=tc["name"], args=tc["args"])
                            if on_event:
                                await on_event({"type": "tool_start", "name": tc["name"], "args": tc["args"]})
                        if m.usage_metadata:
                            tokens_in += m.usage_metadata.get("input_tokens", 0) or 0
                            tokens_out += m.usage_metadata.get("output_tokens", 0) or 0
                            llm_calls += 1
                    elif isinstance(m, ToolMessage):
                        # _tool_text (not str()) — MCP tool results are a list of
                        # content blocks ([{'type': 'text', 'text': '...'}]), and
                        # str() on that list escapes real newlines into literal
                        # '\n' text. ui/stream.py's tool_end renderer splits on
                        # real newlines to detect/color a unified diff
                        # (replace_lines' own diff output) — with str(), a
                        # multi-line diff collapsed into one "line" and never
                        # rendered; the user saw a truncated repr instead of the
                        # actual (tool-generated) diff. See message_utils.py's
                        # _tool_text docstring for the live run this came from.
                        result_text = _tool_text(m.content)
                        # No console.print here either — same reason as
                        # tool_call above, this duplicated tool_end's own
                        # rendering (which shows up to 20 full lines/2000
                        # chars) with a separate, shorter 200-char cutoff.
                        log_event("tool_result", name=m.name, result=result_text[:2000])
                        if on_event:
                            await on_event({"type": "tool_end", "name": m.name, "result": result_text[:2000]})
                emitted = len(msgs)

                # См. mid_turn_queue в докстринге — только сразу после шага
                # tool_node (последнее добавленное сообщение — ToolMessage):
                # текущий tool_call уже точно доведён до конца, следующий
                # вызов модели ещё не начался, никаких "оборванных" tool_calls
                # без ответа в персистентном состоянии не остаётся.
                if (
                    mid_turn_queue is not None
                    and new_msgs
                    and isinstance(new_msgs[-1], ToolMessage)
                    and not mid_turn_queue.empty()
                ):
                    injected_text = mid_turn_queue.get_nowait()
                    break

            if chunk is not None and "__interrupt__" in chunk:
                hitl_request = chunk["__interrupt__"][0].value
                # No console.print here — _ask_decisions below already shows
                # the real approval prompt for this same request.
                log_event("hitl_interrupt", action_requests=hitl_request["action_requests"])
                response = await _ask_decisions(hitl_request)
                current_input = Command(resume=response)
                continue
            if injected_text is not None:
                # Живой фиче-запрос ("подправить ход, не начиная заново"):
                # оборачиваем явной пометкой, что это НЕ новая, вытесняющая
                # задача — модель сама решает, отреагировать сейчас или
                # заметить и вернуться к этому после текущей работы.
                if on_event:
                    await on_event({"type": "mid_turn_injected", "text": injected_text})
                log_event("mid_turn_injected", text=injected_text)
                current_input = {"messages": [HumanMessage(content=(
                    "(The user sent this WHILE you were still mid-task on "
                    "the request above — they are not necessarily "
                    "abandoning or replacing it. Decide: if this changes "
                    "what you should do right now, address it now; if it's "
                    "unrelated or can wait, briefly acknowledge it and "
                    "keep going on the current work, then come back to it "
                    "before you finish this turn.)\n\n" + injected_text
                ))]}
                injected_text = None
                continue
            break
    except GraphRecursionError:
        # chunk уже содержит ПОЛНОЕ состояние всех успешно завершённых шагов
        # до превышения лимита (LangGraph поднимает исключение НА СТАРТЕ
        # шага, который бы его превысил) — только итогового ответа/финальной
        # остановки в нём нет. Отдаём как обычный (усечённый) раунд, а не
        # пустоту — stream_chat решает, ретраить с digest или сдаваться.
        hit_recursion_limit = True
    except Exception as e:
        # Не ловим ВСЁ подряд — только реально узнанный "контекст
        # переполнен" (см. is_context_overflow_error), всё остальное
        # (реальный сбой сети/модели и т.п.) должно всплывать как и раньше,
        # а не тихо притворяться усечённым раундом.
        if not is_context_overflow_error(e):
            raise
        hit_context_overflow = True

    return chunk, emitted, tokens_in, tokens_out, llm_calls, hit_recursion_limit, hit_context_overflow, int(gen_ms)


# Живой прогон (mail-server, 20260707-201534-f48ff36e, GraphRecursion
# digest): список раньше покрывал только read_file-семейство — добавленные с
# тех пор list_directory/directory_tree/lsp/delegate не появлялись в digest
# вообще, хотя это была БОЛЬШАЯ часть реальной разведки того раунда. Общий
# модуль-level список — используется и в _summarize_round (digest между
# попытками), и в _investigation_signals (auto-capture knowledge ниже) —
# один и тот же набор "это разведка", а не два независимых, которые могут
# разъехаться при следующем добавленном тул-имени.
_READ_TOOL_NAMES = ("read_file", "grep_search", "glob_search", "search_code_semantic", "lsp", "delegate")


def _round_call_info(round_msgs: list) -> dict[str, tuple[str, dict]]:
    return {
        tc_id: (tc["name"], tc.get("args") or {})
        for tc_id, tc in _calls_by_id(round_msgs).items()
    }


def _investigation_signals(round_msgs: list) -> tuple[set[str], bool]:
    """Для auto-capture knowledge (см. stream_chat, конец хода): сколько
    РАЗНЫХ мест разведано в этом раунде и вызывался ли update_knowledge —
    решить, стоило ли это исследование сохранить фактом на будущее, не
    полагаясь на то, что об этом вспомнит сама модель (живой прогон: за всю
    историю проекта update_knowledge вызван 2 раза)."""
    call_info = _round_call_info(round_msgs)
    read_items: set[str] = set()
    saved_knowledge = False
    for m in round_msgs:
        if not isinstance(m, ToolMessage) or m.tool_call_id not in call_info:
            continue
        name, args = call_info[m.tool_call_id]
        if name in _READ_TOOL_NAMES:
            read_items.add(str(args.get("path") or args.get("pattern") or args.get("query") or args.get("task") or "?"))
        elif name == "update_knowledge":
            saved_knowledge = True
    return read_items, saved_knowledge


def _summarize_round(round_msgs: list, verdict: dict) -> str:
    """Детерминированная (без отдельного LLM-вызова, в духе остальных
    проверок здесь) выжимка одной retry-попытки для digest между попытками
    — см. mcp_agent/stage_runner.py:_seed_retry, зачем это вообще нужно
    (используется оттуда же через _stage_digest, не только отсюда). Пути/команды вместо
    содержимого: содержимое всё равно физически лежит на диске и дёшево
    перечитывается заново (read_file/sandwich-truncation/дедуп), а
    вот НАРРАТИВ "что делалось и почему не подошло" в истории не
    восстановить, кроме как заново его туда положив."""
    call_info = _round_call_info(round_msgs)

    read_items, changed, ran = [], [], []
    for m in round_msgs:
        if not isinstance(m, ToolMessage) or m.tool_call_id not in call_info:
            continue
        name, args = call_info[m.tool_call_id]
        ok = getattr(m, "status", None) != "error"
        if name in _READ_TOOL_NAMES:
            item = str(args.get("path") or args.get("pattern") or args.get("query") or args.get("task") or "?")
            # read_file — сохраняем ТОЧНОЕ окно (offset/limit), не только
            # путь. Живой прогон (mail-server, Planner): без диапазона
            # попытка 2 перечитала ТЕ ЖЕ строки 170-190/60-70/10-20, что и
            # попытка 1 — дайджест отмечал файл как "explored" целиком,
            # хотя реально был виден только небольшой кусок, и ретрай не
            # мог отличить "уже видел эти строки" от "весь файл прочитан".
            offset, limit = args.get("offset"), args.get("limit")
            if offset is not None or limit is not None:
                item += f":offset={offset or 0}/limit={limit}"
            read_items.append(item)
        elif name in ("write_file", "edit_file"):
            target = args.get("path") or "?"
            changed.append(f"{target} ({'ok' if ok else 'FAILED'})")
        elif name == "bash":
            ran.append(f"`{str(args.get('command', ''))[:120]}` ({'ok' if ok else 'FAILED'})")

    lines = []
    if read_items:
        lines.append("- explored: " + ", ".join(dict.fromkeys(read_items)))
    if changed:
        lines.append("- changed: " + ", ".join(changed))
    if ran:
        lines.append("- ran: " + "; ".join(ran))
    if not lines:
        # Безтуловый раунд (текстовый ответ/вопрос) — единственное, что тут
        # есть для памяти, это сам текст.
        final_msg = round_msgs[-1] if round_msgs else None
        final_text = str(final_msg.content)[:300] if isinstance(final_msg, AIMessage) else ""
        if final_text:
            lines.append(f"- answered (no tools): {final_text!r}")
    lines.append(f"- rejected: {verdict['reason']}")
    return "\n".join(lines)


async def stream_chat(messages: list[dict], on_event=None, mid_turn_queue=None) -> Any:
    """mid_turn_queue — опциональная asyncio.Queue[str], проброшенная
    насквозь в _stream_round (см. её докстринг) — cli.py кладёт туда
    сообщение пользователя, пришедшее ПОКА этот ход уже идёт, вместо того
    чтобы держать его в отдельной очереди до конца хода."""
    turn_start = time.monotonic()
    # Отдельно от turn_start (monotonic, ни к чему не привязан) — нужен
    # wall-clock якорь ДО первой правки этого хода, чтобы auto-revert (см.
    # _revert_turn_paths) откатывал только правки ЭТОГО хода, а не более
    # ранние легитимные правки из прошлых ходов того же процесса (снимки
    # общие на весь _SESSION_ID, а не per-turn).
    turn_start_wall = datetime.now().isoformat(timespec="seconds")
    if not messages:
        if on_event:
            await on_event({"type": "done"})
        yield "⚠️ Нет сообщений для обработки."
        return

    # Оборачиваем ОДИН раз здесь, а не в каждом месте, где ниже вызывается
    # on_event(...) — обёртка глушит answer_*/thinking_* внешнего потока,
    # пока delegate (единственный тул легаси-агента, что зовёт sub_agent на
    # том же `model`) не закрылся своим tool_end (см. docstring в
    # delegate_tool.py про живой баг с утечкой чужого токен-стрима). Кладём
    # в ContextVar, а не передаём отдельным параметром — delegate() собран
    # ОДИН раз на весь кеш агента (agent_builder.py:_agent_cache) и не видит
    # on_event ЭТОГО конкретного хода никаким другим способом. Без reset в
    # конце: ходы в этом CLI идут строго последовательно (не параллельно),
    # так что следующий вызов stream_chat просто перезапишет значение своим
    # — оставлять старое между ходами безопасно, тул всё равно не читает
    # его вне активного astream() этого же хода.
    on_event = _suppress_during_subagent_tools(on_event)
    _delegate_on_event.set(on_event)

    agent, model, judge_model, tools_by_name, read_history, compact_research = await _get_agent()
    # Свежий тред на каждый вызов (см. _get_agent) — модель не помнит прошлые
    # ходы, так что дедуп чтений файлов (_dedupe_read_tool) должен начинаться
    # с чистого листа, а не помнить о том, что "уже читалось" в другом ходу.
    # Тот же принцип для compact_research (mcp_agent/compaction.py) — её
    # кэш дайджестов проиндексирован по содержимому конкретного раунда, а не
    # по thread_id, так что сам по себе не течёт между ходами по смыслу, но
    # без явной очистки рос бы неограниченно за время жизни процесса.
    read_history.clear()
    compact_research.clear_cache()

    task_text = messages[-1].get("content", "")
    lc_messages = _to_lc_messages(messages)
    # Auto-inject knowledge вместо того, чтобы полагаться на модель, которая
    # ДОЛЖНА сама вызвать get_knowledge (system prompt это явно требует —
    # см. prompts.py) — живой прогон: за всю историю проекта update_knowledge
    # вызван 2 раза, get_knowledge не лучше, несмотря на инструкцию. Дешёвый
    # прямой SQLiteMemoryStore.load() (без похода через MCP-подпроцесс) на
    # каждый ход, а не один раз при сборке system prompt (см.
    # mcp_agent/agent_builder.py:_get_agent — агент/промпт кешируются на весь
    # процесс по (chat_model, voice_mode) и НЕ пересобираются, когда
    # обновляется knowledge, так что статическая инъекция в system prompt
    # не увидела бы факт, сохранённый на прошлом ходу той же сессии).
    # Вставляется прямо перед ТЕКУЩИМ сообщением пользователя, а не в самое
    # начало истории — иначе на десятом ходу той же сессии он читался бы как
    # "это было сказано ДО первой реплики", а не как актуальный контекст к
    # СЕЙЧАШНЕМУ вопросу.
    knowledge = await load_knowledge(os.getcwd())
    knowledge_text = format_knowledge(knowledge) if knowledge else None
    if knowledge_text and lc_messages:
        lc_messages = [
            *lc_messages[:-1],
            HumanMessage(content=(
                "(Persistent project knowledge saved by earlier sessions — "
                "use it instead of re-deriving the same thing by re-reading "
                "files it already covers; it may be incomplete or stale, so "
                "verify against real files if something looks off.)\n\n"
                f"{knowledge_text}"
            )),
            lc_messages[-1],
        ]
    payload = {"messages": lc_messages}
    # Исходные сообщения хода передаются в run_stage как есть — дайджест
    # между retry-попытками (эквивалент _start_next_attempt) теперь строит
    # mcp_agent/stage_runner.py:_seed_retry сам, тем же способом, что и для
    # ролей пайплайна.
    original_messages = payload["messages"]

    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Task: {task_text}[/]")
    log_event("task", text=task_text)

    # settings.get("self_heal_enabled")=False -> ровно одна попытка: первый
    # ответ модели становится финальным без единого автоматического
    # ретрая, даже если legacy_verdict (mcp_agent/stages/legacy.py) сочтёт
    # его "не relevant" (ask_user-спасение в run_stage не завязано на
    # max_attempts и остаётся живым независимо от этого тумблера).
    max_attempts_effective = MAX_ATTEMPTS if settings.get("self_heal_enabled") else 1

    # Тот же self-heal движок, что и у mcp_agent/pipeline.py (recursion-
    # limit/context-overflow/ResponseError-восстановление, разбор утёкшей
    # tool-call разметки, punt-to-user rescue, дайджест-ретраи) — легаси-
    # агент отличается только своим verdict_fn/guidance_fn (mcp_agent/
    # stages/legacy.py: единственная роль с финальным LLM-судьёй fallback,
    # т.к. делает исследование+запись+проверку в ОДНОМ раунде без
    # разделения на роли, см. её докстринг) и mid_turn_queue (единственный
    # вызыватель run_stage, которому он вообще нужен — cli.py's live user
    # interjection посреди хода).
    stage_result = await run_stage(
        agent, payload, on_event,
        judge_model=judge_model, tools_by_name=tools_by_name, read_history=read_history,
        verdict_fn=make_legacy_verdict(judge_model, task_text, on_event),
        guidance_fn=legacy_guidance,
        max_attempts=max_attempts_effective, recursion_limit=RECURSION_LIMIT,
        stage_name="legacy", mid_turn_queue=mid_turn_queue,
    )

    tokens_in, tokens_out, llm_calls = stage_result.tokens_in, stage_result.tokens_out, stage_result.llm_calls
    gen_duration_ms = stage_result.gen_duration_ms
    hit_recursion_limit = stage_result.hit_recursion_limit
    hit_generation_error = stage_result.hit_generation_error
    hit_context_overflow = stage_result.hit_context_overflow
    final_verdict = stage_result.verdict
    final_text = stage_result.final_text
    attempt = stage_result.attempts_used - 1
    # Аккумулируется по ВСЕМ попыткам этого хода (StageResult.all_round_msgs,
    # не только последней) — auto-revert (см. _revert_turn_paths) должен
    # откатить каждый файл, тронутый за ход, и auto-capture knowledge должен
    # видеть разведку из КАЖДОЙ попытки, не только финальной.
    touched_paths = _written_paths(stage_result.all_round_msgs)
    investigated_items, saved_knowledge_this_turn = _investigation_signals(stage_result.all_round_msgs)

    # tokens_in считает КАЖДЫЙ вызов модели внутри хода целиком (промпт +
    # история + тул-результаты) — при нескольких tool-calling раундах system
    # prompt (~_SYSTEM_PROMPT_TOKENS_ESTIMATE токенов) пересылается заново
    # llm_calls раз. tokens_in_content — грубая оценка (см. константу выше)
    # того, что осталось БЕЗ этого повторного промпта, то есть примерно
    # реальный диалог+данные, а не накладные расходы протокола.
    prompt_overhead = prompts._SYSTEM_PROMPT_TOKENS_ESTIMATE * llm_calls
    tokens_in_content = max(0, tokens_in - prompt_overhead)

    if on_event:
        await on_event({
            "type": "stats",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_in_content": tokens_in_content,
            "duration_ms": int((time.monotonic() - turn_start) * 1000),
            "gen_duration_ms": gen_duration_ms,
        })
        await on_event({"type": "done"})

    if hit_recursion_limit:
        # Это финальная, ПОСЛЕДНЯЯ попытка из max_attempts_effective, которая
        # тоже упёрлась в лимит; сообщение показываем безусловно, раз мы
        # вообще сюда дошли.
        yield (
            f"⚠️ Не удалось получить ответ за {attempt + 1} попыт{'ку' if attempt == 0 else 'ки'} "
            f"по {RECURSION_LIMIT} шагов каждая — задача требует больше расследования, чем "
            "уместилось в бюджет. Попробуй переформулировать задачу уже, или сузить её."
        )
        return

    if hit_generation_error and not final_text:
        yield (
            "⚠️ Модель несколько раз подряд сгенерировала некорректный вызов "
            "инструмента, который не удалось разобрать. Попробуй повторить запрос."
        )
        return

    if hit_context_overflow:
        yield (
            f"⚠️ За {attempt + 1} попыт{'ку' if attempt == 0 else 'ки'} расследование "
            "каждый раз разрасталось больше, чем помещается в контекст модели, и "
            "запрос отклонялся ещё до генерации ответа. Попробуй сузить задачу "
            "(например, сразу указать точный диапазон времени/файл вместо общего "
            "\"разберись, что случилось\") или поднять num_ctx в /settings."
        )
        return

    # Попытки кончились, а verdict так и остался "не подходит" — модель
    # ответила по неполным/непроверенным данным (см. живой прогон:
    # финальный ответ вообще уехал в другую тему после исчерпания попыток
    # на diff-review + truncation). Помечаем явно вместо того, чтобы тихо
    # выдать это как обычный уверенный ответ.
    if final_verdict is not None and not final_verdict["relevant"]:
        if final_verdict.get("kind") == "execution_failure" and touched_paths:
            # Живой прогон: bash провалился дважды подряд (IndentationError
            # после плохо посчитанных replace_lines-границ), попытки кончились,
            # а сломанный agent.py остался лежать в проекте — пользователю
            # пришлось убивать процесс руками, чтобы модель не жгла третью
            # попытку впустую. Раз check детерминированно подтвердил, что
            # код не работает (не просто "не проверили", а РЕАЛЬНО падает),
            # откатываем правки этого хода сами, а не оставляем починку на
            # пользователя.
            reverted = _revert_turn_paths(touched_paths, turn_start_wall)
            if reverted:
                final_text = (
                    "⚠️ Проверка (bash) показала, что изменения не "
                    f"работают ({final_verdict['reason']}) — правки этого "
                    "хода отменены автоматически:\n"
                    + "\n".join(f"- {r}" for r in reverted)
                    + "\n\nЧто успела сделать модель до отмены:\n\n" + final_text
                )
            else:
                final_text = (
                    "⚠️ Проверка (bash) показала, что изменения не "
                    f"работают ({final_verdict['reason']}), а откатить их "
                    "автоматически не получилось — проверь состояние файлов "
                    f"вручную: {', '.join(sorted(touched_paths))}\n\n" + final_text
                )
        elif settings.get("self_heal_enabled"):
            final_text = (
                "⚠️ Не удалось до конца проверить этот ответ за отведённые "
                "попытки — часть данных могла остаться непрочитанной "
                "(обрезанный вывод инструмента, неполный дифф или незапущенный "
                "код). Вот что успела собрать модель:\n\n" + final_text
            )
        # else: self_heal_enabled=False -> ровно одна попытка (см.
        # max_attempts_effective выше) — "не удалось до конца проверить ЗА
        # ОТВЕДЁННЫЕ ПОПЫТКИ" врёт, попыток было ровно 0 сверх первой, и
        # никакого ретрая всё равно не будет. Живой баг (репорт пользователя):
        # этот баннер вешался на ЛЮБОЙ отклонённый verdict, включая ложные
        # срезы judge'а (см. живой прогон: "топ 20 тяжёлых программ" — судья
        # зацепился за стейл-контекст уже отброшенного промежуточного tool
        # call), выглядело как "модель всегда отвечает неправильно", хотя
        # почти всегда чинить было физически нечем. _semantic_check всё
        # равно считается — он же питает ask_user-punt-rescue и execution_
        # failure auto-revert (ветка выше), которые полезны независимо от
        # self_heal_enabled — просто без ретраев не показываем пользователю
        # предупреждение без всякого выхода из него.

    # Auto-capture knowledge — тот же принцип, что auto-inject выше, только
    # для ЗАПИСИ: не полагаемся на то, что модель сама вспомнит вызвать
    # update_knowledge (живой прогон: 2 раза за всю историю проекта, при
    # явной инструкции в system prompt). Порог ≥4 разных мест разведки —
    # ниже него это типичный однофайловый/точечный вопрос без durable-факта,
    # который стоило бы пересказывать будущей сессии. Пропускаем, если сама
    # модель уже сохранила знание тулом (saved_knowledge_this_turn) или ход
    # так и не прошёл self-heal (turn_succeeded) — сомнительный, непроверенный
    # результат не должен становиться "фактом" для будущих сессий.
    # Сама логика (judge-вызов + запись) — mcp_agent/knowledge.py:
    # maybe_auto_capture, общая с mcp_agent/pipeline.py, чтобы не разъезжаться.
    turn_succeeded = final_verdict is None or bool(final_verdict.get("relevant"))
    if turn_succeeded and not saved_knowledge_this_turn and len(investigated_items) >= 4:
        await maybe_auto_capture(judge_model, os.getcwd(), task_text, investigated_items, final_text)

    if DEBUG:
        console.print(
            f"[dim][MCP-AGENT] Done: tokens_in={tokens_in} tokens_out={tokens_out} "
            f"duration={time.monotonic() - turn_start:.1f}s[/]"
        )
    log_event(
        "done", tokens_in=tokens_in, tokens_out=tokens_out,
        duration_s=round(time.monotonic() - turn_start, 1),
    )

    yield final_text
