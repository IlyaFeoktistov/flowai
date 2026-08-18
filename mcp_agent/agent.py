"""
Новый агент: MCP-серверы (mcp_agent/config.py) + LangGraph (create_agent) +
Ollama, вместо agent/orchestrator.py + planner/executor/verifier/synthesizer
+ tools/registry.py.

Экспортирует stream_chat(messages, on_event=None) с ТЕМ ЖЕ контрактом, что
agent/orchestrator.py:stream_chat() — async-generator, on_event получает
{"type": "tool_start"|"tool_end"|"stats"|"done", ...} — чтобы при решении о
cutover cli.py достаточно было поменять один импорт.

Этот файл — только сам self-heal цикл (_stream_round, _summarize_round,
_start_next_attempt, stream_chat): всё, что нужно держать в голове целиком,
чтобы понять, как раунд ответа модели превращается в вердикт "готово/не
готово" и что происходит дальше. Всё остальное вынесено в подмодули по
смыслу, а не произвольно:
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
Разбито по этим границам, а не как раньше (единственный прошлый разбор на
подфайлы пересочинил код по памяти вместо копирования и перепутал границы
функции stream_chat при вырезании — см. episodic-лог сессии
20260704-105406-790965f2): каждый блок здесь скопирован из этого же файла
построчно (sed по точным диапазонам, без пересборки текста), а не написан
заново.

Permission-диалог переиспользует САМ tools/confirm.py:ask_permission() —
не копия, тот же код. Это значит: сессионный auto-approve, поштучное
одобрение bash-команд по первому слову и, если когда-нибудь этот модуль
будет подключён из cli.py (где уже вызывается connect_confirm_app(app)),
он автоматически покажет настоящий ui/app.py-диалог вместо терминального
Y/N-фоллбэка — никакого отдельного шага интеграции перед cutover не нужно.

Семантическая проверка + retry (_semantic_check в self_heal.py, MAX_ATTEMPTS
цикл в stream_chat) — портированы из verifier/verifier.py, НО без "targeted
retry" в старом смысле (повторить только конкретные failed tool_calls):
здесь create_agent сам решает, что вызвать дальше, на основе всей истории
треда + нашего корректирующего сообщения — это более адаптивная версия
того же принципа, а не буквальный порт списка failed_tools/missing_tools.

Запуск для сравнения:
    source .venv/bin/activate
    python3 mcp_agent/run_cli.py "проведи аудит незакоммиченных изменений"
"""
import os
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import ollama  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.types import Command  # noqa: E402

import settings  # noqa: E402
from tools.confirm import ask_user_question  # noqa: E402
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
from mcp_agent.model_config import DEBUG, MAX_ATTEMPTS, MAX_SELF_HEAL_ASKS, RECURSION_LIMIT  # noqa: E402
from mcp_agent.self_heal import (  # noqa: E402
    _called_ask_user,
    _described_commits_without_diff,
    _execute_leaked_tool_call,
    _execution_evidence_shows_failure,
    _extract_ask_user_shape,
    _extract_diffed_paths,
    _failed_write_messages,
    _final_answer_ignores_diff,
    _git_status_reports_changes,
    _has_diff_evidence,
    _has_execution_evidence,
    _has_truncated_output,
    _LEAK_MARKER_START_RE,
    _LEAK_TAIL_MARGIN,
    _leaked_tool_call_syntax,
    _parse_leaked_tool_calls,
    _retried_after_rejection,
    _semantic_check,
    _truncated_git_diff,
    _verified_with_syntax_check_only_despite_discoverable_tests,
    _wrote_code,
    _written_paths,
)
from mcp_agent.snapshots import _revert_turn_paths, clear_session_file_snapshots  # noqa: E402,F401 (re-exported for cli.py/run_cli.py)

_thread_counter = 0


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
    — см. _start_next_attempt, зачем это вообще нужно. Пути/команды вместо
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


def _start_next_attempt(
    original_messages: list, round_digests: list[str], correction_text: str, read_history: dict
) -> tuple[dict, dict, int]:
    """Раньше каждая retry-попытка продолжала ОДИН thread_id — чекпоинтер
    копил ПОЛНУЮ историю всех прошлых попыток, и она пересылалась целиком
    заново на каждый LLM-вызов следующей попытки. Живой прогон: 3 попытки
    на одном thread_id → 839 360 input-токенов за ход, притом что дедуп/
    sandwich-truncation внутри одной попытки в этом же прогоне ни разу не
    сработали — то есть это не мусорные повторы, а честный вес полной
    истории, компаундящийся 3 раза подряд.

    Вместо этого каждая попытка стартует с НОВЫМ thread_id и явно собранным
    payload: [исходная задача] + [компактный дайджест каждой прошлой
    попытки, см. _summarize_round] + [корректирующее сообщение для этой
    попытки]. Объём растёт линейно (несколько строк на попытку), а не
    квадратично на полный транскрипт. Возвращает (payload, config,
    emitted=0) — emitted должен сброситься, потому что новый thread_id у
    чекпоинтера начинается с пустой истории, и старый счётчик "сколько
    сообщений уже отдано как события" из прошлого треда для него неверен.

    read_history тоже обязан сброситься здесь. _dedupe_read_tool отвечает
    "вы уже читали этот файл, переиспользуйте результат" в расчёте, что
    МОДЕЛЬ реально видела тот более ранний результат в своём контексте —
    это было верно, пока все попытки жили в одном треде. Живой прогон ПОСЛЕ
    введения per-attempt тредов: попытка 2 (новый, чистый тред) попыталась
    прочитать cli.py, получила "вы уже читали это" от кэша, унаследованного
    от попытки 1 — и осталась без содержимого файла, которого в её
    собственной истории просто нет. Без сброса кэш врёт модели, что у неё
    есть данные, которых на самом деле нет в контексте ЭТОЙ попытки."""
    read_history.clear()
    global _thread_counter
    _thread_counter += 1
    new_thread_id = f"mcp-agent-{_thread_counter}"
    new_config = {"configurable": {"thread_id": new_thread_id}, "recursion_limit": RECURSION_LIMIT}
    digest_block = "\n\n".join(f"Attempt {i + 1} summary:\n{d}" for i, d in enumerate(round_digests))
    payload = {"messages": [
        *original_messages,
        HumanMessage(content=(
            f"(Continuing after {len(round_digests)} earlier attempt(s) — summarized below "
            "instead of replayed in full; the underlying files/repo state are unchanged, "
            "re-read a file/diff if you need its exact current content.)\n\n"
            f"{digest_block}\n\n{correction_text}"
        )),
    ]}
    return payload, new_config, 0


async def stream_chat(messages: list[dict], on_event=None, mid_turn_queue=None) -> Any:
    """mid_turn_queue — опциональная asyncio.Queue[str], проброшенная
    насквозь в _stream_round (см. её докстринг) — cli.py кладёт туда
    сообщение пользователя, пришедшее ПОКА этот ход уже идёт, вместо того
    чтобы держать его в отдельной очереди до конца хода."""
    global _thread_counter

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

    _thread_counter += 1
    thread_id = f"mcp-agent-{_thread_counter}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

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
    # Исходные сообщения хода — база, на которую _start_next_attempt
    # накатывает digest прошлых попыток вместо полной истории (см. там же).
    original_messages = payload["messages"]
    round_digests: list[str] = []

    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Task: {task_text}[/]")
    log_event("task", text=task_text)

    tokens_in = tokens_out = llm_calls = 0
    gen_duration_ms = 0  # чистое время генерации токенов, без тулов/judge — см. _stream_round
    emitted = 0  # сколько сообщений в result["messages"] уже отдано как события
    result = None
    hit_recursion_limit = False
    hit_generation_error = False
    hit_context_overflow = False
    final_verdict = None  # последний verdict — нужен, чтобы пометить ответ,
    # если попытки кончились, а verdict так и остался "не подходит"

    # max_attempts_effective растёт (+1), когда self-heal реально получил от
    # пользователя НОВЫЙ ответ (см. ask_user-punt ниже) — это не "провал
    # попытки", это новая информация, которую модель ещё не видела, и ей
    # нужен гарантированный ход, чтобы её использовать, даже если это
    # случилось на последней по счёту попытке.
    attempt = 0
    # settings.get("self_heal_enabled")=False -> ровно одна попытка: первый
    # ответ модели становится финальным без единого автоматического
    # ретрая, даже если _semantic_check/детерминированные вердикты ниже
    # сочтут его "не relevant" (см. settings.py про то, почему это НЕ
    # выключает ask_user-спасение — тот путь не завязан на
    # max_attempts_effective и остаётся живым).
    max_attempts_effective = MAX_ATTEMPTS if settings.get("self_heal_enabled") else 1
    self_heal_asks_used = 0
    # Аккумулируется по ВСЕМ попыткам этого хода — auto-revert (см.
    # _revert_turn_paths) должен откатить каждый файл, тронутый за ход, а
    # не только тот, что попал в последний раунд.
    touched_paths: set[str] = set()
    # Для auto-capture knowledge в конце хода (см. там же) — та же логика
    # накопления по ВСЕМ попыткам, не только последней.
    investigated_items: set[str] = set()
    saved_knowledge_this_turn = False
    while attempt < max_attempts_effective:
        if DEBUG:
            console.print(f"[dim][MCP-AGENT] Attempt {attempt + 1}/{max_attempts_effective} (thread={config['configurable']['thread_id']})[/]")
        log_event("attempt", n=attempt + 1, max=max_attempts_effective, thread=config["configurable"]["thread_id"])

        attempt_start = emitted
        try:
            result, emitted, tin, tout, calls, hit_limit_this_round, hit_overflow_this_round, round_gen_ms = await _stream_round(
                agent, payload, config, on_event, emitted, mid_turn_queue=mid_turn_queue,
            )
            tokens_in += tin
            tokens_out += tout
            llm_calls += calls
            gen_duration_ms += round_gen_ms
        except ollama.ResponseError as e:
            # Живой прогон: Ollama сама не смогла распарсить сгенерированную
            # моделью tool-call разметку (напр. кривой XML) и подняла
            # ResponseError прямо из streaming-клиента — необработанным,
            # это убивало весь процесс. В отличие от GraphRecursionError
            # (модель реально зациклилась, ретрай не лечит) — битая ОДНА
            # генерация часто лечится повторной попыткой, поэтому не
            # аборт, а обычный провал попытки с переходом на новый тред.
            if attempt == max_attempts_effective - 1:
                hit_generation_error = True
                break
            round_digests.append(f"- rejected: the model's last response failed to generate/parse ({e})")
            payload, config, emitted = _start_next_attempt(
                original_messages, round_digests,
                "Your last response failed to generate correctly — the tool-call "
                "markup was malformed and the underlying client couldn't parse it. "
                "Try again, making sure any tool call uses the real tool-calling "
                "mechanism cleanly.",
                read_history,
            )
            attempt += 1
            continue

        round_msgs = result["messages"][attempt_start:]
        new_tool_msgs = [m for m in round_msgs if isinstance(m, ToolMessage)]
        round_final = round_msgs[-1] if round_msgs else None
        round_final_text = str(round_final.content) if isinstance(round_final, AIMessage) else ""
        touched_paths.update(_written_paths(round_msgs))
        round_read_items, round_saved_knowledge = _investigation_signals(round_msgs)
        investigated_items.update(round_read_items)
        saved_knowledge_this_turn = saved_knowledge_this_turn or round_saved_knowledge

        if hit_limit_this_round:
            # Живой прогон (mail-server, 20260707-201534-f48ff36e): агент
            # честно копал реальную цепочку файлов (не повторялся) и упёрся
            # в RECURSION_LIMIT прямо посреди расследования — раньше это
            # было безусловным `break` без единого шанса продолжить. Раз
            # _stream_round теперь отдаёт ПОЛНОЕ состояние усечённого раунда
            # (см. его docstring), из него можно построить обычный digest
            # (_summarize_round уже умеет — see "explored"/"changed"/"ran")
            # и продолжить НОВОЙ попыткой со свежим recursion_limit, вместо
            # того чтобы выбрасывать всё проделанное.
            if attempt == max_attempts_effective - 1:
                hit_recursion_limit = True
                break
            final_verdict = {
                "relevant": False,
                "reason": f"ran out of its {RECURSION_LIMIT}-step budget mid-investigation",
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {final_verdict}[/]")
            log_event("verdict", **final_verdict)
            round_digests.append(_summarize_round(round_msgs, final_verdict))
            payload, config, emitted = _start_next_attempt(
                original_messages, round_digests,
                f"You ran out of your {RECURSION_LIMIT}-step budget before "
                "finishing. The summary above lists WHERE you already "
                "looked (paths/queries), not what each one returned — only "
                "re-call one of them if you actually need to see its "
                "content again, don't just repeat the same exploration "
                "path out of habit. Prioritize genuinely NEW ground over "
                "the same directories/files. If this needs many more files "
                "across many layers, hand the rest off to delegate instead "
                "of reading them all yourself.",
                read_history,
            )
            attempt += 1
            continue

        if hit_overflow_this_round:
            # Живой прогон #8 (compaction.py's module docstring, 20260814):
            # тот же приём, что hit_limit_this_round выше — chunk уже
            # содержит все успешно завершённые шаги до отказавшего вызова
            # модели (_stream_round перехватил реальный 400 "exceeds the
            # available context size" ровно как GraphRecursionError), так
            # что есть из чего строить digest и ретраить с чистого,
            # маленького payload вместо того, чтобы терять весь ход.
            if attempt == max_attempts_effective - 1:
                hit_context_overflow = True
                break
            final_verdict = {
                "relevant": False,
                "reason": "the request grew too large for the model's context window mid-investigation",
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {final_verdict}[/]")
            log_event("verdict", **final_verdict)
            round_digests.append(_summarize_round(round_msgs, final_verdict))
            payload, config, emitted = _start_next_attempt(
                original_messages, round_digests,
                "Your last request grew too large for the model's context window "
                "and was rejected before it could even run. The summary above "
                "lists WHERE you already looked — only re-call one of them if "
                "you actually need to see its content again. Be more selective "
                "this time: prefer narrower/scoped tool calls (grep a specific "
                "pattern instead of dumping a whole log, read a line range "
                "instead of a whole file) over broad dumps.",
                read_history,
            )
            attempt += 1
            continue

        if not new_tool_msgs:
            if _leaked_tool_call_syntax(round_final_text):
                # Модель сгенерировала невалидную разметку вызова тула вместо
                # настоящего structured tool_calls (см. _leaked_tool_call_syntax)
                # — тул не выполнился сам. Живой прогон: простая просьба
                # "вызови по-настоящему" не помогает — модель дважды подряд
                # генерирует ДОСЛОВНО тот же самый битый текст. Раз формат
                # известен, парсим его сами (_parse_leaked_tool_calls) и
                # выполняем тул напрямую вместо того, чтобы жечь оставшиеся
                # попытки на одной и той же генерации.
                leaked_calls = _parse_leaked_tool_calls(round_final_text)
                if leaked_calls:
                    result_parts = []
                    for call in leaked_calls:
                        if on_event:
                            await on_event({"type": "tool_start", "name": call["name"], "args": call["args"]})
                        result_text = await _execute_leaked_tool_call(tools_by_name, call["name"], call["args"])
                        if on_event:
                            await on_event({"type": "tool_end", "name": call["name"], "result": result_text[:2000]})
                        result_parts.append(f"`{call['name']}` result:\n{result_text}")
                    final_verdict = {
                        "relevant": False,
                        "reason": (
                            "the model's tool-call markup didn't parse as a real "
                            "tool call, so it was run directly instead of retried"
                        ),
                    }
                    if DEBUG:
                        console.print(f"[dim][MCP-AGENT] Deterministic verdict: {final_verdict}[/]")
                    log_event("verdict", **final_verdict)
                    if attempt == max_attempts_effective - 1:
                        break
                    round_digests.append(_summarize_round(round_msgs, final_verdict))
                    payload, config, emitted = _start_next_attempt(
                        original_messages, round_digests,
                        "Your tool-call markup didn't parse as a real tool "
                        "call, so it was executed directly instead — here "
                        "are the real results. Continue from them, don't "
                        "repeat the same call:\n\n" + "\n\n".join(result_parts),
                        read_history,
                    )
                    attempt += 1
                    continue
                # Не удалось распарсить (незнакомый/битый формат, не
                # <function=...><parameter=...>) — старое поведение: просим
                # переформулировать НАСТОЯЩИМ вызовом, а не текстом.
                final_verdict = {
                    "relevant": False,
                    "reason": (
                        "the model generated malformed tool-call markup as plain "
                        "text instead of a real structured tool call — no tool "
                        "actually ran this round"
                    ),
                }
                if DEBUG:
                    console.print(f"[dim][MCP-AGENT] Deterministic verdict: {final_verdict}[/]")
                log_event("verdict", **final_verdict)
                if attempt == max_attempts_effective - 1:
                    break
                round_digests.append(_summarize_round(round_msgs, final_verdict))
                payload, config, emitted = _start_next_attempt(
                    original_messages, round_digests,
                    "Your last message contained malformed tool-call markup "
                    "(like '<function=...>' or '<tool_call>...') as part of "
                    "its text instead of an actual tool call — no tool ran. "
                    "Call the tool using the real tool-calling mechanism, "
                    "not text written into your message content.",
                    read_history,
                )
                attempt += 1
                continue
            if "?" in round_final_text or "？" in round_final_text:
                # Дешёвый ФИЛЬТР, не вердикт: не гоняем судью на каждый
                # простой безтуловый ответ ("спасибо", "готово") — только
                # когда в тексте вообще есть вопросительный знак. Само
                # решение "отфутболила ли модель что-то пользователю текстом
                # вместо ask_user" — семантическое (см. _semantic_check) и
                # НЕ обязано зависеть от буквального "?" в тексте (можно
                # спросить и без него: "Скажите, что предпочитаете") — здесь
                # это лишь дешёвый повод спросить судью, а не сам критерий.
                if on_event:
                    await on_event({"type": "verifying_start"})
                verdict = await _semantic_check(judge_model, task_text, [], round_final_text, False)
                if on_event:
                    await on_event({"type": "verifying_end"})
                final_verdict = verdict
                if verdict["relevant"]:
                    break
                if self_heal_asks_used < MAX_SELF_HEAL_ASKS:
                    # Живой прогон: просьба "вызови ask_user как надо" не
                    # лечит — модель повторяет тот же паттерн на следующей
                    # попытке. Раз сам вопрос уже есть (это и есть текст
                    # ответа), сразу открываем настоящий интерактивный
                    # диалог с ним, а не жжём попытки на переформулировку.
                    self_heal_asks_used += 1
                    shape = await _extract_ask_user_shape(judge_model, round_final_text)
                    if on_event:
                        await on_event({"type": "tool_start", "name": "ask_user", "args": shape})
                    answer = await ask_user_question(shape["question"], shape["options"], shape["recommended"])
                    if on_event:
                        await on_event({"type": "tool_end", "name": "ask_user", "result": answer[:2000]})
                    max_attempts_effective += 1  # гарантированный ход на использование ответа
                    # Сбрасываем — пользователь только что дал НАСТОЯЩИЙ ответ,
                    # это не "непроверенный" результат, а верный источник для
                    # следующего хода. Без сброса финальный (уже нормальный)
                    # ответ модели ниже по ошибке получил бы предупреждение
                    # "⚠️ не удалось до конца проверить" от СТАРОГО verdict
                    # этого же раунда (см. живой прогон/тест).
                    final_verdict = None
                    # Этот раунд НЕ проходит через _start_next_attempt (тред
                    # продолжается как есть — полный контекст важнее экономии
                    # для настоящего ответа пользователя), но digest всё равно
                    # нужен: если КАКАЯ-ТО следующая попытка позже вызовет
                    # компакцию, она соберёт payload из original_messages +
                    # round_digests — без записи здесь обмен с пользователем
                    # молча потерялся бы.
                    round_digests.append(f"- asked user: {shape['question']!r}\n- user answered: {answer!r}")
                    payload = {"messages": [HumanMessage(content=f"The user's answer: {answer}")]}
                    attempt += 1
                    continue
                if attempt == max_attempts_effective - 1:
                    break
                round_digests.append(_summarize_round(round_msgs, verdict))
                payload, config, emitted = _start_next_attempt(
                    original_messages, round_digests,
                    f"Your last answer doesn't move the task forward "
                    f"(reason: {verdict['reason']}). If it just handed a "
                    "decision or open question back to the user in plain "
                    "text, call the ask_user tool now with that question "
                    "and the concrete options — a question typed into "
                    "your response doesn't wait for or receive an answer. "
                    "If the task actually told you to decide yourself, "
                    "answer with your own reasoned pick instead of asking.",
                    read_history,
                )
                attempt += 1
                continue
            # Ничего не вызывалось в этом раунде и это не один из случаев
            # выше (прямой ответ без вопроса, либо модель сдалась) — нечего
            # верифицировать, отдаём как есть.
            break

        # Считаем один раз — нужно и для выбора verdict, и для guidance ниже,
        # а не просто bool (нужно имя тула для текста сообщения).
        rejected_retry_tool = _retried_after_rejection(new_tool_msgs)
        diffed_paths = _extract_diffed_paths(new_tool_msgs)
        failed_writes = _failed_write_messages(new_tool_msgs)
        semantic_verdict_used = False  # переопределяется в else-ветке ниже

        if failed_writes:
            verdict = {
                "relevant": False,
                "reason": (
                    f"the `{failed_writes[0].name}` call failed with a tool "
                    "error — nothing was actually written/edited, the task "
                    "isn't done"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif rejected_retry_tool:
            verdict = {
                "relevant": False,
                "reason": (
                    f"the model called `{rejected_retry_tool}` again "
                    "immediately after the user rejected it — the tool "
                    "result explicitly said not to retry unless the user "
                    "asks, and that was ignored"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _wrote_code(new_tool_msgs) and _execution_evidence_shows_failure(round_msgs):
            # Живой прогон: _has_execution_evidence только проверяло, что
            # bash ВЫЗВАН, не что он прошёл — self-heal дважды подряд
            # засчитывал раунд как проверенный, пока bash отвечал
            # "Error (exit 1): ... IndentationError". "kind" здесь читает
            # блок после цикла (см. ниже) — если попытки кончатся именно на
            # этом вердикте, правки хода откатываются автоматически вместо
            # того, чтобы оставить сломанный файл лежать в проекте.
            verdict = {
                "relevant": False,
                "kind": "execution_failure",
                "reason": (
                    "the command run to verify the change (bash) failed "
                    "with an error — the code doesn't actually work yet, "
                    "writing/editing the file is not enough"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _wrote_code(new_tool_msgs) and not _has_execution_evidence(new_tool_msgs):
            verdict = {
                "relevant": False,
                "reason": (
                    "a file was written/edited but no command was actually "
                    "executed (no bash call) — writing/editing a file "
                    "is not verification that it works"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _wrote_code(new_tool_msgs) and _verified_with_syntax_check_only_despite_discoverable_tests(round_msgs):
            verdict = {
                "relevant": False,
                "kind": "syntax_only_verification",
                "reason": (
                    "the only bash calls this round were bare syntax/"
                    "lint checks (e.g. `php -l`, `py_compile`, `tsc "
                    "--noEmit`) — that confirms the file parses, not that "
                    "the change behaves correctly, and a real test file "
                    "surfaced in this same round's tool results was never "
                    "actually run"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif (
            _git_status_reports_changes(new_tool_msgs)
            and not _has_diff_evidence(new_tool_msgs)
        ):
            verdict = {
                "relevant": False,
                "reason": (
                    "git_status reported non-empty staged/unstaged changes, "
                    "but the diff tools called don't cover the full picture "
                    "— either call git_diff('HEAD'), or call BOTH "
                    "git_diff_staged AND git_diff_unstaged (one of them "
                    "alone can miss half the changes if there's content in "
                    "both the staged and unstaged sections)"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _truncated_git_diff(new_tool_msgs):
            verdict = {
                "relevant": False,
                "reason": (
                    "a git_diff/git_diff_staged/git_diff_unstaged result was "
                    "truncated — these tools can't be scoped to a single "
                    "file (only context_lines, which makes output bigger, "
                    "not smaller); use bash with `git diff -- <path>` "
                    "per file instead of retrying the same call"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _has_truncated_output(new_tool_msgs):
            verdict = {
                "relevant": False,
                "reason": (
                    "a tool result was truncated (marked with '...[TRUNCATED') "
                    "before showing the whole output — answering from the "
                    "truncated part alone risks missing the actual content "
                    "of the change; narrow the query (specific file/path, "
                    "smaller context_lines/max_results) or call the tool "
                    "again to see the rest before answering"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _final_answer_ignores_diff(round_final_text, diffed_paths):
            verdict = {
                "relevant": False,
                "reason": (
                    "the diff tools returned real changes to specific files "
                    f"({', '.join(sorted(diffed_paths))}), but the answer "
                    "doesn't mention any of them — it looks like it answered "
                    "from something else (a tangential lookup) instead of "
                    "that diff; rewrite the answer based on the diff content "
                    "already retrieved, don't call more tools"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        elif _described_commits_without_diff(new_tool_msgs, round_final_text):
            verdict = {
                "relevant": False,
                "kind": "commits_described_without_diff",
                "reason": (
                    "the answer describes specific commits by hash, but "
                    "git_log only returned metadata (hash/author/date/"
                    "message) — no git_show or git_diff call ever read what "
                    "those commits actually changed, so the per-commit "
                    "descriptions were written from commit messages alone, "
                    "not real diff content"
                ),
            }
            if DEBUG:
                console.print(f"[dim][MCP-AGENT] Deterministic verdict: {verdict}[/]")
            log_event("verdict", **verdict)
        else:
            # Живой репорт: этот вызов сам по себе может занять минуты на
            # медленной локальной модели (свой промпт из TASK+TOOL RESULTS+
            # ANSWER) — без явного события пользователь видит "ответ уже
            # написан, а крутится неизвестно почему" (см. _VERIFYING_PHRASES
            # в ui/stream.py).
            if on_event:
                await on_event({"type": "verifying_start"})
            verdict = await _semantic_check(
                judge_model, task_text, new_tool_msgs, round_final_text, _called_ask_user(new_tool_msgs)
            )
            if on_event:
                await on_event({"type": "verifying_end"})
            semantic_verdict_used = True
        final_verdict = verdict
        if not verdict["relevant"] and on_event:
            # Живой прогон (post-mortem вопрос "почему он ретраил?"): причина
            # отказа self-heal раньше уходила ТОЛЬКО в console.print под
            # DEBUG=1 — в реальный терминал, без следа в episodic_messages,
            # так что разбор задним числом по БД мог только гадать по
            # паттерну tool-коллов, какой именно вердикт вызвал retry. Этот
            # тип события идёт через ту же on_event-трубу, что и tool_start/
            # tool_end — cli.py уже пишет её в БД под тем же DEBUG=1, так что
            # причина отказа теперь настоящий факт в логе, а не реконструкция.
            await on_event({
                "type": "self_heal_reject",
                "reason": verdict["reason"],
                "kind": verdict.get("kind"),
            })
        if verdict["relevant"]:
            break

        punted_to_user = (
            semantic_verdict_used
            and not _called_ask_user(new_tool_msgs)
            and ("?" in round_final_text or "？" in round_final_text)
        )
        if punted_to_user and self_heal_asks_used < MAX_SELF_HEAL_ASKS:
            # Тот же живой баг, что и в ветке без tool-вызовов выше: модель
            # исследовала (new_tool_msgs непустой), а закончила ход текстовым
            # вопросом вместо ask_user. Ретрай с просьбой "вызови тул как
            # надо" не лечит — сразу открываем настоящий диалог с этим же
            # вопросом.
            self_heal_asks_used += 1
            shape = await _extract_ask_user_shape(judge_model, round_final_text)
            if on_event:
                await on_event({"type": "tool_start", "name": "ask_user", "args": shape})
            answer = await ask_user_question(shape["question"], shape["options"], shape["recommended"])
            if on_event:
                await on_event({"type": "tool_end", "name": "ask_user", "result": answer[:2000]})
            max_attempts_effective += 1
            final_verdict = None  # см. комментарий у первого self-heal выше
            # Тред продолжается как есть (см. комментарий у первого self-heal
            # выше) — но digest всё равно копим на случай компакции позже.
            round_digests.append(_summarize_round(
                round_msgs,
                {"reason": f"punted to user via text instead of ask_user; asked: {shape['question']!r}, answered: {answer!r}"},
            ))
            payload = {"messages": [HumanMessage(content=f"The user's answer: {answer}")]}
            attempt += 1
            continue

        if attempt == max_attempts_effective - 1:
            break

        # Не перестраиваем весь план заново (в отличие от старого
        # orchestrator.py) — create_agent сам решает, какие тулы вызвать
        # дальше, учитывая ВСЮ историю по этому thread_id; мы просто даём
        # ему знать, что предыдущий раунд не подошёл, и просим продолжить.
        # Собираем корректирующую подсказку из применимых частей вместо
        # одного жёстко зашитого текста — иначе сценарий "записан код"
        # получал бы совет про bash даже когда провал был из-за
        # непрочитанного диффа, и наоборот. Условия здесь — те же
        # структурные проверки по tool_messages, что и в выборе verdict
        # выше, а не разбор текста задачи.
        # Живой прогон: когда срабатывал diff-review или truncation-чек ниже,
        # к их подсказке ВСЕГДА добавлялся общий совет "читай СОДЕРЖИМОЕ
        # файлов через read_file" (см. блок ниже) — модель выполняла ОБА
        # совета разом: дозывала нужные diff-тулы И читала весь файл целиком
        # (read_text_file), что вызывало НОВОЕ обрезание уже на последней
        # попытке и выжигало её впустую на нерелевантном исследовании. Общий
        # совет теперь даём, только если ни один конкретный чек не сработал —
        # иначе он противоречит конкретной инструкции.
        guidance_parts = []
        if failed_writes:
            errors = "\n".join(
                f"- {m.name}: {_tool_text(m.content)[:500]}" for m in failed_writes
            )
            guidance_parts.append(
                "Your write_file/edit_file call failed with a tool error — "
                "nothing was written, don't treat it as done or move on. "
                "Read the error below and fix your arguments to match its "
                "exact schema before retrying (edit_file in particular: "
                "`edits` must be an ARRAY of {oldText, newText} objects, "
                f"even for a single change, never a bare object):\n{errors}"
            )
        if rejected_retry_tool:
            guidance_parts.append(
                f"The user already rejected `{rejected_retry_tool}` and the "
                "tool result told you not to retry it unless asked — do not "
                "call it again. Answer with whatever information you "
                "already have, or explain in your answer what you wanted to "
                "do and why, instead of retrying a denied action."
            )
        if semantic_verdict_used and not verdict["relevant"] and not _called_ask_user(new_tool_msgs):
            # Судья (см. _semantic_check/_SEMANTIC_SYSTEM_PROMPT, режим 3)
            # уже мог отбраковать ответ именно за то, что он отфутболил
            # решение/вопрос пользователю текстом вместо ask_user — даём
            # конкретное действие, а не только словесную причину в verdict.
            guidance_parts.append(
                "If your last answer just handed a decision or open question "
                "back to the user in plain text instead of calling ask_user "
                "— don't call more research tools first, you likely already "
                "have enough context. Call the ask_user tool now with that "
                "same question and the concrete options you were weighing "
                "(or, if the task actually asked you to decide yourself, "
                "answer with your own reasoned choice instead of asking)."
            )
        if _wrote_code(new_tool_msgs) and not _has_execution_evidence(new_tool_msgs) and not failed_writes:
            guidance_parts.append(
                "A file was written/edited — call bash and actually run "
                "it (or its tests), don't claim it works without running it."
            )
        elif verdict.get("kind") == "execution_failure":
            bash_errors = "\n".join(
                f"- {_tool_text(m.content)[:500]}" for m in new_tool_msgs if m.name == "bash"
            )
            guidance_parts.append(
                "The command you ran to verify the change FAILED — but the "
                "edit itself from the previous attempt is still there, you "
                "don't need to redo it or re-explore the files you already "
                "found. Do NOT restate the task or start over from scratch; "
                "your only remaining job this round is to get a real "
                "verification result. First check whether the error below is "
                "actually about YOUR code, or about how you ran the check "
                "(wrong interpreter/command, missing tool that's normally "
                "available in this project) — if it's the latter, find the "
                "right way to run it (project launcher, Makefile, README/CI, "
                "existing venv/toolchain) and rerun, don't conclude the code "
                "is broken. If you can't pin down the fix within your "
                "remaining attempts, say so plainly in your final answer "
                "instead of reporting success:\n" + bash_errors
            )
        elif verdict.get("kind") == "syntax_only_verification":
            guidance_parts.append(
                "A syntax/lint check (php -l, py_compile, tsc --noEmit, ...) "
                "only proves the file parses — it does not verify the "
                "changed behavior is correct. This round's own tool results "
                "already named a real test file for this area — run the "
                "project's actual test runner (phpunit, pytest, npm test, go "
                "test, ...) covering the file(s) you changed, and use ITS "
                "pass/fail result as your verification instead of the "
                "syntax check alone."
            )
        if (
            _git_status_reports_changes(new_tool_msgs)
            and not _has_diff_evidence(new_tool_msgs)
        ):
            guidance_parts.append(
                "git_status reported non-empty changes — call git_diff('HEAD'), "
                "or call BOTH git_diff_staged and git_diff_unstaged (one of "
                "the two alone is not enough if changes exist in both "
                "sections) — git_status only names files, it doesn't show "
                "what changed inside them. That's enough to answer — do NOT "
                "also read whole files (read_file/read_text_file) or call "
                "list_directory/directory_tree, the diff already has "
                "everything you need."
            )
        if _truncated_git_diff(new_tool_msgs):
            guidance_parts.append(
                "Part of the previous git_diff/git_diff_staged/git_diff_unstaged "
                "result was truncated — these tools have NO way to scope to a "
                "single file (their only parameter besides the repo is "
                "context_lines, which makes output BIGGER, not smaller). "
                "Retrying with a different context_lines won't help. Instead, "
                "use bash with a plain `git diff -- <path>` (or `git diff "
                "--cached -- <path>`) for ONE file at a time, and repeat per "
                "file if there are several."
            )
        elif _has_truncated_output(new_tool_msgs):
            guidance_parts.append(
                "Part of the previous tool result was truncated (marked "
                "'[TRUNCATED...]') — don't answer as if you saw all of it; "
                "retry the SAME tool with a narrower query (specific file/"
                "path, smaller context_lines/max_results). Don't switch to a "
                "different tool (read_file on the whole file, list_directory, "
                "etc.) — that won't show the truncated part and will waste "
                "your last attempt."
            )
        if _final_answer_ignores_diff(round_final_text, diffed_paths):
            guidance_parts.append(
                "You already have the real diff content from earlier in this "
                f"investigation (changes to {', '.join(sorted(diffed_paths))}) "
                "— your last answer didn't use it and talked about something "
                "else instead. Don't call more tools or go research another "
                "tangent — write the answer directly from that diff content."
            )
        if verdict.get("kind") == "commits_described_without_diff":
            guidance_parts.append(
                "You described specific commits by hash without ever reading "
                "what they actually changed — git_log only gives you the "
                "message, not the code. Call git_show(<hash>) (or "
                "git_diff(target=<hash>)) for each commit you're about to "
                "describe, then rewrite the answer quoting the real diff "
                "content for each one instead of paraphrasing its commit "
                "message."
            )
        if not guidance_parts:
            # Живой прогон (mail-server, 20260707-205626-d3ae9e2c): судья
            # отбраковал ЧИСТО ОПИСАТЕЛЬНЫЙ ответ ("вот как это сейчас
            # работает") без единого предложения/правки — reason был "Ответ
            # не адресует основную проблему из задачи", не "недостаточно
            # информации". Старый фолбэк здесь ("читай СОДЕРЖИМОЕ файлов
            # через read_file") предполагал ТОЛЬКО нехватку данных и активно
            # толкал модель читать ЕЩЁ — следующая попытка честно
            # переслушалась этому совету и заново перечитала ВСЕ те же
            # директории и файлы с нуля (несмотря на digest выше, который
            # уже перечислял их как "explored"), так и не дописав ответ до
            # конца, когда пользователь остановил ход вручную. Фолбэк не
            # должен ОДНОСТОРОННЕ толкать в "читай больше" — самая частая
            # причина именно этого (безальтернативного) вердикта это не
            # нехватка чтения, а то, что ответ описал текущее поведение
            # вместо того, чтобы решить задачу.
            guidance_parts.append(
                "The judge's reason above is the real signal — don't default "
                "to 'gather more information' without reading it. You may "
                "already have everything you need from what's in the digest "
                "above: if so, don't call ANY more research tools — answer "
                "now with a concrete decision, and if the task implies "
                "changing code, propose AND WRITE the actual fix (a "
                "description of how the current code behaves, with no "
                "proposed change, is likely what got this rejected). Only "
                "call more tools if you genuinely lack a specific piece of "
                "information, and even then don't repeat a structural "
                "listing (list_directory/directory_tree) with a different "
                "path — it already showed what exists."
            )

        round_digests.append(_summarize_round(round_msgs, verdict))
        payload, config, emitted = _start_next_attempt(
            original_messages, round_digests,
            f"The previous tool results don't answer the task "
            f"(reason: {verdict['reason']}). " + " ".join(guidance_parts),
            read_history,
        )
        attempt += 1

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
        # result уже НЕ гарантирует "result is None or нет messages" — после
        # digest-ретрая (см. выше) это финальная, ПОСЛЕДНЯЯ попытка из
        # max_attempts_effective, которая тоже упёрлась в лимit; сообщение
        # показываем безусловно, раз мы вообще сюда дошли.
        yield (
            f"⚠️ Не удалось получить ответ за {attempt + 1} попыт{'ку' if attempt == 0 else 'ки'} "
            f"по {RECURSION_LIMIT} шагов каждая — задача требует больше расследования, чем "
            "уместилось в бюджет. Попробуй переформулировать задачу уже, или сузить её."
        )
        return

    if hit_generation_error and (result is None or not result.get("messages")):
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

    final = result["messages"][-1]
    final_text = final.content if isinstance(final, AIMessage) else str(final.content)

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
