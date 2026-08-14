"""
_CompactResearchMiddleware — сжимает всё до ПОСЛЕДНЕГО успешного write/edit-
тула (исследование, и любые более ранние правки) в компактную выжимку
находок — вместо того, чтобы каждый следующий вызов модели в этом же ходе
(верификация, исправление ошибки после неё, финальный ответ) тащил весь
сырой трафик заново.

Живой прогон (mail-server, сессия 20260707-135011-31723e6e): после ask_user
и первой правки в ОДНОМ и том же ходе накопились ещё чтение + вторая правка
+ несколько bash_exec поверх уже большой исследовательской части — общий
объём перевалил за num_ctx=32768, и llama.cpp дважды сделал "context shift"
(см. agent_builder.py:_ChatOllamaWithNumKeep), один раз прямо посреди
генерации финального ответа, оборвав его на полуслове. num_keep защищает
системный промпт при таком переполнении, но само переполнение всё равно
происходит и всё равно режет историю (просто менее катастрофично). Эта
мидлварь не даёт истории вообще так разрастись.

Живой прогон #2 (mail-server, 20260707-163337-11afd268): первая версия этой
мидлвари сжимала ТОЛЬКО до ПЕРВОГО успешного write — весь дальнейший трафик
одного хода (в этом прогоне: сообщения 109-282, повторные правки одного и
того же метода после несколько раз промазанных replace_lines, туда-сюда
верификация) остался несжатым и разросся до 1.7М входных токенов за один
ход. Теперь режем до ПОСЛЕДНЕГО успешного write на момент каждого вызова
модели — по мере того как появляются новые успешные правки, точка среза
сама сдвигается вперёд (новый префикс — новый кэш-ключ, старый дайджест
просто становится неактуальным и не используется).

Точность самой правки не страдает: сжатие происходит только ПОСЛЕ того, как
write-тул уже вызван и его результат подтверждён успешным — сама правка была
сгенерирована на ПОЛНОМ, несжатом контексте. Сжимается только то, что уже
"отработало своё" и дальше моделью в основном не нужно текстом, а не то, что
ей ещё только предстоит использовать. Неудачный/висящий write НЕ двигает
точку среза — тот обмен (и всё после него) остаётся как есть, чтобы модель
видела полную картину своей текущей, ещё не решённой проблемы.

Не трогает реальное состояние графа/чекпоинтер — тот же принцип, что и у
_DedupeToolResultsMiddleware в message_utils.py: request.override(messages=...)
меняет только то, что физически уйдёт В ЭТОТ КОНКРЕТНЫЙ запрос к Ollama.

Живой прогон #3 (mail-server, 20260708-2038): _compact_periodic_research (ниже,
до-write разведка) один раз сжала в дайджест тот самый read_file, который
только что вернул полное содержимое MailboxConvertTask.php. Модель на
следующем шаге больше не видела код и попыталась перечитать файл —
tool_wrappers.py:_dedupe_read_tool не знает о сжатии (свой отдельный
read_history, не сообщение в истории) и ответил "вы уже читали этот файл,
переиспользуйте тот результат" — а переиспользовать было нечего, дайджест
прозы не содержал кода. Модель потратила 3 тул-вызова (read_file,
read_text_file, read_file_range с другим диапазоном — только это случайно
обошло дедуп), гоняясь за собственным же прочитанным файлом. Раздел ниже
теперь никогда не упаковывает результаты read_file/read_text_file/
read_file_range/read_multiple_files/lsp/project_tree — только настоящий
разведочный шум (search_files, list_directory, get_knowledge и т.п.), см.
STICKY_TOOL_NAMES и _group_is_sticky. project_tree добавлен отдельно от
остальных read-тулов: это не содержимое файла, а карта проекта, которая и
дальше нужна модели для ориентации, а сам объём у неё и так ограничен
(max_entries в mcp_agent/servers/code_search_server.py).

Живой прогон #4 (mail-server, 20260708-2109): search_code изначально считался
"шумом" наравне с project_tree/list_directory — но на задаче-расследовании
(поиск бага через прицельные грепы по константам/методам) это ОСНОВНОЙ канал
информации, а не шум: 16 из 38 тул-вызовов за прогон — search_code, и его
результат — это как раз "путь+строка", которые нужны дословно. Конкретный
случай: search_code(handleFinalFailure) вернул точные номера строк (123, 132,
153); через один шаг после упаковки эта информация ушла в прозу дайджеста, и
модель дословно повторила тот же самый search_code, чтобы получить обратно те
же номера строк для последующего read_file_range. search_code/search_symbols/
find_files_by_name — тоже sticky теперь: их вывод и так компактный (см. лимиты
MAX_RESULTS/max_results в code_search_server.py), терять из компрессии
там особо нечего, а грепами держится расследование целиком.

Живой прогон #5 (XOR-на-Go задача, 20260810): триггер компакта раньше был
"в истории есть УСПЕШНЫЙ write" (см. _last_write_result_index), а не "история
реально приближается к num_ctx" — на этом прогоне сработал уже на 3-message
префиксе (один mkdir), дав дайджест в 108 символов при контексте в 65536
токенов: ноль пользы (сэкономлена пара сотен символов) за не-нулевой риск
(любой пересказ — это шанс потерять деталь, см. live-run #3 выше). Теперь
awrap_model_call сначала проверяет _needs_compaction (грубая chars//4 оценка
всей истории против OLLAMA_NUM_CTX * COMPACT_HISTORY_CTX_RATIO, см.
model_config.py) и пропускает ОБА пути (write-triggered и periodic) целиком,
если контекст ещё далеко не заполнен. settings.compact_history_enabled — общий
выключатель поверх этого (/settings), на случай если даже гейтед-по-размеру
компакт где-то потеряет для конкретной задачи что-то важное.

Живой прогон #6 (XOR-на-Go задача, 20260812): _needs_compaction сравнивал
объём истории с OLLAMA_NUM_CTX*COMPACT_HISTORY_CTX_RATIO — той же
захардкоженной-на-импорте константой (65536 по умолчанию), а не с
settings.get("num_ctx"), реально переданным в ChatOllama для этой чат/
judge-модели (agent_builder.py:_build_chat_model). num_ctx на этом прогоне
был занижен до 16384 (см. settings.py, тестировали expert_streaming_enabled)
— порог компакта получился 32768, вдвое больше реального окна модели.
Компакт не сработал НИ РАЗУ за весь ~14-минутный ход (3393 сообщения),
модель потеряла ориентацию в файле и один раз выдумала expected_first_hash,
которого не было ни в одном её собственном read_file_range. _needs_
compaction и num_ctx в options _summarize_research теперь читают
settings.get("num_ctx") — тот же параметр, что реально уходит в Ollama для
этого вызова, а не отдельную, независимо выставленную константу.

Живой прогон #7 (glm-4.7-flash/expert-streaming, 20260814): реальный 400 —
"request (30739 tokens) exceeds the available context size (30208
tokens)" — при том что _needs_compaction's собственная оценка утверждала,
что история далеко не дотягивает до порога. Причина: `messages`, которые
эта функция получала (request.messages) — это LangChain ModelRequest,
который явно ИСКЛЮЧАЕТ системное сообщение ("# excluding system message"),
а схемы тулов (request.tools) вообще никогда не входили в оценку.
Analyzer/Planner с их ~2.7k-токенным системным промптом (prompts.py) и
~20+ забинженными тулами (у каждого реальные description+parameters)
реально стоили несколько тысяч токенов, которые эта проверка попросту
никогда не видела — доля от num_ctx (0.5) держалась только потому, что
щедрый запас случайно перекрывал эту неучтённую дыру, пока список тулов не
разросся достаточно, чтобы перекрыть и его. _needs_compaction теперь
принимает system_message/tools и учитывает их тем же chars//4; порог
сменился с доли от num_ctx на num_ctx минус фиксированный резерв под
генерацию (OLLAMA_NUM_PREDICT) и под неточность самой оценки — см. её
собственный докстринг.
"""
import hashlib
import json

import settings
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from mcp_agent.debug_log import log_event
from mcp_agent.message_utils import _tool_text
from mcp_agent.model_config import DEBUG, OLLAMA_NUM_PREDICT
from mcp_agent.self_heal import _failed_write_messages
from ui.console import console
from utils.parsing import parse_json_loose

_WRITE_TOOL_NAMES = ("write_file", "edit_file", "replace_lines", "insert_lines", "copy_lines")

# Pre-write research has no "last successful write" to cut at (see
# _last_write_result_index below) — left alone it grows unbounded for the
# whole exploration phase: directory listings, dead-end greps, file reads
# that turn out irrelevant. _compact_periodic_research freezes exploration
# into fixed-size chunks of this many tool calls each, once a chunk is
# complete its digest is cached by that chunk's OWN exact content forever —
# unlike the write-triggered path above, which re-summarizes the whole
# (growing) prefix from scratch on every new write, a frozen chunk here is
# never re-fed to the summarizer once packed, so a long exploration doesn't
# pay a bigger LLM call each time it runs and an early digest can't drift
# between calls. Only counts tool calls OUTSIDE of STICKY_TOOL_NAMES (see
# below) — those never get folded into a chunk in the first place.
PERIODIC_CHUNK_TOOL_CALLS = 8

# Tool calls whose result the model may still need verbatim later: actual
# file/code content (read_file/read_text_file/read_file_range/
# read_multiple_files/lsp), the project's own map (project_tree, kept small
# by max_entries in code_search_server.py), or exact grep/symbol hits
# (search_code/search_symbols/find_files_by_name — "path:line: match", the
# precise locations an investigation is actually built on, not noise; see
# live-run-#4 in the module docstring). Orientation noise that's genuinely
# safe to paraphrase away — search_files, list_directory, get_knowledge,
# dead-end greps that found nothing — is NOT in this set. A group
# containing any of these names is "sticky" — _compact_periodic_research
# always keeps it raw, never folds it into a digest. See the live-run-#3
# note in the module docstring: folding an already-returned read_file
# result into prose left the model unable to see that file's actual code on
# the next turn, and tool_wrappers.py:_dedupe_read_tool (a separate
# mechanism, with its own history, unaware of compaction) then refused to
# let it re-read the same path+params, telling it to "reuse that earlier
# result" — which no longer existed anywhere in what the model could see.
STICKY_TOOL_NAMES = {
    "read_file", "read_text_file", "read_file_range", "read_multiple_files",
    "lsp", "project_tree", "search_code", "search_symbols", "find_files_by_name",
}

_COMPACT_SYSTEM_PROMPT = (
    "You compress a coding agent's own history in the current task into a "
    "short, concrete digest for its OWN later use — not a message to the "
    "user. Input is the ORIGINAL TASK plus a transcript of tool calls: "
    "read-only investigation (directory listings, file reads, searches) "
    "and, often, one or more EARLIER edits that already happened and don't "
    "need to happen again. Write what a developer would need to remember "
    "to keep working effectively: the relevant file(s) and what's actually "
    "in them, the specific cause/convention that justified any edit, and — "
    "whenever the transcript shows an edit — EXACTLY what was already "
    "changed (file path plus the substance of the change, e.g. 'added "
    "STATUS_CANCELLED=50 and its $statuses entry to "
    "MailboxConvertationPersister.php') so it is never redone, "
    "contradicted, or second-guessed later. Keep it to a few dense, "
    "concrete sentences (file paths, line numbers, function/class/constant "
    "names), not a transcript replay and not vague generalities. Do NOT "
    "describe whatever edit or verification comes AFTER this transcript "
    "(that's kept separately, verbatim, right after this digest) — only "
    "summarize what's already in the transcript you were given. Drop dead "
    "ends and files that turned out irrelevant entirely; don't pad the "
    "digest to mention everything that was looked at.\n"
    'Respond with ONLY this JSON: {"findings": "..."}'
)

# Prefix digests are cached per exact prefix content (see _prefix_cache_key)
# so a long attempt with several model turns after the compaction point
# doesn't re-run the summarization call on every single one of them — only
# once per distinct prefix. _MISSING (vs. None) distinguishes "not attempted
# yet" from "attempted and genuinely produced nothing" so a real digest of
# "" is never mistaken for a cache miss.
_MISSING = object()


def _task_frame_len(messages: list) -> int:
    """Число ВЕДУЩИХ HumanMessage подряд в начале истории — неприкосновенная
    рамка задачи, а не история для сжатия. Раньше неприкосновенным считался
    только messages[0] — верно для старой архитектуры с одним seed-
    сообщением, но в пайплайне Coder/Verifier получает ДВА ведущих
    HumanMessage подряд (mcp_agent/pipeline.py:_seed_stage_payload —
    исходный диалог + дайджест с Планом Planner'а, приклеенный последним
    HumanMessage перед тем, как модель начинает действовать). Второй молча
    резался при сжатии и не попадал ни в task_text, ни в transcript (оба
    ниже раньше видели только AIMessage/ToolMessage) — План вместе с
    точными путями к файлам исчезал из контекста после первого же успешного
    write (живой прогон 67fac007f4c34ba88d899fbb175f7313: Coder после этого
    перепутал путь к контроллеру и потратил 6 тулов на его повторный поиск)."""
    n = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            n += 1
        else:
            break
    return max(n, 1)


def _render_transcript(messages: list) -> str:
    calls_by_id = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc.get("id"):
                    calls_by_id[tc["id"]] = tc
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            lines.append(f"[context] {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"[agent] {m.content}")
        elif isinstance(m, ToolMessage):
            tc = calls_by_id.get(m.tool_call_id, {})
            args = tc.get("args") or {}
            lines.append(f"[{m.name}{args}] -> {_tool_text(m.content)}")
    return "\n".join(lines)


def _prefix_cache_key(prefix: list) -> str:
    raw = "|".join(f"{type(m).__name__}:{m.content}" for m in prefix)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def _estimate_message_tokens(m) -> int:
    """chars // 4 — same rough heuristic as _SYSTEM_PROMPT_TOKENS_ESTIMATE
    in prompts.py, not a real tokenizer count (none is available here
    without calling Ollama). Good enough for a "are we anywhere near
    num_ctx" gate — see _needs_compaction below — not for an exact budget.
    Counts tool_calls' args too (AIMessage.content is often empty/short for
    a message that's mostly a big write_file call — undercounting THAT is
    exactly the case this gate exists to catch)."""
    n = len(str(getattr(m, "content", "") or ""))
    if isinstance(m, AIMessage):
        for tc in (m.tool_calls or []):
            try:
                n += len(json.dumps(tc.get("args") or {}, ensure_ascii=False))
            except TypeError:
                n += len(str(tc.get("args") or ""))
    return n // 4


def _estimate_tool_tokens(tool) -> int:
    """Same chars//4 heuristic as _estimate_message_tokens, for one bound
    tool's OWN schema (name + description + parameters) — this is real
    prompt content too (every bound tool's definition gets serialized into
    the request the model actually sees), just never arriving as a
    `messages` entry. `tool` here is whatever LangChain's ModelRequest.tools
    holds — usually BaseTool instances, but accept a plain dict too (an
    already-converted OpenAI-style tool spec) since middleware ordering
    isn't guaranteed to leave that conversion for later."""
    if isinstance(tool, dict):
        try:
            return len(json.dumps(tool, ensure_ascii=False)) // 4
        except TypeError:
            return 50
    n = len(getattr(tool, "name", "") or "") + len(getattr(tool, "description", "") or "")
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            n += len(json.dumps(schema.model_json_schema(), ensure_ascii=False))
        except Exception:
            pass
    return n // 4


def _needs_compaction(messages: list, system_message=None, tools=None) -> bool:
    """Real-context-size gate — replaced the old unconditional "there's a
    successful write in history" trigger (live-run 20260810: a 3-message,
    ~1-2k-token prefix got compacted into a 108-char digest for zero real
    benefit, purely because a write had happened, not because context was
    anywhere near full).

    Threshold is settings.get("num_ctx") — the value actually handed to
    ChatOllama for this session's chat/judge model (agent_builder.py:
    _build_chat_model) — NOT model_config.OLLAMA_NUM_CTX. That constant is
    read once at import and frozen; num_ctx is a runtime-editable setting
    (see its own docstring in settings.py) that can sit well below it. Live
    run (20260812, XOR-in-Go task): num_ctx had been lowered to 16384
    (leftover from an expert_streaming_enabled test) while this gate still
    compared against OLLAMA_NUM_CTX=65536 — the threshold ended up at
    32768, TWICE the model's actual window, so compaction never fired for
    the whole ~14-minute run and the model eventually lost track of the
    file (hallucinated a expected_first_hash that appears nowhere in its
    own read history). Compaction's whole job is protecting THIS call's
    real context, so it must measure against the same number that call
    actually uses.

    system_message/tools — added 2026-08-14 after a live 400 (glm-4.7-
    flash/expert-streaming): "request (30739 tokens) exceeds the available
    context size (30208 tokens)" fired even though this gate's OWN estimate
    said the history was nowhere near its threshold. Root cause:
    LangChain's ModelRequest.messages explicitly EXCLUDES the system
    message ("messages: list[AnyMessage]  # excluding system message", see
    langchain.agents.middleware.types.ModelRequest) and never included tool
    schemas either — a role with ~2.7k system-prompt tokens (Analyzer, see
    prompts.py) and ~20+ bound tools (each with a real description +
    parameter schema) was contributing several thousand real prompt tokens
    this gate had literally never counted, regardless of how far below
    threshold the conversation history itself looked.

    Threshold changed from a ratio of num_ctx (COMPACT_HISTORY_CTX_RATIO,
    0.5) to num_ctx minus a fixed reserve for what THIS call's own
    generation still needs to fit in the same window (OLLAMA_NUM_PREDICT,
    4096 — the largest num_predict any role here actually requests, see
    agent_builder.py) plus a flat safety margin for this whole estimate
    being chars//4, not a real tokenizer count. A ratio made sense when the
    gate only ever saw a fraction of the real prompt anyway (extra headroom
    was accidentally absorbing that unmeasured gap); now that system+tools
    are counted for real, the actual constraint is simpler and more literal:
    leave enough room for this call's own answer, not "leave half the
    window free" regardless of how large or small num_ctx is."""
    total = sum(_estimate_message_tokens(m) for m in messages)
    if system_message is not None:
        total += _estimate_message_tokens(system_message)
    if tools:
        total += sum(_estimate_tool_tokens(t) for t in tools)
    # +3000 (not a smaller round number) on top of the generation reserve —
    # live measurement (2026-08-14, same incident): investigator_tools()'s
    # ~27 tools alone cost ~3085 estimated tokens from name+description
    # text ALONE, before even adding each tool's own parameters JSON schema
    # (which _estimate_tool_tokens above does count, so the real number is
    # higher still) — chars//4 has real, demonstrated slack in exactly this
    # direction (code/JSON-shaped content tokenizes less efficiently per
    # character than the prose this heuristic was tuned against), not just
    # theoretical rounding error.
    reserve = OLLAMA_NUM_PREDICT + 3000
    return total > settings.get("num_ctx") - reserve


def _group_turns(messages: list) -> list[list]:
    """Groups each AIMessage-with-tool_calls together with the ToolMessages
    answering it into one unit; anything else (a lone AIMessage with no
    tool_calls, a stray HumanMessage) is its own one-message unit. Chunking
    in _compact_periodic_research operates on these units, never splitting
    an AIMessage from its own ToolMessages across a chunk/digest boundary
    the way a plain per-ToolMessage count could (a single AIMessage can
    carry several parallel tool_calls)."""
    groups: list[list] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, AIMessage) and m.tool_calls:
            ids = {tc["id"] for tc in m.tool_calls if tc.get("id")}
            group = [m]
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage) and messages[j].tool_call_id in ids:
                group.append(messages[j])
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([m])
            i += 1
    return groups


def _group_tool_calls(group: list) -> list[dict]:
    return [tc for m in group if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]


def _group_is_sticky(group: list) -> bool:
    """True if this turn's result is content the model may still need
    verbatim later — see STICKY_TOOL_NAMES."""
    return any(tc.get("name") in STICKY_TOOL_NAMES for tc in _group_tool_calls(group))


def _last_write_result_index(messages: list) -> int | None:
    """Index of the AIMessage whose tool_calls include the MOST RECENT
    SUCCESSFULLY completed write/edit tool — that message plus everything
    after it is kept verbatim (see module docstring); everything before it
    is what gets compacted. Scans the WHOLE list (not just up to the first
    match) so the cut point keeps moving forward as more edits succeed
    later in a long round, instead of freezing at the first one and
    leaving everything after it — often most of a long debugging round —
    to grow unbounded (see module docstring, live run #2).

    A write that's still pending (its ToolMessage hasn't arrived yet) or
    FAILED does not move the cut point — that exchange, and everything
    after it, stays verbatim so the model sees exactly what it just tried
    and why it failed, without losing that thread to a later, unrelated
    successful edit elsewhere in the file/round."""
    last_ok = None
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage):
            continue
        write_call_ids = {tc["id"] for tc in (m.tool_calls or []) if tc.get("name") in _WRITE_TOOL_NAMES}
        if not write_call_ids:
            continue
        results = [
            t for t in messages[i + 1:]
            if isinstance(t, ToolMessage) and t.tool_call_id in write_call_ids
        ]
        # .status alone misses replace_lines/insert_lines/copy_lines
        # (fs_extra_server.py) — they never raise a protocol-level MCP
        # error on a semantic failure (stale expected_first_line, bad
        # range, ...), they just return a normal "Error: ..." string, so
        # .status stays "success" even when nothing was written. Reuses
        # the same check as coder_verdict (self_heal.py:_failed_write_
        # messages) so "counts as a successful write" means the same
        # thing everywhere — see that function's docstring for the live
        # run this fixes (a failed write kept getting treated as done,
        # erasing the exact mismatch the model needed to see to retry
        # correctly, so it just resent the identical broken call forever).
        if results and not _failed_write_messages(results):
            last_ok = i
    return last_ok


async def _summarize_research(judge_model, prefix: list) -> str:
    task_len = _task_frame_len(prefix) if prefix else 0
    task_text = "\n\n".join(str(m.content) for m in prefix[:task_len])
    transcript = _render_transcript(prefix[task_len:])
    prompt = f"ORIGINAL TASK:\n{task_text}\n\nTRANSCRIPT SO FAR:\n{transcript}"
    try:
        # Explicit options (not JUDGE_NUM_PREDICT=200, tuned for a one-line
        # relevant/reason verdict elsewhere) — a real findings digest needs
        # more room than a binary verdict. num_keep gets re-injected
        # automatically by _ChatOllamaWithNumKeep._chat_params regardless of
        # what's passed here (see agent_builder.py), so it doesn't need to
        # be repeated in this dict. num_ctx here must match settings.get(
        # "num_ctx") — judge_model itself was already built with that value
        # (agent_builder.py:_build_chat_model); passing the stale
        # model_config.OLLAMA_NUM_CTX constant here would silently ask
        # Ollama to re-negotiate a different context size than the one the
        # rest of this session's judge-model calls use, for this one call.
        #
        # options={} is Ollama-specific (ChatOllama._chat_params merges it
        # into params["options"]) — judge_model is the same _build_chat_model
        # fork as the main model (agent_builder.py), so with settings.
        # expert_streaming_enabled it's a ChatOpenAI, not a ChatOllama. Live
        # bug (2026-08-14, glm-4.7-flash/expert-streaming): options={...} on
        # ChatOpenAI.ainvoke fell straight through to AsyncCompletions.
        # create(), which doesn't know that param — "unexpected keyword
        # argument 'options'" (TypeError, caught below, so compaction just
        # silently never worked rather than crashing the turn) — same class
        # of bug already fixed in self_heal.py's _extract_ask_user_shape,
        # same fix here. ChatOpenAI uses max_tokens directly (not via
        # options{}) and has no per-call num_ctx equivalent — expert-
        # streaming's context is fixed at process start (-c, see
        # expert_streaming.py), so max_tokens alone is enough there.
        extra_kwargs = (
            {"options": {"num_ctx": settings.get("num_ctx"), "num_predict": 600}}
            if isinstance(judge_model, ChatOllama)
            else {"max_tokens": 600}
        )
        resp = await judge_model.ainvoke(
            [
                {"role": "system", "content": _COMPACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **extra_kwargs,
        )
        data = parse_json_loose(resp.content) or {}
        findings = str(data.get("findings") or "").strip()
        if not findings:
            raise ValueError("empty findings")
        return findings
    except Exception as e:
        # fail open — тот же принцип, что в self_heal.py: сломанное сжатие
        # не должно ронять ход, просто в этот раз ход останется несжатым.
        if DEBUG:
            console.print(f"[dim][MCP-AGENT] research compaction failed: {e}[/]")
        log_event("research_compaction_failed", error=str(e))
        return ""


class _CompactResearchMiddleware(AgentMiddleware):
    def __init__(self, judge_model):
        self._judge_model = judge_model
        self._cache: dict[str, str] = {}
        self._periodic_cache: dict[str, str] = {}

    def clear_cache(self) -> None:
        """Called once per stream_chat turn (see agent.py), same convention
        as read_history.clear() — both caches are keyed by exact message
        content, so a stale entry from an earlier, unrelated task would
        just be dead weight rather than wrong, but there's no reason to
        let it accumulate for the lifetime of a long-running process."""
        self._cache.clear()
        self._periodic_cache.clear()

    async def _compact_periodic_research(self, request, handler, messages):
        """No write has happened yet, so there's no _last_write_result_index
        anchor for the write-triggered path below to cut at. Walks the
        pre-write research turn by turn (_group_turns — never splitting an
        AIMessage from its own ToolMessages): turns whose result the model
        may still need verbatim (_group_is_sticky, see STICKY_TOOL_NAMES)
        are always kept raw, in place; everything else (search_code,
        dead-end greps, ...) accumulates into a pending buffer that gets
        frozen into its own digest once it reaches
        PERIODIC_CHUNK_TOOL_CALLS tool calls — cached by that buffer's own
        exact content, so once packed it's never re-summarized. Whatever's
        left in the pending buffer below threshold at the end stays raw, as
        the most recent activity."""
        task_len = _task_frame_len(messages)
        task_frame = messages[:task_len]
        groups = _group_turns(messages[task_len:])

        middle: list = []
        pending: list = []
        pending_calls = 0
        digest_count = 0
        chunk_index = 0

        for group in groups:
            if _group_is_sticky(group):
                middle.extend(group)
                continue
            pending.extend(group)
            pending_calls += len(_group_tool_calls(group))
            if pending_calls < PERIODIC_CHUNK_TOOL_CALLS:
                continue

            chunk_index += 1
            cache_key = _prefix_cache_key(pending)
            digest = self._periodic_cache.get(cache_key, _MISSING)
            if digest is _MISSING:
                digest = await _summarize_research(self._judge_model, task_frame + pending)
                self._periodic_cache[cache_key] = digest
            if digest:
                digest_count += 1
                middle.append(HumanMessage(content=(
                    f"(Earlier read-only investigation, part {chunk_index} — "
                    "summarized below; nothing was written yet at this "
                    "point, so the repo itself is unchanged. File/code "
                    "content you already read is kept verbatim elsewhere "
                    "in this history, not summarized away — re-read only "
                    "if you need something else.)\n\n" + digest
                )))
            else:
                # Summarization failed for this chunk (fail-open, same as
                # _summarize_research's own except-clause) — keep it raw
                # rather than silently dropping it.
                middle.extend(pending)
            pending, pending_calls = [], 0

        middle.extend(pending)  # below threshold — stays raw, most recent

        if digest_count == 0:
            return await handler(request)
        if DEBUG:
            console.print(
                f"[dim][MCP-AGENT] periodic research compaction: "
                f"{chunk_index} chunk(s) attempted -> {digest_count} "
                "digest(s)[/]"
            )
        log_event(
            "periodic_research_compacted", chunks=chunk_index,
            digest_count=digest_count,
        )
        return await handler(request.override(messages=task_frame + middle))

    async def awrap_model_call(self, request, handler):
        if not settings.get("compact_history_enabled"):
            return await handler(request)
        messages = request.messages
        if not _needs_compaction(messages, request.system_message, request.tools):
            # Below threshold — nothing to gain from summarizing yet, and
            # every digest is a chance to lose something the model still
            # needs verbatim (see module docstring, live runs #1/#3). Skip
            # BOTH paths below, not just the write-triggered one — a short
            # pre-write exploration doesn't need periodic chunking either.
            return await handler(request)
        cut = _last_write_result_index(messages)
        if cut is None or cut < 1:
            return await self._compact_periodic_research(request, handler, messages)
        prefix, rest = messages[:cut], messages[cut:]
        task_len = _task_frame_len(prefix)
        if len(prefix) <= task_len + 1:
            # Barely anything happened before this write — not worth an
            # extra LLM call and a cache entry to save a couple of messages.
            return await handler(request)

        cache_key = _prefix_cache_key(prefix)
        digest = self._cache.get(cache_key, _MISSING)
        if digest is _MISSING:
            digest = await _summarize_research(self._judge_model, prefix)
            self._cache[cache_key] = digest
        if not digest:
            return await handler(request)
        if DEBUG:
            console.print(
                f"[dim][MCP-AGENT] compacted history: {len(prefix)} messages "
                f"-> digest ({len(digest)} chars) + {len(rest)} kept verbatim[/]"
            )
        log_event(
            "history_compacted", prefix_messages=len(prefix),
            digest_chars=len(digest), kept_verbatim=len(rest),
        )

        compacted = [
            *prefix[:_task_frame_len(prefix)],
            HumanMessage(content=(
                "(Everything you did up to and including your last "
                "successful edit is summarized below instead of replayed "
                "in full — the underlying files/repo state already reflect "
                "it; re-read something if you need its exact current "
                "content. What follows this digest, if anything, is kept "
                "verbatim.)\n\n" + digest
            )),
        ]
        return await handler(request.override(messages=compacted + rest))


# Duplicated from mcp_agent/pipeline.py:_WRITE_TOOL_PATH_KEY on purpose, not
# imported — pipeline.py already imports agent_builder.py (which imports
# this module) to build role agents, so importing pipeline.py FROM here
# would be circular. Keep the two in sync by hand if a new write tool with
# its own path-arg name is ever added (currently: copy_lines uses
# dest_path, everything else uses path).
_WRITE_TOOL_PATH_KEY = {
    "write_file": "path", "edit_file": "path", "replace_lines": "path",
    "insert_lines": "path", "copy_lines": "dest_path",
}

# Tools whose whole job is "hand back this path's current content" —
# exactly what a later successful write to that SAME path invalidates. NOT
# search_code/search_symbols/find_files_by_name/lsp: those return short
# match/line-number snippets, not a file dump, so the size win from marking
# them stale is small, while their line numbers becoming stale after an
# edit is arguably still useful context ("here's where it USED to be").
_READ_TOOL_PATH_KEY = {"read_file": "path", "read_text_file": "path", "read_file_range": "path"}
_READ_MULTI_TOOL_NAME = "read_multiple_files"  # args: {"paths": [...]}


def _stale_read_marker(paths: list[str]) -> str:
    where = ", ".join(paths)
    return (
        f"(stale — {where} {'was' if len(paths) == 1 else 'were'} "
        "successfully rewritten LATER in this same conversation; this "
        "earlier read no longer reflects the file's actual current "
        "content. Don't reason from what was shown here — re-read the "
        "path again if you need its current text.)"
    )


class _DropStaleReadsMiddleware(AgentMiddleware):
    """Mechanical (no LLM call, unlike _CompactResearchMiddleware above) —
    replaces the CONTENT of a read_file/read_text_file/read_file_range/
    read_multiple_files ToolMessage with a short marker, IF a later
    successful write in the same conversation touched that same path. The
    old content is dropped outright, not summarized into prose — there is
    nothing to paraphrase, it's simply wrong now (the file has since
    changed), so keeping it verbatim only risks the model reasoning from
    stale line numbers/content instead of re-reading.

    Runs BEFORE _CompactResearchMiddleware in the middleware list (see
    agent_builder.py) — a stale read replaced by a one-line marker here
    is that much less raw content for the compaction pass below to have
    to summarize at all.

    Only touches request.messages for THIS model call, same non-invasive
    pattern as _CompactResearchMiddleware/_DedupeToolResultsMiddleware
    (message_utils.py) — the real graph state/checkpointer is untouched."""

    async def awrap_model_call(self, request, handler):
        messages = request.messages

        last_write_index: dict[str, int] = {}
        for i, m in enumerate(messages):
            if not isinstance(m, AIMessage):
                continue
            for tc in (m.tool_calls or []):
                path_key = _WRITE_TOOL_PATH_KEY.get(tc.get("name"))
                if not path_key:
                    continue
                path = (tc.get("args") or {}).get(path_key)
                if not path:
                    continue
                result = next(
                    (t for t in messages[i + 1:]
                     if isinstance(t, ToolMessage) and t.tool_call_id == tc.get("id")),
                    None,
                )
                if result is None or _failed_write_messages([result]):
                    continue  # pending or failed write — doesn't invalidate anything
                last_write_index[path] = i  # keep the LATEST successful write per path

        if not last_write_index:
            return await handler(request)

        read_paths_by_call_id: dict[str, list[str]] = {}
        for m in messages:
            if not isinstance(m, AIMessage):
                continue
            for tc in (m.tool_calls or []):
                name = tc.get("name")
                args = tc.get("args") or {}
                if name in _READ_TOOL_PATH_KEY:
                    p = args.get(_READ_TOOL_PATH_KEY[name])
                    if p:
                        read_paths_by_call_id[tc["id"]] = [p]
                elif name == _READ_MULTI_TOOL_NAME:
                    ps = [p for p in (args.get("paths") or []) if p]
                    if ps:
                        read_paths_by_call_id[tc["id"]] = ps

        new_messages = list(messages)
        dropped = 0
        for i, m in enumerate(messages):
            if not isinstance(m, ToolMessage):
                continue
            paths = read_paths_by_call_id.get(m.tool_call_id)
            if not paths:
                continue
            stale = [p for p in paths if i < last_write_index.get(p, -1)]
            if not stale:
                continue
            new_messages[i] = ToolMessage(
                content=_stale_read_marker(stale), name=m.name,
                tool_call_id=m.tool_call_id, status=m.status,
            )
            dropped += 1

        if not dropped:
            return await handler(request)
        if DEBUG:
            console.print(f"[dim][MCP-AGENT] dropped {dropped} stale read(s), superseded by a later write[/]")
        log_event("stale_reads_dropped", count=dropped)
        return await handler(request.override(messages=new_messages))
