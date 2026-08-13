"""
Self-heal цикл stream_chat (mcp_agent/agent.py): всё, что нужно, чтобы
решить, годится ли раунд ответа модели, и что делать, если нет.

- Детерминированные verdict-проверки (_wrote_code, _has_execution_evidence,
  _execution_evidence_shows_failure, _failed_write_messages,
  _git_status_reports_changes, _has_diff_evidence, _has_truncated_output,
  _truncated_git_diff, _extract_diffed_paths, _written_paths,
  _final_answer_ignores_diff, _retried_after_rejection, _called_ask_user) —
  структурные проверки по tool_messages, без обращения к LLM, отлавливающие
  типичные живые баги раньше, чем до них дойдёт судья.
- LLM-судья (_semantic_check, _extract_ask_user_shape) — когда
  детерминированных проверок недостаточно, семантическую оценку делает
  отдельная (та же) модель.
- Восстановление утёкшей tool-call разметки (_leaked_tool_call_syntax,
  _parse_leaked_tool_calls, _execute_leaked_tool_call) — когда модель
  сгенерировала вызов тула как обычный текст вместо structured tool_calls.

Ничего из этого не имеет собственного состояния между вызовами — все
функции принимают tool_messages/round_msgs текущего раунда и возвращают
готовый вердикт/факт, stream_chat сам решает, что делать дальше.
"""
import json
import os
import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from ui.console import console
from utils.parsing import parse_json_loose
from mcp_agent.ask_user_tool import _action_and_detail
from mcp_agent.config import TOOLS_REQUIRING_APPROVAL
from mcp_agent.debug_log import log_event
from mcp_agent.message_utils import _tool_text
from mcp_agent.model_config import DEBUG, OLLAMA_NUM_CTX
from tools.confirm import ask_permission

_SEMANTIC_SYSTEM_PROMPT = (
    "You are a strict, deterministic verifier. Judge whether the ANSWER, "
    "backed by the tool results, actually addresses the task — not whether "
    "the tools ran successfully, and not just whether they're on-topic. "
    "Three different failure modes, don't conflate them: (1) for "
    "fact-finding tasks (summarize a diff, find a bug, report what "
    "changed), the tool results themselves must contain the evidence — a "
    "result can be on-topic and still insufficient if it's incomplete or "
    "cuts off mid-way. (2) for judgment/opinion/recommendation tasks (which "
    "option is better, what would you choose, decide for me), the tool "
    "results are just supporting context — they will NEVER themselves "
    "contain a recommendation, only the answer's own reasoning does. There, "
    "judge whether the ANSWER makes an actual reasoned call informed by the "
    "context, not whether the raw tool output states a conclusion. (3) "
    "regardless of task type: if the answer hands a decision, choice, or "
    "open question back to the user in plain text (asking them to pick / "
    "confirm / decide / specify something) instead of actually calling the "
    "ask_user tool, that is NEVER relevant, no matter how reasonable the "
    "question itself sounds — text in an answer cannot receive a reply, "
    "only the ask_user tool's result can. You'll be told explicitly whether "
    "ask_user was called this round; trust that flag, not guesses about "
    "phrasing (a deferred question doesn't need a literal '?' to count). (4) "
    "if the TASK explicitly says NOT to do something ('не X', 'don't X', "
    "'avoid X', 'stop doing X') and the ANSWER's chosen/recommended/"
    "implemented option does exactly X anyway, that is NEVER relevant, no "
    "matter how well-reasoned the answer otherwise sounds — quote the "
    "contradicting phrase from the task in your reason. A live failure this "
    "caught: task said 'don't mark it FAILED on cancel, do something "
    "smarter', and the answer's own recommended fix was 'mark it FAILED on "
    "cancel' — reasonable-sounding, still exactly the forbidden thing. (5) "
    "if the task names a SPECIFIC mechanism/workflow (e.g. 'status handling "
    "when a mailbox conversion is cancelled'), but the actual change "
    "(check the file(s) touched by write/edit tool results) is to a "
    "DIFFERENT file/class/subsystem that the task's mechanism never calls "
    "into, that is NOT relevant even if it's a real, legitimate bug fix and "
    "even if the answer confidently claims a connection between the two — "
    "a plausible-sounding claim of relevance is not evidence of it; the "
    "tool results (reads/searches of the actual mechanism's files) are "
    "your evidence for whether the connection is real. A live failure this "
    "caught: task was about mailbox-conversion-cancellation FAILED status; "
    "the answer instead fixed an unrelated bug in a shared daemon-task "
    "helper never used by that conversion code, then claimed (falsely) "
    "that this would fix mailbox conversion events. "
    "Respond with ONLY a JSON object, no other text."
)

# Сколько символов каждого tool-результата видит судья. Раньше было 300 —
# на реальном прогоне из-за этого судья дважды подряд забраковал ЯВНО
# релевантные результаты (git_status со списком файлов, git_diff с реальным
# диффом): при 300 символах он просто не видел содержательную часть вывода и
# судил по обрубку. 2000 — достаточно, чтобы не резать типичный
# git_status/git_diff/search_code вывод на самом интересном месте, но всё ещё
# заметно меньше, чем полный вывод, который получает основная модель.
_JUDGE_SNIPPET_CHARS = 2000


# На живом прогоне маленький judge_model (тот же qwen3:8b, что и основная
# модель) один раз пометил "relevant: true" раунд, где была только успешная
# запись файла (write_file) без единого bash_exec — то есть сам факт "файл
# записан" судья спутал с "код проверен, что работает". Ловим это
# ДЕТЕРМИНИРОВАННО, без ещё одного вызова маленькой модели, которая уже
# показала, что путает эти две вещи.
#
# Раньше это правило ещё и требовало, чтобы пользователь дословно попросил
# "проверь"/"убедись" — но это угадывание намерения по словам задачи ломается
# на любой перефразировке и не следует из самого правила: системный промпт
# (шаг 4 воркфлоу) и так уже требует верификации ПОСЛЕ ЛЮБОЙ записи кода, а
# не только когда юзер попросил явно. Убрали угадывание — теперь правило
# завязано только на структуру вызовов этого раунда: код записан → нужна
# bash_exec-верификация, вне зависимости от формулировки задачи.
def _wrote_code(tool_messages: list[ToolMessage]) -> bool:
    # replace_lines (fs_extra_server.py) mutates file content exactly like
    # write_file/edit_file — live run: a round that only called replace_lines
    # (no write_file/edit_file) slipped past this check entirely and fell
    # through to the semantic judge instead of the deterministic "no
    # bash_exec" verdict below. insert_lines belongs here for the same
    # reason (a round whose only edit was a successful insert_lines was
    # missing from this tuple while _failed_write_messages below already
    # covers it — the two must agree on what counts as "code was written").
    return any(m.name in ("write_file", "edit_file", "replace_lines", "insert_lines", "copy_lines") for m in tool_messages)


def _has_successful_write(tool_messages: list[ToolMessage]) -> bool:
    """Same tool set as _wrote_code, but requires an actual non-error
    result. Paired with _failed_write_messages in coder_verdict/
    quick_fix_verdict so an EARLIER failed attempt that the model then
    retried and got right within the SAME round doesn't make a real,
    later success count as "nothing was actually written". Live run
    (some-site, styles.css): two replace_lines calls were rejected by the
    expected_first_line multi-line guard, the model then fixed its
    arguments and a third call actually deleted the target line — but the
    round was still rejected outright because _failed_write_messages saw
    the first two failures and nothing checked for the third call's
    success. Mirrors the "only the LAST result matters" principle already
    applied to bash_exec in _execution_evidence_shows_failure above —
    without it, this fix would just be applying that lesson only halfway."""
    write_names = ("write_file", "edit_file", "replace_lines", "insert_lines", "copy_lines")
    return any(
        m.name in write_names and m.status != "error" and not _tool_text(m.content).lstrip().startswith("Error")
        for m in tool_messages
    )


def _has_execution_evidence(tool_messages: list[ToolMessage]) -> bool:
    return any(m.name == "bash_exec" for m in tool_messages)


def _execution_evidence_shows_failure(round_msgs: list) -> bool:
    """_has_execution_evidence only checks that bash_exec was CALLED, not
    that it succeeded — live run: self-heal saw bash_exec in the round and
    treated that as sufficient verification, twice in a row, while its
    result started with 'Error (exit 1): ... IndentationError' both times.
    The broken file was never caught by the retry logic; the user had to
    stop the process by hand instead. bash_exec (see bash_exec_server.py)
    always prefixes a non-zero exit with 'Error (exit N):'.

    Needs round_msgs (not just tool_messages) — uses _bash_commands to see
    each call's actual command text, needed for the two refinements below.

    Live run (mail-server, 20260707-163337-11afd268): this check, once
    fixed to actually see error text (see _tool_text), became too blunt in
    the OTHER direction — it flagged the whole round as broken and the
    stream_chat auto-revert (see agent.py) threw away an already-fixed,
    already-reverified-clean set of edits, because TWO unrelated things
    earlier in the same (long) round also started with 'Error':
      1. A `php -l` on a file the model had just introduced a real syntax
         error into via a botched replace_lines — genuine evidence, but
         the model noticed it too (via the same php -l), fixed the file,
         and re-ran the EXACT SAME command, which came back clean. The
         earlier failure shouldn't keep counting once its own re-run says
         otherwise — only the LAST result of each distinct command matters.
      2. A repo-wide `find . -exec php -l {} \\;` that hit bash_exec's own
         timeout ('Error: command timeout', no exit code at all) — that's
         a badly-scoped verification command, not evidence the CODE is
         broken; it never even finished running. Only a completed run with
         a real non-zero exit ('Error (exit N):') is evidence about the
         code — a bare 'Error: ...' (timeout, no command, an exception
         raised by bash_exec itself) says nothing either way and must not
         overwrite a real prior result for that same command.

    Live run (2026-08-13): Verifier guessed two nonexistent project
    directories ('cd /home/ifeoktistov/neural_network', ...) before finding
    the real one — both `cd`s failed with 'Error (exit 2): ... cd: can't cd
    to ...'. The correct dir it settled on ran a real, clean check right
    after ('Создание нейронной сети...' printed, exit 0) — but that's a
    DIFFERENT command string (different cd target), so it never overwrote
    the two wrong-dir failures already recorded. `any(...)` stayed True on
    those alone, and the whole round — including a verifier report that
    read as 100% passing — got marked execution_failure and reverted. A `cd`
    that fails because the AGENT guessed a directory that doesn't exist is
    evidence about the agent's own navigation, not about whether the CODE
    works — must not count as execution evidence either way, same as the
    bare-error case above."""
    _NAV_ERROR_MARKERS = ("cd: can't cd to", "cd: no such file or directory")
    last_verdict_by_command: dict[str, bool] = {}
    for cmd, result in _bash_commands(round_msgs):
        text = result.lstrip().lower()
        if text.startswith("error (exit") and any(marker in text for marker in _NAV_ERROR_MARKERS):
            continue
        if text.startswith("error (exit"):
            last_verdict_by_command[cmd] = True
        elif not text.startswith("error"):
            last_verdict_by_command[cmd] = False
        # else: bare "error:" (timeout/no-command/exception) — leaves
        # whatever verdict this exact command already had untouched; it's
        # not evidence either way about this command's target.
    return any(last_verdict_by_command.values())


# Живой прогон: edit_file упал с MCP-ошибкой валидации аргументов
# ("edits": expected array, got object — модель передала один объект правки
# вместо массива [{oldText, newText}]) — langchain_mcp_adapters превращает
# CallToolResult(isError=True) в ToolMessage(status="error"), само сообщение
# при этом остаётся с name="edit_file" в new_tool_msgs. _wrote_code видит
# только ИМЯ вызванного тула, не его исход — упавшая правка засчитывалась
# как "код записан", и модель получала совет "запусти bash_exec, чтобы
# проверить" вместо того, чтобы узнать, что файл вообще не тронут и почему.
#
# replace_lines/insert_lines/copy_lines (fs_extra_server.py) никогда не
# кидают протокольную MCP-ошибку на семантический провал (несовпадение
# expected_first_line/expected_last_line, промах диапазона и т.п.) — они
# просто ВОЗВРАЩАЮТ обычную строку "Error: ...", так что .status у них
# всегда не "error" даже когда файл не тронут вообще. Живой прогон
# (f9557fc89f824e2cac92b51b9181a500): Coder 5 раз подряд отправил
# ПОБАЙТОВО идентичный replace_lines, 5 раз получил "Error: line 59
# doesn't match expected_first_line", ни разу не перечитал файл — потому
# что coder_verdict (stages/coder.py) не видел в этом провал: _wrote_code
# засчитывал сам факт вызова replace_lines по имени, а этот же провал не
# ловился и здесь (проверялся только .status). Раунд отчитывался как
# "правки применены" при нуле реальных изменений на диске.
def _failed_write_messages(tool_messages: list[ToolMessage]) -> list[ToolMessage]:
    return [
        m for m in tool_messages
        if m.name in ("write_file", "edit_file", "replace_lines", "insert_lines", "copy_lines")
        and (m.status == "error" or _tool_text(m.content).lstrip().startswith("Error"))
    ]


# То же самое угадывание по ключевым словам ("правки", "diff", "изменения")
# было и здесь — заменили на структурный чек: смотрим не на текст задачи, а
# на СОДЕРЖИМОЕ реального результата git_status. Если он сообщил о
# непустых staged/unstaged изменениях, а после этого в раунде не было ни
# одного diff-вызова — это неполно, независимо от того, как была
# сформулирована задача (даже если git_status был вызван мимоходом, а не
# по прямой просьбе "покажи правки").
#
# Живой прогон, из-за которого появилось это правило: judge_model пометил
# "relevant: true" (с пустой reason) раунд, где агент вызвал только
# git_status (список ИМЁН изменившихся файлов) и read_file на usage.json —
# ни разу не вызвал ни один git_diff*-тул, то есть не увидел ни строчки
# реального содержимого правок в agent.py, где и была вся суть изменений.
def _git_status_reports_changes(tool_messages: list[ToolMessage]) -> bool:
    for m in tool_messages:
        if m.name == "git_status" and (
            "Changes to be committed" in _tool_text(m.content)
            or "Changes not staged" in _tool_text(m.content)
        ):
            return True
    return False


# Живой прогон, ДВАЖДЫ подряд (даже после того, как системный промпт уже
# явно потребовал звать git_show/git_diff по каждому коммиту): задача
# "покажи код последних изменений" — модель вызвала только git_log
# (даёт ТОЛЬКО хэш/автора/дату/сообщение, никогда сам код), затем написала
# по абзацу на каждый коммит, процитировав его полный хэш и пересказав
# содержимое — по одному лишь сообщению коммита ("fix", "spinner"), не
# прочитав ни одного реального диффа. Судья это поймал оба раза, но раз
# промпт-инструкции одной не хватило дважды подряд на слабой локальной
# модели — нужна структурная страховка: если финальный ответ цитирует
# конкретные хэши коммитов из git_log, а git_show/git_diff по НИ ОДНОМУ из
# них в этом раунде не вызывался, значит содержимое этих коммитов
# нафантазировано по одним сообщениям, а не прочитано.
def _described_commits_without_diff(tool_messages: list[ToolMessage], final_text: str) -> bool:
    log_hashes: set[str] = set()
    for m in tool_messages:
        if m.name == "git_log":
            log_hashes.update(re.findall(r"\b[0-9a-f]{40}\b", _tool_text(m.content)))
    if not log_hashes:
        return False
    if not any(h in final_text or h[:7] in final_text for h in log_hashes):
        return False
    return not any(m.name in ("git_show", "git_diff") for m in tool_messages)


def _has_diff_evidence(tool_messages: list[ToolMessage]) -> bool:
    # Живой прогон ПОСЛЕ первой версии этой проверки: модель, следуя новому
    # системному промпту, вызвала git_diff_unstaged — но НЕ git_diff_staged,
    # хотя правки лежали и в staged, и в unstaged (правки этого самого
    # коммита поверх уже застейдженных правок предыдущего). Тогда её ответ
    # честно пересказал только незастейдженную половину и заявил "изменён
    # только комментарий", хотя основная логика была именно в staged-части.
    # git_diff("HEAD") в одиночку покрывает и то, и другое разом, поэтому
    # считаем условие выполненным либо им, либо ОБОИМИ staged+unstaged —
    # одного из двух недостаточно, если другой не вызывался.
    called = {m.name for m in tool_messages}
    if "git_diff" in called:
        return True
    return "git_diff_staged" in called and "git_diff_unstaged" in called


# Маркер, которым _cap_tool_output (см. ниже) помечает обрезанный результат
# тула. Живой прогон: git_diff_unstaged с context_lines=50 на файле с
# небольшой правкой в начале и большой веткой логики в конце — весь объём
# ушёл за TOOL_OUTPUT_CHAR_CAP, обрезка пришлась ровно перед той самой
# крупной веткой, и модель молча ответила по обрубку, ни разу не упомянув,
# что видела не весь дифф (хотя системный промпт явно требует это
# проговаривать). Судья тоже не заметил — обрубок на 2000 символов ещё
# короче того, что видит основная модель, и по нему выглядит вполне
# самодостаточным. Раз оба уровня (модель и судья) пропустили это на
# практике, ловим ДЕТЕРМИНИРОВАННО по самому факту наличия маркера.
_TRUNCATION_MARKER = "...[TRUNCATED"

_GIT_DIFF_TOOL_NAMES = ("git_diff", "git_diff_staged", "git_diff_unstaged")

# Тулы, чей УСПЕШНЫЙ (не обрезанный) результат реально даёт полную картину
# файла/диффа — используются, чтобы понять, "восстановилась" ли модель
# после обрезания где-то раньше в этом же раунде.
_GROUNDING_TOOL_NAMES = _GIT_DIFF_TOOL_NAMES + (
    "bash_exec", "read_file", "read_text_file", "read_file_range",
    "search_code", "search_code_semantic",
)


def _recovered_after_truncation(tool_messages: list[ToolMessage], last_truncated: int) -> bool:
    """Живой прогон: retry на обрезанный результат срабатывал ДАЖЕ когда
    модель после этого честно последовала совету промпта — позвала более
    узкий bash_exec `git diff -- <path>`, перечитала конкретный файл, или
    повторила тот же тул с более узким запросом — и получила ПОЛНУЮ,
    необрезанную картину. Штрафовать раунд в этом случае значит наказывать
    модель за то, что она поступила ровно так, как просит система, вместо
    того чтобы засчитать восстановление. Считаем восстановлением: после
    ПОСЛЕДНЕГО обрезанного результата (last_truncated) есть хотя бы один
    более поздний результат от "заземляющего" тула (см. _GROUNDING_TOOL_NAMES)
    БЕЗ метки обрезания — не обязательно тот же самый вызов, просто сигнал,
    что модель продолжила добывать полную информацию, а не осталась с
    обрубком как с единственным источником."""
    return any(
        m.name in _GROUNDING_TOOL_NAMES and _TRUNCATION_MARKER not in _tool_text(m.content)
        for m in tool_messages[last_truncated + 1:]
    )


# Навигационные/discovery-тулы — обрезание их результата почти никогда не
# прячет "тот самый" ответ (просто "файлов/коммитов было больше, чем
# показано"), а совет "сузь запрос" для многих из них невыполним в принципе
# (у directory_tree/git_log вообще нет параметра, которым можно сузить).
# Штрафовать раунд за их обрезание — чистые потери без защиты от чего-либо.
_NAV_TOOL_NAMES = (
    "list_directory", "list_directory_with_sizes", "directory_tree",
    "search_files", "git_log", "list_deleted_paths", "list_file_snapshots",
)


def _has_truncated_output(tool_messages: list[ToolMessage]) -> bool:
    truncated_idx = [
        i for i, m in enumerate(tool_messages)
        if m.name not in _NAV_TOOL_NAMES and _TRUNCATION_MARKER in _tool_text(m.content)
    ]
    if not truncated_idx:
        return False
    return not _recovered_after_truncation(tool_messages, max(truncated_idx))


def _truncated_git_diff(tool_messages: list[ToolMessage]) -> bool:
    # git_diff*-тулы — особый случай обрезания: их args_schema (проверено
    # напрямую) содержит ТОЛЬКО repo_path и context_lines — никакого способа
    # сузить на один файл. Обычный совет "сузь запрос" тут вводит в
    # заблуждение: сузить нечем, единственный параметр (context_lines)
    # только увеличивает вывод. Ловим отдельно, чтобы дать другой совет.
    truncated_idx = [
        i for i, m in enumerate(tool_messages)
        if m.name in _GIT_DIFF_TOOL_NAMES and _TRUNCATION_MARKER in _tool_text(m.content)
    ]
    if not truncated_idx:
        return False
    return not _recovered_after_truncation(tool_messages, max(truncated_idx))


def _extract_diffed_paths(tool_messages: list[ToolMessage]) -> set[str]:
    paths = set()
    for m in tool_messages:
        if m.name in _GIT_DIFF_TOOL_NAMES:
            paths.update(re.findall(r"^diff --git a/(\S+) b/\S+", _tool_text(m.content), re.MULTILINE))
    return paths


_WRITE_TOOL_NAMES = ("write_file", "edit_file", "replace_lines", "move_file", "copy_lines")


def _written_paths(round_msgs: list) -> set[str]:
    """Paths touched by a write/edit tool this round, used by stream_chat's
    execution-failure fallback (see _execution_evidence_shows_failure) to
    know what to auto-revert. ToolMessage itself doesn't carry the call's
    args — only the matching AIMessage.tool_calls does (same lookup pattern
    as _sibling_tool_names above), matched by tool_call_id."""
    calls_by_id = {}
    for m in round_msgs:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc.get("id"):
                    calls_by_id[tc["id"]] = tc
    paths = set()
    for m in round_msgs:
        if not isinstance(m, ToolMessage) or m.name not in _WRITE_TOOL_NAMES or m.status == "error":
            continue
        tc = calls_by_id.get(m.tool_call_id)
        if not tc:
            continue
        args = tc.get("args") or {}
        for key in ("path", "destination", "dest_path"):
            p = args.get(key)
            if p:
                paths.add(p)
    return paths


# Syntax/lint-only checks — they confirm a file PARSES, never that it BEHAVES
# correctly. Grows by the same rule as _COMMON_TOOLS in prompts.py: add an
# entry per real live bug on a new stack, not a preemptive exhaustive list.
_SYNTAX_ONLY_CMD_RE = re.compile(
    r"\bphp\s+-l\b"
    r"|\bpython3?\s+-m\s+py_compile\b"
    r"|\bnode\s+--check\b"
    r"|\btsc\b[^\n]*--noEmit\b"
    r"|\bgo\s+vet\b"
    r"|\bruby\s+-c\b",
    re.IGNORECASE,
)

# A command that actually EXECUTES a test suite (as opposed to a bare `find`/
# `grep` that merely locates test files, or a syntax check above).
_TEST_RUNNER_CMD_RE = re.compile(
    r"\bphpunit\b"
    r"|\bpytest\b"
    r"|\bpython3?\s+-m\s+unittest\b"
    r"|\bn(?:pm|px)\s+(?:run\s+)?test\b"
    r"|\byarn\s+test\b"
    r"|\bgo\s+test\b"
    r"|\bjest\b"
    r"|\brspec\b"
    r"|\bmvn\s+test\b"
    r"|\bgradle\s+test\b"
    r"|\bcargo\s+test\b",
    re.IGNORECASE,
)

# A path that looks like a test file, wherever it shows up (a `find`/`grep`
# result, a search_code hit, ...) — signals a real test suite is reachable.
_TEST_PATH_RE = re.compile(r"[\w./\\-]*(?:test|spec)[\w./\\-]*\.\w+", re.IGNORECASE)


def _bash_commands(round_msgs: list) -> list[tuple[str, str]]:
    """[(command, result_text)] for every bash_exec call this round — same
    tool_call_id lookup pattern as _written_paths above (ToolMessage doesn't
    carry the call's args, only the matching AIMessage.tool_calls does)."""
    calls_by_id = {}
    for m in round_msgs:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc.get("id"):
                    calls_by_id[tc["id"]] = tc
    out = []
    for m in round_msgs:
        if isinstance(m, ToolMessage) and m.name == "bash_exec":
            tc = calls_by_id.get(m.tool_call_id)
            command = str((tc.get("args") or {}).get("command", "")) if tc else ""
            out.append((command, _tool_text(m.content)))
    return out


# Live run (mail-server session, 20260707-135011-31723e6e): after editing two
# PHP files, the model ran `php -l` on each and reported success as its ONLY
# verification — despite having ALREADY run `find . -name "*Test*.php" |
# grep -i mailbox` a few tool calls earlier in this SAME round and gotten
# back real test file paths. `php -l` only parses the file; it proves
# nothing about whether the changed behavior is correct, and the discovered
# tests were simply never run. _has_execution_evidence only checks that
# bash_exec was called at all — a syntax-only command satisfies it just as
# well as a real test run, so this slipped through as "verified". Flags the
# narrower, common case: every bash_exec call this round is a bare syntax/
# lint check, none of them is an actual test-runner invocation, and some
# tool result from this same round names a real test file — i.e. a proper
# check was one command away and wasn't taken.
#
# Live run (mail-server, ee810cad03a24f33beb9b21b9d2b25c0): Verifier was
# checking a one-line change in MailboxConverter.php, which genuinely has
# no test file. A `list_directory` on the tests tree turned up
# MessageLinkShortnerTest.php and ValidatorTest.php — real test files, but
# for UNRELATED classes — and this function (matching ANY test-like path,
# anywhere) flagged that as "a real test was discoverable", forcing a
# retry that could never succeed (no test for this class exists) and
# burned the whole recursion budget hunting for one. Now requires the
# discovered test path's basename to actually reference one of THIS
# round's diffed files (via _extract_diffed_paths) — "some test file
# exists somewhere in the repo" is not evidence THIS change is testable.
# Falls back to the old broad match if no diff was seen yet this round
# (nothing to narrow against — stay conservative rather than let a
# same-round syntax-only check slip through unnoticed).
def _verified_with_syntax_check_only_despite_discoverable_tests(round_msgs: list) -> bool:
    commands = _bash_commands(round_msgs)
    if not commands:
        return False
    ran_syntax_only = any(_SYNTAX_ONLY_CMD_RE.search(cmd) for cmd, _ in commands)
    if not ran_syntax_only:
        return False
    if any(_TEST_RUNNER_CMD_RE.search(cmd) for cmd, _ in commands):
        return False  # a real test runner also ran this round — good enough

    tool_msgs = [m for m in round_msgs if isinstance(m, ToolMessage)]
    stems = {
        os.path.splitext(os.path.basename(p))[0].lower()
        for p in _extract_diffed_paths(tool_msgs)
    }

    def _is_relevant(test_path: str) -> bool:
        if not stems:
            return True
        base = os.path.basename(test_path).lower()
        return any(stem and stem in base for stem in stems)

    for _, result in commands:
        if any(_is_relevant(m.group(0)) for m in _TEST_PATH_RE.finditer(result)):
            return True
    for m in tool_msgs:
        if m.name == "bash_exec":
            continue
        if any(_is_relevant(mm.group(0)) for mm in _TEST_PATH_RE.finditer(_tool_text(m.content))):
            return True
    return False


# Живой прогон: задача "разбери дифф, найди баги". Модель на попытке 2
# честно вызвала git_diff('HEAD') и получила настоящий дифф по конкретным
# файлам — но ПОСЛЕ этого ещё несколько раз вызвала search_code/
# search_code_semantic по касательным темам (сами по себе разумные вопросы —
# "почему заменили X на Y", "где конфигурируются модели"), и в финальный
# ответ попало только последнее, что она посмотрела (случайный search_code
# по "permission-политика"), а не сам дифф, который она уже успешно
# прочитала. Промпт теперь просит синтезировать ответ из первичных
# доказательств, а не из последнего, что попалось на глаза — это
# ДЕТЕРМИНИРОВАННАЯ страховка на случай, если модель всё равно проигнорирует:
# если дифф реально показал изменения по конкретным файлам, а финальный
# текст не упоминает НИ ОДИН из них — ответ явно построен не по диффу.
def _final_answer_ignores_diff(final_text: str, diffed_paths: set[str]) -> bool:
    if not diffed_paths:
        return False
    return not any(p in final_text for p in diffed_paths)


# Живой прогон: спрошенный чисто read-only вопросом ("как дела с
# репозиторием"), агент увидел в дифф-фрагменте свою же функцию с маленьким
# context_lines и решил, что она неполная — сам вызвал edit_file, HITL
# отклонил (безопасный дефолт для неинтерактивного stdin), тул честно
# ответил "User rejected... Do not retry this tool call unless the user
# explicitly requests it" — и модель ТУТ ЖЕ вызвала edit_file снова с чуть
# другим содержимым, проигнорировав эту инструкцию. Системный промпт теперь
# тоже просит этого не делать — это страховка на случай, если модель всё
# равно проигнорирует. Возвращает имя тула для сообщения, а не просто bool.
def _retried_after_rejection(tool_messages: list[ToolMessage]) -> str | None:
    for prev, curr in zip(tool_messages, tool_messages[1:]):
        if curr.name == prev.name and "rejected the tool call" in _tool_text(prev.content).lower():
            return curr.name
    return None


# Живой прогон на qwen3-coder:30b: раунд без единого настоящего tool_calls
# вернул content = "Посмотрю, какие незакоммиченные изменения есть в
# проекте.\n\n<function=git_status>  </tool_call>" — модель сама сгенерировала
# невалидную разметку вызова (у неё же РАССОГЛАСОВАНЫ открывающий и
# закрывающий теги: "<function=X>" открывает, "</tool_call>" закрывает — это
# не пропущенный парсинг корректного вызова, сама генерация испорчена).
# Ollama (PARSER qwen3-coder, см. `ollama show qwen3-coder:30b --modelfile`)
# не смогла это распознать как tool call, git_status НИКОГДА не выполнился,
# а обрывок разметки утёк в content — create_agent видит AIMessage без
# tool_calls и считает это ЗАВЕРШЁННЫМ ответом. Раньше (до потокового
# answer_chunk) это было незаметно ровно до последнего yield — тот же баг,
# просто не видно было, что именно произошло, до самого конца. Ловим
# ДЕТЕРМИНИРОВАННО: пустой new_tool_msgs — это НЕ обязательно "модель дала
# прямой ответ", единственный способ отличить настоящий прямой ответ от
# сорвавшейся попытки вызова тула — поискать в content обрывки этой самой
# разметки.
_LEAKED_TOOL_CALL_MARKERS = re.compile(r"</?tool_call>|</?function(?:=|>)", re.IGNORECASE)


# Живой прогон (qwen2.5-coder:14b, НЕ qwen3-coder): другая модель, другой
# формат утечки — вместо тегов "<function=X>...</function>" она пишет
# tool-call как ГОЛЫЙ JSON-объект прямо в content: '{ "name": "get_knowledge",
# "arguments": {} }', без единого тега. _LEAKED_TOOL_CALL_MARKERS его в
# принципе не видит (нет ни "<function", ни "<tool_call" — вообще никаких
# угловых скобок), так что этот путь утечки раньше не детектировался и не
# восстанавливался ВООБЩЕ: раунд считался обычным прямым ответом без вызова
# тула, ход просто заканчивался на этом мусоре. Ищем скобочно (а не regex по
# "{...}") — значение arguments сам может содержать вложенные {}, наивный
# non-greedy regex обрубил бы совпадение на первой внутренней "}".
def _find_leaked_json_calls(text: str) -> list[dict]:
    calls = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        else:
            break  # unbalanced braces from here on — nothing more to find
        try:
            obj = json.loads(text[i:j + 1])
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and (
            "arguments" in obj or "parameters" in obj
        ):
            calls.append(obj)
        i = j + 1
    return calls


def _leaked_tool_call_syntax(text: str) -> bool:
    if _LEAKED_TOOL_CALL_MARKERS.search(text):
        return True
    return bool(_find_leaked_json_calls(text))


# Живой прогон: утечка НЕ обязательно в начале сообщения — модель сначала
# честно написала намерение ("Я вижу, что вы спрашиваете... Давайте
# исследуем структуру проекта.") и только ПОСЛЕ него пошла битая разметка
# "<function=list_directory>...". Проверка по первым N символам сообщения
# такое пропустит — вместо этого при стриминге ищем НАЧАЛО маркера в ещё
# непоказанном хвосте буфера (см. _stream_round) и, если он найден,
# показываем всё, что было ДО него (это настоящий текст), а дальше — молча
# копим для парсинга, ничего больше не печатая.
_LEAK_MARKER_START_RE = re.compile(r"<(?:tool_call|function=)", re.IGNORECASE)

# Сколько символов НЕ показываем сразу, а придерживаем как потенциальное
# начало ещё не полностью пришедшего маркера — длиннее любого реального
# маркера ("<function=" — 10 символов, "<tool_call>" — 11), так что если за
# этим "хвостом" маркер так и не сложился, значит его там и не было.
_LEAK_TAIL_MARGIN = 20


# Живой прогон: ретрай с просьбой "вызови по-настоящему" НЕ помог — модель
# дважды подряд сгенерировала ДОСЛОВНО тот же самый битый
# "<function=list_directory><parameter=path>...</parameter></function>
# </tool_call>", сжигая обе оставшиеся попытки на одной и той же генерации.
# Раз формат утечки нам точно известен (это родной формат этой же модели,
# см. PARSER qwen3-coder выше — она не путает синтаксис, его просто не
# распознал парсер Ollama), надёжнее самим вытащить {name, args} и
# выполнить тул напрямую в обход сломанного парсера, чем ждать, что модель
# в кои-то веки сгенерирует иначе.
_LEAKED_FUNCTION_RE = re.compile(r"<function=([a-zA-Z0-9_]+)>(.*?)</function>", re.DOTALL)
_LEAKED_PARAM_RE = re.compile(r"<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>", re.DOTALL)


def _coerce_leaked_param(raw: str):
    # Утёкшие параметры приходят как голый текст ("5", "true", "[\"a\"]") —
    # пробуем распознать реальный тип через JSON, иначе оставляем строкой
    # как есть (обычный случай: путь, текст).
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _parse_leaked_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _LEAKED_FUNCTION_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        args = {pm.group(1): _coerce_leaked_param(pm.group(2)) for pm in _LEAKED_PARAM_RE.finditer(body)}
        calls.append({"name": name, "args": args})
    if calls:
        return calls
    # No <function=...> tags found — fall back to the bare-JSON leak shape
    # (see _find_leaked_json_calls above). Args here are already real JSON
    # values (numbers/bools/objects), not strings needing _coerce_leaked_param.
    for obj in _find_leaked_json_calls(text):
        args = obj.get("arguments", obj.get("parameters"))
        calls.append({"name": obj["name"], "args": args if isinstance(args, dict) else {}})
    return calls


async def _execute_leaked_tool_call(tools_by_name: dict, name: str, args: dict) -> str:
    """Выполняет распарсенный из _parse_leaked_tool_calls вызов напрямую,
    в обход create_agent/HumanInTheLoopMiddleware — раз мы сами дёргаем тул
    вместо графа, именно этот код теперь единственное место, где решается,
    нужно ли спросить подтверждение (см. TOOLS_REQUIRING_APPROVAL) — иначе
    мутирующий тул (bash_exec, write_file, ...) выполнился бы в обход
    permission-диалога только потому, что его вызов не распознал парсер
    модели. По той же причине — единственное место (для ЭТОГО, обходного
    пути) где нужно повторить VRAM-side-effect
    _UnloadImageGenBeforeGenModelMiddleware (agent_builder.py) делает для
    обычного, распознанного пути: generate_3d_model/generate_texture_for_model
    запускают Hunyuan3D-2GP-subprocess, которому резидентный SDXL/FLUX pipe
    в image_gen_server.py мешал бы конкурировать за VRAM."""
    matched_tool = tools_by_name.get(name)
    if matched_tool is None:
        return f"Error: no tool named `{name}`"
    if name in TOOLS_REQUIRING_APPROVAL:
        action, detail = _action_and_detail(name, args)
        if not await ask_permission(action, detail):
            return f"Rejected by user — `{name}` was not approved to run."
    # Deferred import -- agent_builder.py imports compaction.py, which imports
    # THIS module (for _failed_write_messages), so a top-level import here
    # would be circular. By call time (this only runs mid-conversation, well
    # after module load) agent_builder is already fully imported elsewhere.
    from mcp_agent.agent_builder import _GEN_MODEL_GPU_TOOLS, _unload_subprocess_models
    if name in _GEN_MODEL_GPU_TOOLS:
        await _unload_subprocess_models()
    try:
        result = await matched_tool.ainvoke(args)
    except Exception as e:
        return f"Error running `{name}`: {e}"
    return str(result)


def _called_ask_user(tool_messages: list[ToolMessage]) -> bool:
    return any(m.name == "ask_user" for m in tool_messages)


_ASK_EXTRACT_SYSTEM_PROMPT = (
    "You extract a clean, interactive question from a rambling answer that "
    "asked the user something in plain text instead of using a proper "
    "question tool. Produce a SHORT, direct question (ONE sentence, not the "
    "whole answer), at most 4 options (each a short label — a few words — "
    "plus an EVEN SHORTER, under-10-word rationale), and which option (if "
    "any) the answer leaned towards — null if it didn't lean towards one. "
    "If the answer didn't present distinct options at all (a genuinely "
    "open-ended question), return an empty options list. Keep EVERYTHING "
    "terse — you have a strict output budget, and a cut-off response is "
    "useless. Write the \"question\" key FIRST, so it survives even if you "
    'run out of room for the rest. Respond with ONLY this JSON: {"question": '
    '"...", "options": [{"label": "...", "description": "..."}], '
    '"recommended": "..."|null}'
)


async def _extract_ask_user_shape(judge_model, answer_text: str) -> dict:
    """Самолечение punt-to-user (см. stream_chat) раньше открывало диалог с
    ВСЕМ текстом ответа модели как question и без единого варианта — живой
    прогон: 561-токенное эссе целиком стало "вопросом", хотя внутри уже
    были готовые варианты (React/Vue/Alpine.js). Сжимаем судьёй (той же
    моделью, что и _semantic_check) в компактные {question, options,
    recommended} — тот же формат, что и у самого ask_user.

    options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": 500} — увеличенный
    num_predict в обход общего JUDGE_NUM_PREDICT=200 (используется тут же
    рядом для куда более короткого _semantic_check): живой прогон показал,
    что со стандартным бюджетом реальная модель обрывала JSON ДО того, как
    дописывала "question" ("ask_user extraction failed: empty extracted
    question"), и весь смысл сжатия терялся, снова падая на fallback с сырым
    текстом целиком. num_ctx передан явно, а не через .bind(num_predict=...)
    — тоже живой прогон: .bind() кладёт kwarg на верхний уровень запроса, а
    не в ollama-шный options{}, из-за чего AsyncClient.chat() падал с
    "unexpected keyword argument 'num_predict'"; options{} ЗАМЕЩАЕТ, а не
    дополняет параметры модели (см. ChatOllama._chat_params), так что при
    явной передаче options{} нужно продублировать в нём и num_ctx — иначе
    контекст этого конкретного вызова тихо схлопнется к дефолту Ollama.

    options={} — Ollama-специфичный kwarg (ChatOllama._chat_params сливает
    его в params["options"]) — judge_model — это та же _build_chat_model
    развилка, что и основная модель (см. agent_builder.py), так что при
    settings.expert_streaming_enabled это ChatOpenAI, не ChatOllama. Живой
    баг: options={...} на ChatOpenAI.ainvoke прозрачно долетал до
    AsyncCompletions.create(), который такого параметра не знает —
    "unexpected keyword argument 'options'", извлечение вопроса тихо падало
    в fallback на except ниже. ChatOpenAI использует max_tokens напрямую
    (не через options{}) и не имеет per-call аналога num_ctx — контекст у
    expert-streaming фиксирован при старте процесса (-c, см.
    expert_streaming.py), так что там достаточно одного max_tokens."""
    try:
        extra_kwargs = (
            {"options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 500}}
            if isinstance(judge_model, ChatOllama)
            else {"max_tokens": 500}
        )
        resp = await judge_model.ainvoke(
            [
                {"role": "system", "content": _ASK_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": answer_text},
            ],
            **extra_kwargs,
        )
        data = parse_json_loose(resp.content) or {}
        question = str(data.get("question") or "").strip()
        options = [
            {"label": str(o.get("label", "")).strip(), "description": str(o.get("description", "")).strip()}
            for o in data.get("options", []) or []
            if isinstance(o, dict) and o.get("label")
        ]
        recommended = data.get("recommended")
        recommended = str(recommended).strip() if recommended else None
        if not question:
            raise ValueError("empty extracted question")
        return {"question": question, "options": options, "recommended": recommended}
    except Exception as e:
        # fail open — тот же принцип, что в _semantic_check: сломанное
        # извлечение не должно блокировать сам self-heal, просто откатываемся
        # к исходному (менее аккуратному, но рабочему) варианту без вариантов.
        if DEBUG:
            console.print(f"[dim][MCP-AGENT] ask_user extraction failed: {e}[/]")
        log_event("ask_user_extraction_failed", error=str(e))
        return {"question": answer_text, "options": [], "recommended": None}


async def _semantic_check(
    model,
    task: str,
    tool_messages: list[ToolMessage],
    answer_text: str,
    ask_user_called: bool,
) -> dict:
    """Портирован из verifier/verifier.py:_semantic_check — то же назначение:
    coverage (тулы отработали) не значит relevance (результат отвечает на
    задачу). create_agent сам решает, когда звать тулы, но не проверяет
    результат против исходной задачи перед тем как сдаться.

    answer_text (финальный текст модели) передаётся судье НАРЯДУ с сырыми
    tool-результатами — без этого судья технически не может засчитать ни
    один judgment-call/рекомендательный запрос ("что лучше выбрать", "реши
    сам"): вывод/рекомендация — это синтез модели в её ОТВЕТЕ, а не факт,
    который лежит в сыром выводе тула (файл/diff/поиск). Живой прогон:
    задача "React или Vue, реши сам" — судья, видя только содержимое
    прочитанных файлов (JS/HTML), раз за разом браковал результат с
    причиной "results ... do not provide a comparison or recommendation",
    хотя сама модель к тому моменту уже сформулировала рекомендацию в
    ответе — судья её просто не видел и жёг все MAX_ATTEMPTS впустую.

    ask_user_called — тоже нужен судье НАРЯДУ с answer_text: определить
    "отфутболила ли модель решение обратно пользователю текстом вместо
    настоящего вызова ask_user" — это семантическое суждение о естественном
    языке (живой пример: "Хотели бы вы React или Vue? Я могу предложить
    план..." — вопрос не последним предложением и вообще не обязан
    заканчиваться на "?", ср. "Скажите, что предпочитаете"), а не то, что
    надёжно ловится regex'ом по пунктуации — раньше здесь стояла именно
    такая эвристика (_ended_with_unresolved_question), она уже один раз не
    сработала на этом самом примере. А вот тот факт, что ask_user
    ДЕЙСТВИТЕЛЬНО был вызван в этом раунде — чисто структурный (есть ли
    такое имя тула среди new_tool_msgs), в LLM для этого обращаться незачем
    — вычисляем один раз кодом и просто сообщаем судье как готовый факт.

    Живой прогон (mail-server, 20260707-135011-31723e6e): задача явно сказала
    "не проставляй FAILED при отмене, сделай умнее" — модель сама же вызвала
    ask_user, но её СОБСТВЕННЫЙ recommended-вариант был "проставлять FAILED
    при отмене", и после ответа пользователя (согласившегося с рекомендацией)
    это же и было реализовано — прямая противоположность задаче, никем не
    пойманная. Проверка "разворот направления" (см. mode 4 в
    _SEMANTIC_SYSTEM_PROMPT и абзац в prompt ниже) детерминированно её не
    ловит (это не структурный факт про tool_messages, а смысловое сравнение
    ANSWER с TASK) — сознательно отдана судье, а не отдельной эвристике."""
    summary = (
        "\n".join(f"- {m.name}: {_tool_text(m.content)[:_JUDGE_SNIPPET_CHARS]}" for m in tool_messages)
        if tool_messages else "(no tools were called this round)"
    )
    prompt = (
        f"TASK:\n{task}\n\nTOOL RESULTS:\n{summary}\n\nANSWER:\n{answer_text}\n\n"
        f"ASK_USER TOOL CALLED THIS ROUND: {ask_user_called}\n\n"
        "Off-topic, empty, or contradictory tool results are NOT relevant "
        'even if their status is "ok". For fact-finding tasks, an answer '
        "based on a result that only PARTIALLY covers the task (misses some "
        "of what was asked, or looks truncated) is also NOT relevant — say "
        "so in the reason so the agent knows to fetch the rest, not just "
        "retry blindly. For judgment/opinion/recommendation tasks, judge "
        "the ANSWER itself: does it make a real, reasoned call (not a vague "
        "non-answer or a refusal to decide)? Don't fail it just because the "
        "raw tool results don't literally contain a recommendation — they "
        "never will, that's the answer's job. If instead the answer hands a "
        "decision/choice/open question back to the user in plain text and "
        "ASK_USER TOOL CALLED THIS ROUND is False, that's NOT relevant no "
        "matter how reasonable the question sounds. Separately, re-read TASK "
        "for an explicit 'don't do X' — if the ANSWER's chosen or recommended "
        "option, or the change actually made (see TOOL RESULTS for what was "
        "written/edited), does X anyway, that's NOT relevant regardless of "
        "how sound the reasoning looks in isolation; quote the contradicting "
        "phrase from TASK in the reason. Also check: if TASK names a "
        "specific mechanism/workflow, does the file/class actually "
        "written/edited (see TOOL RESULTS) belong to THAT mechanism, or to "
        "something else that merely shares a keyword with it? A real, "
        "well-verified fix to the wrong file is still NOT relevant — don't "
        "take the ANSWER's own claim of a connection at face value, check "
        "whether the TOOL RESULTS actually show that file being part of "
        "the workflow TASK describes.\n\n"
        'Respond with ONLY this JSON: {"relevant": true|false, "reason": "short reason"}'
    )
    try:
        resp = await model.ainvoke(
            [{"role": "system", "content": _SEMANTIC_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        data = parse_json_loose(resp.content) or {}
        verdict = {"relevant": bool(data.get("relevant", True)), "reason": str(data.get("reason", ""))}
    except Exception as e:
        # fail open — тот же принцип, что в verifier.py: ломаный чекер не
        # должен блокировать иначе валидный результат
        verdict = {"relevant": True, "reason": f"semantic check unavailable: {e}"}

    if DEBUG:
        console.print(f"[dim][MCP-AGENT] Semantic verdict: {verdict}[/]")
    log_event("semantic_verdict", **verdict)
    return verdict
