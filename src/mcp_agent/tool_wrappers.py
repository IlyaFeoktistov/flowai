"""
Обёртки над MCP-тулами, применяемые в agent_builder.py:_build_agent при
сборке итогового списка тулов агента. Каждая — чистая функция
`BaseTool -> BaseTool`, оборачивающая исходный coroutine и/или description,
без общего состояния между собой (кроме read_history/repo_path/constants,
передаваемых явно параметрами):

- _cap_tool_output/_sandwich_truncate — обрезка непомерно длинных
  tool-результатов с сохранением головы И хвоста (там обычно самое важное).
- _dedupe_read_tool/_invalidate_read_cache_tool/_wrap_read_invalidation —
  дедуп повторных чтений одного файла + инвалидация этого кэша после любой
  мутации, которая могла его устареть.
- _add_verify_reminder/_add_regex_warning — дописывают напоминание/
  предупреждение прямо в description тула, видимое модели ДО вызова.
- _bind_constant_args — прячет от модели аргументы, значение которых и так
  константно на всю сессию (repo_path у git-тула).

file_ops_server.py's read_file/write_file/edit_file/grep_search/glob_search
(заменили filesystem MCP-сервер + code_search_server.py + fs_extra_server.py)
сделали часть прежних обёрток здесь ненужными: бинарный/размерный гард —
теперь ВНУТРИ read_file (свой тул, гард можно держать в естественном месте,
не внешней обёрткой над чужим npm-пакетом); head/tail-конфликт
(_split_head_tail_tool) — offset+limit read_file не имеет такого конфликта в
принципе; JSON-нормализация edits (_normalize_edit_file_args) — новый
edit_file берёт plain old_string/new_string, не JSON-массив;
expected_*_hash-машинерия (_require_expected_lines/_cache_line_content_tool/
_autofill_expected_lines) — новый edit_file адресуется по уникальности
old_string, не по номеру строки, дрейф номеров при параллельных правках эту
проблему не создаёт вообще.
"""
from langchain_core.tools import BaseTool, StructuredTool

from mcp_agent.message_utils import _content_text

# Голова получает больше бюджета, чем хвост (60/40) — там обычно заголовок/
# аргументы команды, но хвост остаётся достаточно большим, чтобы вместить
# итог/traceback/последний assert, которые почти всегда важнее середины.
_TRUNCATE_HEAD_RATIO = 0.6


def _rewrap_tool(
    tool: BaseTool, *, coroutine, description: str | None = None, args_schema=None,
) -> BaseTool:
    """Rebuilds `tool` as a StructuredTool around a new `coroutine` (and,
    optionally, a new `description`/`args_schema`), copying every other
    field (response_format/metadata/handle_tool_error) unchanged. Every
    wrapper below — and mcp_agent/snapshots.py:_snapshot_before_write —
    used to reconstruct this same 7-field StructuredTool by hand, varying
    only the 1-3 fields that actually change for that wrapper; single home
    for that reconstruction now."""
    return StructuredTool(
        name=tool.name,
        description=tool.description if description is None else description,
        args_schema=tool.args_schema if args_schema is None else args_schema,
        coroutine=coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _sandwich_truncate(text: str, max_chars: int) -> str:
    """Раньше обрезка была чисто head (первые max_chars, остальное отброшено)
    — для вывода тестов/линтеров/git diff самое важное (итог, traceback,
    последний failed assert) почти всегда в ХВОСТЕ, так что head-обрезка
    топила именно ту часть, ради которой модель звала тул, и она была
    вынуждена переспрашивать/сужать запрос отдельным ходом. Сэндвич (голова +
    хвост, дыра в середине) даёт тот же бюджет символов, но с сигналом с
    обоих концов."""
    head_budget = int(max_chars * _TRUNCATE_HEAD_RATIO)
    tail_budget = max_chars - head_budget
    omitted = len(text) - max_chars
    marker = (
        f"\n\n...[TRUNCATED: {omitted} characters omitted from the middle "
        "(showing first "
        f"{head_budget} and last {tail_budget} of {len(text)} total) — "
        "narrow the query (smaller context_lines/max_results, a more "
        "specific path or pattern, or bash with head/tail/grep) if you "
        "need the omitted part]...\n\n"
    )
    return text[:head_budget] + marker + text[-tail_budget:]


def _cap_tool_output(tool: BaseTool, max_chars: int) -> BaseTool:
    """Обрезает текстовые content-блоки в результате `tool`, если они длиннее
    `max_chars` (см. _sandwich_truncate для того, ПОЧЕМУ обрезка не с конца),
    и явно помечает обрезание — модель видит, что данных больше, и может сама
    сузить запрос, вместо того чтобы утопить контекст целиком."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    def _truncate(block):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if len(text) > max_chars:
                return {**block, "text": _sandwich_truncate(text, max_chars)}
        return block

    async def _call(**kwargs):
        content, artifact = await original_coroutine(**kwargs)
        if isinstance(content, list):
            content = [_truncate(b) for b in content]
        elif isinstance(content, str) and len(content) > max_chars:
            content = _sandwich_truncate(content, max_chars)
        return content, artifact

    return _rewrap_tool(tool, coroutine=_call)


def _dedupe_read_tool(tool: BaseTool, read_history: dict) -> BaseTool:
    """read_file's offset/limit window means the model can end up hunting
    for a spot in a large file with a series of overlapping windows
    (offset=0/limit=50, offset=30/limit=50, ...) instead
    of reading once with intent — each one re-drags already-seen content
    into context; 5 such reads of one 1055-line file in a single turn can
    balloon the final round to 500k+ input tokens. read_history — a plain
    {path: [key, ...]} dict, created in _build_agent and cleared at the
    start of every stream_chat (a fresh thread — the model doesn't remember
    past turns, so dedup across a turn boundary would be wrong).

    Key is every kwarg except path (offset/limit) — a repeat with the exact
    same window returns a pointer to the earlier result instead of paying
    for it again; 2+ reads of the same path with DIFFERENT windows earns an
    explicit nudge to stop narrowing by trial and error."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        path = kwargs.get("path")
        key = tuple(sorted((k, v) for k, v in kwargs.items() if k != "path"))
        seen = read_history.setdefault(path, [])

        if key in seen:
            return (
                f"(You already read `{path}` with these exact parameters "
                "earlier this turn — reuse that earlier result instead of "
                "reading it again.)",
                None,
            )

        content, artifact = await original_coroutine(**kwargs)
        seen.append(key)
        if len(seen) >= 2:
            hint = (
                f"\n\n[You've now read `{path}` {len(seen)} times this turn "
                "with different offset/limit windows — that's hunting for a "
                "spot by trial and error, not reading with intent. If the "
                "file is a few hundred lines or less, read it ONCE with no "
                "offset/limit instead of piecing it together from "
                "fragments; if you need a specific symbol's surroundings, "
                "use grep_search(pattern, output_mode='content', "
                "context=N) to get the match AND its context in one call "
                "instead of grep-then-read.]"
            )
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": hint}]
            elif isinstance(content, str):
                content = content + hint
        return content, artifact

    return _rewrap_tool(tool, coroutine=_call)


def _invalidate_read_cache_tool(tool: BaseTool, read_history: dict, get_paths) -> BaseTool:
    """После write_file/edit_file/bash/git-мутаций читает read_history устарел
    для затронутых путей — без инвалидации _dedupe_read_tool продолжил бы
    отвечать "вы уже читали этот файл" и отдавать модели её же дочтения ДО
    правки вместо актуального содержимого (например, модель правит файл,
    перечитывает его же для verify-шага — и получает закэшированный текст с
    якобы неизменным кодом).

    get_paths(kwargs) -> list[str] для точечной инвалидации конкретных
    путей, или None — тогда read_history чистится целиком. Полная очистка
    используется там, где команда произвольна и заранее неизвестно, какие
    файлы она затронула (bash, git checkout/reset).

    Инвалидируем только при реальном успехе — ошибочный результат (файл не
    изменился) не должен топить валидный кэш."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        result = await original_coroutine(**kwargs)
        content = result[0] if isinstance(result, tuple) else result
        if _content_text(content).strip().lower().startswith(("error", "mcp error")):
            return result
        paths = get_paths(kwargs)
        if paths is None:
            read_history.clear()
        else:
            for path in paths:
                read_history.pop(path, None)
        return result

    return _rewrap_tool(tool, coroutine=_call)


# Тулы, после которых read_history инвалидируется точечно (только путь(и) из
# аргументов вызова) — файловые операции с известным затронутым файлом.
_SINGLE_PATH_INVALIDATORS = ("write_file", "edit_file", "restore_file_snapshot", "delete_path")
# Тулы, после которых read_history чистится целиком — произвольная bash-
# команда (включая любую git-мутацию — своего тула для них больше нет,
# всё идёт через bash) может задеть любой файл в репозитории, и заранее
# неизвестно, какой именно. restore_deleted_path сюда же: аргумент —
# trash_id, не path, целевой путь известен только после лукапа в БД,
# точечная инвалидация по kwargs невозможна.
_WHOLE_CACHE_INVALIDATORS = ("bash", "restore_deleted_path")


def _wrap_read_invalidation(tool: BaseTool, read_history: dict) -> BaseTool:
    if tool.name in _SINGLE_PATH_INVALIDATORS:
        return _invalidate_read_cache_tool(
            tool, read_history, lambda kw: [kw["path"]] if kw.get("path") else []
        )
    if tool.name in _WHOLE_CACHE_INVALIDATORS:
        return _invalidate_read_cache_tool(tool, read_history, lambda kw: None)
    return tool


# Дописывается к description write_file/edit_file, чтобы модель видела
# требование проверки в момент, когда решает вызвать тул, а не только в
# системном промпте (~3000 токенов, где такой же пункт регулярно
# терялся — модель могла записать файл и ни разу не вызвать bash;
# retry-цикл ловит это детерминированной проверкой (_wrote_code без
# _has_execution_evidence), сжигая на этом целую попытку).
_VERIFY_REMINDER = (
    " After this succeeds, verifying it actually works (running it, its "
    "tests, or at minimum confirming it imports/parses) via bash is "
    "REQUIRED before reporting success — a successful write/edit only means "
    "the file was saved, not that it works."
)


def _add_verify_reminder(tool: BaseTool) -> BaseTool:
    """Дублирует напоминание про верификацию в двух проксимальных к
    write_file/edit_file местах — description (видна ДО вызова, влияет на
    решение) и хинт на сам успешный результат (видна СРАЗУ ПОСЛЕ, пока
    контекст ещё "здесь") — вместо единственного места в системном
    промпте, которое легко теряется среди остального текста."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        content, artifact = await original_coroutine(**kwargs)
        # Не вешаем хинт на ошибку — там уже есть подробное сообщение об
        # ошибке, а "проверь через bash" на незаписанный файл сбивает с
        # толку. status="error" здесь недоступен (выставляется выше по
        # стеку, см. _failed_write_messages) — грубая проверка по тексту
        # достаточна, т.к. это лишь дополнительная подсказка, не единственный
        # источник правды (тот — детерминированный verdict в stream_chat).
        if not _content_text(content).lower().startswith(("error", "mcp error")):
            hint = "\n\n[Reminder: verify this actually works via bash before reporting success.]"
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": hint}]
            elif isinstance(content, str):
                content = content + hint
        return content, artifact

    return _rewrap_tool(tool, coroutine=_call, description=tool.description + _VERIFY_REMINDER)


# Дописывается к description grep_search (file_ops_server.py): без
# case_insensitive=True поиск матчит регистр буквально — модель иногда
# ожидает case-insensitive по умолчанию и молча получает пустой результат
# вместо ошибки, объясняющей почему.
_REGEX_WARNING = (
    " NOTE: matching is case-SENSITIVE by default — pass "
    "case_insensitive=true if the exact case isn't known/relevant. Pattern "
    "is a real regex (ripgrep-style) — a literal '.'/'*'/'\\\\' etc. still "
    "needs escaping if you mean it literally, e.g. search for '\\\\.foo' to "
    "match a literal '.foo'."
)


def _add_regex_warning(tool: BaseTool) -> BaseTool:
    return _rewrap_tool(tool, coroutine=tool.coroutine, description=tool.description + _REGEX_WARNING)


def _bind_constant_args(tool: BaseTool, constants: dict) -> BaseTool:
    """Убирает ключи `constants` из схемы, которую видит модель для `tool`, и
    подставляет их значения в коде при реальном вызове — для аргументов,
    значение которых известно коду заранее и не меняется в рамках сессии
    (например repo_path для тула, чей процесс уже запущен с этим путём).
    Модели незачем вообще его видеть — тогда ошибиться в нём физически
    невозможно, а не просто "менее вероятно" (когда такой аргумент виден в
    схеме, модель может по ошибке подставить плейсхолдер вместо реального
    значения). Не применяется ни к одному текущему
    тулу — общая утилита, используемая при необходимости (см. dnd_tools.py
    про тот же принцип для game_id)."""
    schema = tool.args_schema
    if not isinstance(schema, dict) or not any(k in schema.get("properties", {}) for k in constants):
        return tool  # схема не в этой форме или не содержит ни одного из constants — не наш случай

    reduced_schema = {
        **schema,
        "properties": {k: v for k, v in schema["properties"].items() if k not in constants},
        "required": [r for r in schema.get("required", []) if r not in constants],
    }
    original_coroutine = tool.coroutine

    async def _call(**kwargs):
        # Урезанная JSON-схема не запрещает additionalProperties, а модель на
        # практике может подставить ключ, которого нет в схеме вообще. Если
        # она угадает и сам constants-ключ — без этой строки
        # получаем `TypeError: got multiple values for keyword argument`,
        # так как он придёт и в kwargs от модели, и в constants от кода.
        # Код должен побеждать безусловно — модель этот ключ не видит и не
        # должна на него влиять.
        for k in constants:
            kwargs.pop(k, None)
        return await original_coroutine(**kwargs, **constants)

    return _rewrap_tool(tool, coroutine=_call, args_schema=reduced_schema)
