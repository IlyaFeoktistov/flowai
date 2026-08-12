"""
Обёртки над MCP-тулами, применяемые в agent_builder.py:_build_agent при
сборке итогового списка тулов агента. Каждая — чистая функция
`BaseTool -> BaseTool`, оборачивающая исходный coroutine и/или description,
без общего состояния между собой (кроме read_history/repo_path/constants,
передаваемых явно параметрами):

- _cap_tool_output/_sandwich_truncate — обрезка непомерно длинных
  tool-результатов с сохранением головы И хвоста (там обычно самое важное).
- _normalize_edit_file_args/_escape_raw_control_chars_in_json_strings —
  чинит типичные способы, которыми модель ломает `edits` у edit_file.
- _dedupe_read_tool/_invalidate_read_cache_tool/_wrap_read_invalidation —
  дедуп повторных чтений одного файла + инвалидация этого кэша после любой
  мутации, которая могла его устареть.
- _add_verify_reminder/_add_glob_warning/_add_regex_warning — дописывают
  напоминание/предупреждение прямо в description тула, видимое модели ДО
  вызова.
- _require_expected_lines — делает expected_first_line/expected_last_line/
  expected_line ОБЯЗАТЕЛЬНЫМИ для replace_lines/copy_lines/insert_lines
  (в схеме тула они опциональны) — без них правка по номерам строк
  выполняется вслепую, без проверки актуальности после сдвига от
  предыдущей правки. Также считает по turn_state (= read_history с другим
  namespace ключей), сколько раз за ход каждый из этих тулов словил
  несовпадение строки/формата, и с 1-й же ошибки усиливает подсказку.
- _cache_line_content_tool/_autofill_expected_lines — модель на живых
  прогонах регулярно НЕ может корректно перепечатать expected_first_line/
  expected_last_line/expected_line (весь старый многострочный блок, пустая
  строка после отказа, и т.п.), даже когда только что сама прочитала
  нужный диапазон. Первая кэширует реальное содержимое строк с диска после
  каждого удачного read_file/read_text_file/read_file_range/
  read_multiple_files; вторая при вызове replace_lines/insert_lines/
  copy_lines подставляет expected_*_line из этого кэша сама, если диапазон
  им покрыт — вместо того чтобы полагаться на то, что модель донесёт его
  без искажений. Промах кэша падает обратно на _require_expected_lines.
- _bind_constant_args — прячет от модели аргументы, значение которых и так
  константно на всю сессию (repo_path у git-тулов).
"""
import json
import re

from langchain_core.tools import BaseTool, StructuredTool

from mcp_agent.message_utils import _content_text

# Голова получает больше бюджета, чем хвост (60/40) — там обычно заголовок/
# аргументы команды, но хвост остаётся достаточно большим, чтобы вместить
# итог/traceback/последний assert, которые почти всегда важнее середины.
_TRUNCATE_HEAD_RATIO = 0.6


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
        "specific path or pattern, or bash_exec with head/tail/grep) if you "
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

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


# Матчит ОДИН JSON-строковый литерал целиком, включая экранированные
# символы внутри (\\" не обрывает строку раньше времени) — нужен, чтобы
# чинить только СОДЕРЖИМОЕ строк, не трогая структурные пробелы/переносы
# JSON СНАРУЖИ строк (там перенос строки — это просто незначащий пробел,
# экранировать его в '\n' сломало бы синтаксис, а не починил бы его).
_JSON_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


def _escape_raw_control_chars_in_json_strings(text: str) -> str:
    """Живой прогон: модель регулярно передаёт edit_file's `edits` как
    структурно верный JSON-массив (кавычки на месте, экранированы), но с
    НАСТОЯЩИМИ переносами строк внутри значений oldText/newText вместо
    экранированных '\\n' — technically невалидный JSON ("Invalid control
    character"), хотя сама структура правильная и однозначно намеренная.
    json.loads() эту форму отвергает целиком, даже если 99% значения —
    валидный многострочный код. Экранируем control-символы ТОЛЬКО внутри
    строковых литералов (не снаружи, где перенос — не значащий пробел) и
    даём json.loads() ещё один шанс, прежде чем сдаться на оригинал."""
    def _fix(m: re.Match) -> str:
        s = m.group(0)
        return s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n").replace("\t", "\\t")
    return _JSON_STRING_LITERAL_RE.sub(_fix, text)


def _normalize_edit_file_args(tool: BaseTool) -> BaseTool:
    """edit_file (filesystem MCP server) требует edits: [{oldText, newText},
    ...] — настоящий массив объектов. Живой прогон: модель дважды подряд не
    смогла сгенерировать эту форму верно — один раз передала edits как ОДИН
    объект вместо массива, второй раз как JSON-СТРОКУ вместо нативного
    массива (обе попытки упали на MCP-валидации: "expected array"). Ретрай
    с подсказкой про формат не лечит — на третьей (последней) попытке модель
    просто сдалась и выдала код в чат вместо повторного вызова. Раз модель
    ненадёжно генерирует именно эту форму, нормализуем сами, до строгого
    валидатора MCP-сервера — тогда ошибиться в этом месте физически
    невозможно, а не просто "менее вероятно"."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    def _maybe_parse_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            try:
                return json.loads(_escape_raw_control_chars_in_json_strings(value))
            except (json.JSONDecodeError, ValueError, TypeError):
                return value  # действительно не JSON — оставляем как есть, пусть падает с исходной понятной ошибкой
        return value

    def _normalize_edits(edits):
        edits = _maybe_parse_json(edits)
        if isinstance(edits, dict):
            edits = [edits]
        if isinstance(edits, list):
            edits = [_maybe_parse_json(e) for e in edits]
        return edits

    async def _call(**kwargs):
        if "edits" in kwargs:
            kwargs = {**kwargs, "edits": _normalize_edits(kwargs["edits"])}
        return await original_coroutine(**kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _dedupe_read_tool(tool: BaseTool, read_history: dict) -> BaseTool:
    """read_file/read_text_file (filesystem MCP server) не умеют читать
    произвольный диапазон строк — только "первые N" (head) или "последние N"
    (tail). При поиске нужного места в большом файле модель нащупывает его
    серией пересекающихся чтений (head:800, head:1200, head:1030, ...) —
    каждое повторно тащит в контекст уже виденное содержимое. Живой прогон:
    5 таких чтений ОДНОГО 1055-строчного файла за один ход разогнали
    финальный раунд до 531 394 входных токенов. read_history — общий
    словарь {path: [key, ...]}, создаётся в _build_agent и очищается в
    начале каждого stream_chat (свежий тред — модель не помнит прошлые
    ходы, так что дедуп через границу ходов был бы неверен).

    Ключ дедупа — все параметры вызова, кроме path, а не жёстко (head, tail):
    read_file_range (code_search-сервер, точный диапазон строк) заведён
    сюда же тем же кодом — head/tail и start_line/end_line одинаково
    определяют "тот же самый запрос к тому же файлу", а раз сюда попадает
    только read_file/read_text_file/read_file_range (см. вызов ниже), других
    комбинаций параметров тут не бывает.

    Живой прогон (mail-server, 20260708-2109): порог подсказки раньше стоял
    на 3-м чтении — модель успевала прочитать один ~200-строчный файл 5-6
    раз пересекающимися диапазонами (130-145, 140-152, 150-200, 153-180,
    179-200 — всё это уже было в самом первом read_file этого же файла),
    прежде чем подсказка вообще появлялась. Порог снижен до 2-го чтения, а
    формулировка — с расплывчатого "у тебя, наверное, уже достаточно
    контекста" на прямое указание, что делать вместо ещё одного диапазона."""
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
                "with different ranges — that's hunting for a spot by trial "
                "and error, not reading with intent. Stop narrowing with "
                "more overlapping ranges: if the file is a few hundred lines "
                "or less, read it ONCE with read_file/read_text_file (no "
                "head/tail) instead of piecing it together from fragments; "
                "if you need a specific symbol's surroundings, use "
                "search_code(query, context_lines=N) to get the match AND "
                "its context in one call instead of grep-then-read.]"
            )
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": hint}]
            elif isinstance(content, str):
                content = content + hint
        return content, artifact

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _invalidate_read_cache_tool(tool: BaseTool, read_history: dict, get_paths) -> BaseTool:
    """После write_file/edit_file/move_file/bash_exec/git-мутаций читает
    read_history устарел для затронутых путей — без инвалидации
    _dedupe_read_tool продолжил бы отвечать "вы уже читали этот файл" и
    отдавать модели её же дочтения ДО правки вместо актуального содержимого
    (например, модель правит файл, перечитывает его же диапазон строк для
    verify-шага — и получает закэшированный текст с якобы неизменным кодом).

    get_paths(kwargs) -> list[str] для точечной инвалидации конкретных
    путей, или None — тогда read_history чистится целиком. Полная очистка
    используется там, где команда произвольна и заранее неизвестно, какие
    файлы она затронула (bash_exec, git checkout/reset).

    Живой прогон: модель трижды подряд передала edit_file невалидный
    (JSON-строка с необработанными переносами строк вместо нативного
    массива) `edits` — каждая из трёх ошибок ВСЁ РАВНО чистила кэш, хотя
    файл не менялся, и модель заново перечитывала один и тот же контроллер
    по нескольку раз, раздувая ход на четверть часа и три четверти миллиона
    токенов. Инвалидируем только при реальном успехе — ошибочный результат
    (тот же текст, что file не изменился) не должен топить валидный кэш."""
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

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


# Тулы, после которых read_history инвалидируется точечно (только путь(и) из
# аргументов вызова) — филсистем-операции с известным затронутым файлом.
_SINGLE_PATH_INVALIDATORS = ("write_file", "edit_file", "git_restore_file", "restore_file_snapshot", "delete_path", "replace_lines", "insert_lines")
# move_file меняет оба конца: источник исчезает, у назначения новое
# содержимое.
_MOVE_INVALIDATORS = ("move_file",)
# copy_lines мутирует ТОЛЬКО dest_path (source_path остаётся нетронутым) —
# путь лежит под другим ключом аргумента, чем у остальных single-path тулов.
_DEST_PATH_INVALIDATORS = ("copy_lines",)
# Тулы, после которых read_history чистится целиком — произвольная команда/
# git-операция может задеть любой файл в репозитории, и заранее неизвестно,
# какой именно. restore_deleted_path сюда же: аргумент — trash_id, не path,
# целевой путь известен только после лукапа в БД, точечная инвалидация
# по kwargs невозможна.
_WHOLE_CACHE_INVALIDATORS = ("bash_exec", "git_checkout", "git_reset", "restore_deleted_path")


def _wrap_read_invalidation(tool: BaseTool, read_history: dict) -> BaseTool:
    if tool.name in _SINGLE_PATH_INVALIDATORS:
        return _invalidate_read_cache_tool(
            tool, read_history, lambda kw: [kw["path"]] if kw.get("path") else []
        )
    if tool.name in _MOVE_INVALIDATORS:
        return _invalidate_read_cache_tool(
            tool, read_history,
            lambda kw: [p for p in (kw.get("source"), kw.get("destination")) if p],
        )
    if tool.name in _DEST_PATH_INVALIDATORS:
        return _invalidate_read_cache_tool(
            tool, read_history, lambda kw: [kw["dest_path"]] if kw.get("dest_path") else []
        )
    if tool.name in _WHOLE_CACHE_INVALIDATORS:
        return _invalidate_read_cache_tool(tool, read_history, lambda kw: None)
    return tool


# Дописывается к description write_file/edit_file, чтобы модель видела
# требование проверки в момент, когда решает вызвать тул, а не только в
# системном промпте (~3000 токенов, где такой же пункт регулярно терялся —
# живой прогон: попытка 2/3 записала файл и ни разу не вызвала bash_exec,
# retry-цикл поймал это детерминированной проверкой (_wrote_code без
# _has_execution_evidence) и сжёг на этом целую попытку).
_VERIFY_REMINDER = (
    " After this succeeds, verifying it actually works (running it, its "
    "tests, or at minimum confirming it imports/parses) via bash_exec is "
    "REQUIRED before reporting success — a successful write/edit only means "
    "the file was saved, not that it works."
)


def _add_verify_reminder(tool: BaseTool) -> BaseTool:
    """Дублирует напоминание про верификацию в двух проксимальных к
    write_file/edit_file местах — description (видна ДО вызова, влияет на
    решение) и хинт на сам успешный результат (видна СРАЗУ ПОСЛЕ, пока
    контекст ещё "здесь") — вместо единственного места в системном
    промпте, которое теряется среди остального текста на живых прогонах."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        content, artifact = await original_coroutine(**kwargs)
        # Не вешаем хинт на ошибку — там уже есть подробное сообщение об
        # ошибке, а "проверь через bash_exec" на незаписанный файл сбивает с
        # толку. status="error" здесь недоступен (выставляется выше по
        # стеку, см. _failed_write_messages) — грубая проверка по тексту
        # достаточна, т.к. это лишь дополнительная подсказка, не единственный
        # источник правды (тот — детерминированный verdict в stream_chat).
        if not _content_text(content).lower().startswith(("error", "mcp error")):
            hint = "\n\n[Reminder: verify this actually works via bash_exec before reporting success.]"
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": hint}]
            elif isinstance(content, str):
                content = content + hint
        return content, artifact

    return StructuredTool(
        name=tool.name,
        description=tool.description + _VERIFY_REMINDER,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


# Дописывается к description search_files (filesystem MCP server), чтобы
# модель видела предупреждение ДО вызова, а не только после пустого
# результата. Живой прогон: search_files(pattern="*agent*.py") вернул "No
# matches found", хотя mcp_agent/agent.py существует — search_files делает
# плоский substring-матч по имени, '*'/'?' в паттерне не поддерживаются и не
# матчат ничего буквально. code_search:find_files_by_name уже умеет
# настоящий glob (shell `find -name`) — это существующий тул, не новый,
# модель просто вызвала не тот из двух похожих.
_GLOB_WARNING = (
    " NOTE: pattern here is a plain case-insensitive SUBSTRING match on the "
    "filename — wildcards like '*' or '?' are NOT supported and will not "
    "match anything (a pattern like '*agent*.py' returns no results even "
    "when a matching file exists, because the literal '*' isn't in any real "
    "filename). For real glob patterns (e.g. '*.py', 'test_*.js'), use "
    "code_search:find_files_by_name instead."
)


def _add_glob_warning(tool: BaseTool) -> BaseTool:
    return StructuredTool(
        name=tool.name,
        description=tool.description + _GLOB_WARNING,
        args_schema=tool.args_schema,
        coroutine=tool.coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


# Дописывается к description search_code (code_search_server.py): regex
# defaults to False, so query is matched as a LITERAL fixed string
# (ripgrep/grep -F) — a backslash in the query is itself just a character
# to search for, not regex syntax. Two separate live runs typed a regex-
# looking query without regex=True and silently got "No matches" instead
# of an error: search_code(query="\\.container", ...) and later
# search_code(query="\\.hero-container", ...)/search_code(query=
# "\\.contact-container", ...)/search_code(query="\\.footer-container",
# ...) — each time the file DID contain the plain substring (e.g.
# ".hero-container {"), but the literal backslash the model added for
# "escaping the dot" made the fixed-string search look for a backslash
# character that isn't actually there. Both runs then fell back to
# bash_exec grep (which succeeded immediately on the exact same pattern,
# since shell grep here defaults to basic-regex, not fixed-string) instead
# of just fixing the search_code call — the fallback worked but wasted
# several rounds getting there.
_REGEX_WARNING = (
    " NOTE: regex defaults to False — query is matched as a LITERAL "
    "string, backslashes included. A query like '\\\\.foo' does NOT find "
    "'.foo' in this mode (it looks for a literal backslash followed by "
    "'.foo', which normally isn't there) — for a plain literal match, "
    "drop the backslash entirely ('.foo' already matches '.foo' as-is, "
    "no escaping needed for fixed-string search). Only pass regex=True "
    "(and only then use backslash-escapes/regex metacharacters) when you "
    "actually need pattern matching, e.g. alternation or a wildcard."
)


def _add_regex_warning(tool: BaseTool) -> BaseTool:
    return StructuredTool(
        name=tool.name,
        description=tool.description + _REGEX_WARNING,
        args_schema=tool.args_schema,
        coroutine=tool.coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _is_line_mismatch_error(text: str) -> bool:
    t = text.lstrip()
    return t.startswith("Error") and (
        "expected_first_line" in t or "expected_last_line" in t or "expected_line" in t
    )


def _require_expected_lines(tool: BaseTool, turn_state: dict) -> BaseTool:
    """replace_lines/copy_lines (expected_first_line/expected_last_line) и
    insert_lines (expected_line) — все три принимают их как ОПЦИОНАЛЬНЫЕ
    (fs_extra_server.py, default ""), проверка срабатывает только если
    непусто. Промпт везде говорит "ALWAYS pass... every single call, not
    just when unsure" — живой прогон (mail-server, Coder-стадия): 4 подряд
    replace_lines НИ РАЗУ их не передали. Первая правка сдвинула номера
    строк ниже себя, вторая (на СТАРЫХ номерах) попала не туда, третья и
    четвёртая начали чинить друг друга на тех же двух диапазонах взад-
    вперёд — потому что без expected_*_line тул тихо режет по номерам без
    единой проверки, что там вообще то, что модель думает. Раз промпт-
    инструкция это не гарантирует, тул сам отказывается работать вслепую —
    громкая ошибка вместо тихой неверной правки, вынуждающая перечитать
    актуальное содержимое перед повтором.

    turn_state — тот же словарь, что read_history (агент_builder.py:
    _build_tools), просто с другим namespace ключей (кортеж вместо пути) —
    он уже чистится в нужные моменты (начало stream_chat, каждый self-heal
    retry в _start_next_attempt), так что отдельный словарь с собственной
    жизнью заводить не нужно. Считает подряд/за ход рубежи expected_*_line-
    несовпадений на КАЖДОМ из трёх тулов отдельно и дописывает усиленную
    подсказку с САМОЙ ПЕРВОЙ ошибки формата, не дожидаясь повтора. Живой
    прогон (some-site, styles.css): один и тот же класс ошибки
    ("expected_first_line must be a SINGLE line") прилетел 4 раза за один
    раунд, на 4 РАЗНЫХ правках — порог был на 2-й ошибке, так что первая
    ничем не отличалась от голого текста ошибки тула. Другой живой прогон
    (theme-toggle, 3 файла) — то же самое 6 раз подряд, прежде чем модель
    вообще применила хоть одну правку. Каждое сообщение об ошибке само по
    себе было понятным и точным, но ничего не заставляло модель заметить
    повторяющийся ПАТТЕРН между попытками достаточно рано; она каждый раз
    чинила только номер строки, а не сам формат аргумента. Раз конкретно
    этот класс ошибки (многострочный expected_*_line вместо одной строки)
    почти никогда не проходит бесследно сам собой на следующей попытке —
    эскалация не ждёт повтора."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool
    failure_key = ("__line_edit_failures__", tool.name)

    def _with_escalation(content, artifact):
        text = _content_text(content)
        if _is_line_mismatch_error(text):
            count = turn_state.get(failure_key, 0) + 1
            turn_state[failure_key] = count
            hint = (
                f"\n\n[{tool.name} call rejected for a line-range/format "
                f"mismatch (attempt {count} this turn) — don't just adjust "
                "the number and retry the same way. Before the next attempt: "
                "call read_file_range for the SINGLE line you actually intend "
                "to target, confirm it's the right SELECTOR/SYMBOL and not a "
                "similarly-named one nearby (e.g. '.container' vs "
                "'.nav-container' vs '.hero-container' are different rules at "
                "different lines), then copy that ONE line's exact text into "
                "expected_first_line/expected_last_line/expected_line.]"
            )
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": hint}]
            elif isinstance(content, str):
                content = content + hint
        return content, artifact

    # content_and_artifact (см. response_format этих трёх тулов) требует
    # ДВУХэлементный tuple от coroutine, а не голую строку — короткое
    # замыкание на ошибке обязано вернуть ту же форму, что и обычный успех/
    # неудача нижестоящего original_coroutine, иначе LangChain падает на
    # валидации формата раньше, чем модель вообще увидит текст ошибки.
    if tool.name == "insert_lines":
        async def _call(**kwargs):
            if not kwargs.get("expected_line") and kwargs.get("line", -1) != 0:
                return _with_escalation(
                    "Error: expected_line is required (unless line=0, appending "
                    "at the very end) — pass the EXACT current text of the line "
                    "you're inserting before, verbatim from a recent "
                    "read_file_range/read_file result. This tool refuses to "
                    "insert blind: if the file shifted since you last read it "
                    "(e.g. from an earlier edit to it this same turn), re-read "
                    "it with read_file_range first to get the current line "
                    "number and text, then retry with expected_line set.",
                    None,
                )
            content, artifact = await original_coroutine(**kwargs)
            return _with_escalation(content, artifact)
    else:  # replace_lines, copy_lines
        async def _call(**kwargs):
            if not kwargs.get("expected_first_line") or not kwargs.get("expected_last_line"):
                return _with_escalation(
                    "Error: expected_first_line AND expected_last_line are both "
                    "required — pass the EXACT current text of the first and "
                    "last line of the range, verbatim from a recent "
                    "read_file_range/read_file result. This tool refuses to "
                    "edit blind: if the file shifted since you last read it "
                    "(e.g. from an earlier edit to it this same turn), re-read "
                    "it with read_file_range first to get the current line "
                    "numbers and text, then retry with both set.",
                    None,
                )
            content, artifact = await original_coroutine(**kwargs)
            return _with_escalation(content, artifact)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _read_lines_from_disk(path: str) -> list[str] | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()]
    except OSError:
        return None


def _record_read_range(cache: dict, tool_name: str, kwargs: dict, content_text: str) -> None:
    """Populates `cache` — see _cache_line_content_tool below for why this
    exists. Skips on error results and on _dedupe_read_tool's cache-hit
    stand-in text (that call fetched nothing new; whatever's already cached
    for this path is still the accurate, most recent read)."""
    if not content_text or content_text.lower().startswith(("error", "mcp error")):
        return
    if content_text.startswith("(You already read"):
        return
    if tool_name == "read_multiple_files":
        for p in kwargs.get("paths") or []:
            lines = _read_lines_from_disk(p)
            if lines is not None:
                cache[p] = {"start": 1, "end": len(lines), "lines": lines}
        return
    path = kwargs.get("path")
    if not path:
        return
    lines = _read_lines_from_disk(path)
    if lines is None:
        return
    total = len(lines)
    if tool_name == "read_file_range":
        start = kwargs.get("start_line", 1)
        end = min(kwargs.get("end_line", total), total)
    else:  # read_file, read_text_file — possibly head=/tail=, else whole file
        tail = kwargs.get("tail")
        head = kwargs.get("head")
        if tail:
            start, end = max(1, total - tail + 1), total
        elif head:
            start, end = 1, min(head, total)
        else:
            start, end = 1, total
    if start < 1 or end < start:
        return
    cache[path] = {"start": start, "end": end, "lines": lines[start - 1:end]}


def _cache_line_content_tool(tool: BaseTool, cache: dict) -> BaseTool:
    """Populates `cache` (agent_builder.py: line_content_cache, per-turn like
    read_history) with the ACTUAL current lines covered by every successful
    read_file/read_text_file/read_file_range/read_multiple_files call —
    consumed by _autofill_expected_lines below to fill replace_lines/
    insert_lines/copy_lines' expected_*_line params from the model's own
    recent read instead of trusting it to retype that text correctly.

    Reads the file itself, directly, rather than re-parsing the tool's
    returned text — the returned text may be truncated (_cap_tool_output) or
    reformatted, and re-parsing "N. content" back into raw lines is one more
    place a stray line that happens to start with digits+'.' could confuse a
    regex; opening the file is simpler and exactly as trustworthy (the
    underlying MCP tool just did the same read moments earlier, on the same
    local filesystem)."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        content, artifact = await original_coroutine(**kwargs)
        _record_read_range(cache, tool.name, kwargs, _content_text(content))
        return content, artifact

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _lookup_cached_lines(cache: dict, path: str, lo: int, hi: int) -> list[str] | None:
    entry = cache.get(path)
    if not entry or lo < entry["start"] or hi > entry["end"]:
        return None
    off = entry["start"]
    return entry["lines"][lo - off:hi - off + 1]


def _autofill_expected_lines(tool: BaseTool, cache: dict) -> BaseTool:
    """Live runs (theme-toggle task, twice): the model reliably FOUND the
    right line range (read_file_range right before editing, sometimes even
    on the correct line) but unreliably TRANSCRIBED expected_first_line/
    expected_last_line/expected_line into the next call — passing the whole
    multi-line new_content block instead, or the whole old block, or (after
    a rejection) just omitting the field outright, sometimes 3+ times in a
    row without ever converging on 'the one line the tool itself just
    showed you'. _require_expected_lines' escalating hint helps but can't
    fix a model that isn't reliably copying text between messages at all.

    Since the exact current text at any [start_line, end_line]/line the
    model JUST read this turn is already known — from _cache_line_content_
    tool above, itself read straight off disk — there's no need to trust a
    retyped copy of it when one is available: this wrapper overwrites
    whatever the model passed (right, wrong, or empty) with the cached
    text whenever the requested range is fully covered by the model's own
    most recent read of that path. This does not weaken the staleness
    check _require_expected_lines/fs_extra_server.py's own comparison
    still perform — `cache` is invalidated (see _wrap_read_invalidation in
    agent_builder.py) on every successful write to the path, so a hit here
    always reflects a read that is still current, not a stale one; a MISS
    (range not covered, or covered by a since-invalidated entry) falls
    through unchanged to _require_expected_lines, which still refuses to
    edit blind and asks for a fresh read_file_range. Must run AFTER
    _require_expected_lines in the wrapping order (i.e. applied to the
    tool list AFTER it) so this filled-in value reaches that gate BEFORE
    it decides whether expected_*_line is missing."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool
    path_key = "source_path" if tool.name == "copy_lines" else "path"

    async def _call(**kwargs):
        path = kwargs.get(path_key)
        if path:
            if tool.name == "insert_lines":
                line = kwargs.get("line")
                if line:
                    hit = _lookup_cached_lines(cache, path, line, line)
                    if hit:
                        kwargs = {**kwargs, "expected_line": hit[0]}
            else:  # replace_lines, copy_lines
                start_line, end_line = kwargs.get("start_line"), kwargs.get("end_line")
                if start_line is not None and end_line is not None:
                    hit = _lookup_cached_lines(cache, path, start_line, end_line)
                    if hit:
                        kwargs = {**kwargs, "expected_first_line": hit[0], "expected_last_line": hit[-1]}
        return await original_coroutine(**kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def _bind_constant_args(tool: BaseTool, constants: dict) -> BaseTool:
    """Убирает ключи `constants` из схемы, которую видит модель для `tool`, и
    подставляет их значения в коде при реальном вызове.

    mcp-server-git требует `repo_path` в КАЖДОМ вызове, хотя сам процесс уже
    запущен с этим путём через `-r` (см. build_mcp_connections) — модель на
    живом прогоне дважды подряд подставляла туда плейсхолдер '/path/to/repo'
    вместо реального пути. Раз значение известно коду заранее и не меняется
    в рамках сессии, модели незачем вообще его видеть — тогда ошибиться в
    нём физически невозможно, а не просто "менее вероятно"."""
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
        # практике подставляет ключи, которых нет в схеме вообще (см. живой
        # прогон — придумала 'path' для git_status, которого не было ни в
        # исходной, ни в урезанной схеме). Если она когда-нибудь угадает и
        # сам constants-ключ (например repo_path) — без этой строки
        # получаем `TypeError: got multiple values for keyword argument`,
        # так как он придёт и в kwargs от модели, и в constants от кода.
        # Код должен побеждать безусловно — модель этот ключ не видит и не
        # должна на него влиять.
        for k in constants:
            kwargs.pop(k, None)
        return await original_coroutine(**kwargs, **constants)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=reduced_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


