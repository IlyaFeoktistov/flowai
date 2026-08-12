"""
"Оптимизированный" набор тулов для СТАРОГО (legacy mcp_agent/agent.py)
агента — settings.optimized_tools (см. settings.py/ui/tui/settings.py).
Только урезание списка, БЕЗ переименования: тулы остаются под своими
родными MCP-именами (write_file/edit_file/bash_exec/read_file_range/...) —
self_heal.py, config.py:TOOLS_REQUIRING_APPROVAL, ask_user_tool.py,
compaction.py и весь остальной код, matching по имени тула, продолжают
работать без единой правки. Переименование в духе Claude Code (Bash/Read/
Write/...) обсуждалось и было отклонено — цепочка мест, matching'ящих по
буквальному имени тула (approval-гейт HumanInTheLoopMiddleware в первую
очередь — тихая потеря approval на Bash/Write было бы security-регрессией,
не просто багом), сделала бы риск несоразмерным цели "покороче список".

Смысл в другом: и Analyzer, и Coder, и voice_mode-агент (когда он всё же
получает тулы) в легаси-пути видят ВСЕ ~60 тулов сразу, независимо от
задачи (см. router.py/roles.py — ровно то, чего у легаси-агента НЕТ и что
уже чинит новый пайплайн). Этот флаг — узкая, отдельная от пайплайна мера:
статичный (не per-запрос, без classify_intent) урезанный список в духе
Claude Code — по одному тулу на "смысл" (bash/read/grep/glob/write), но
write здесь НЕ один: write_file (целиком) + replace_lines/insert_lines/
copy_lines (точечно, по номерам строк) — БЕЗ edit_file, см.
_CORE_TOOL_NAMES про то, почему именно edit_file исключён, а точечные
line-based тулы оставлены. НЕ трогает новый пайплайн
(mcp_agent/pipeline.py/roles.py) — там своя, per-request композиция уже
решает эту же задачу иначе.

git-тулы (git_status/git_diff/git_commit/...) исключены целиком — Claude
Code не имеет отдельных git-тулов вообще, всё идёт через Bash; здесь то же
самое — git-операции в этом режиме идут через bash_exec("git ...").
"""

# Один тул на "смысл", без дублей — see module docstring.
_CORE_TOOL_NAMES = {
    # bash — bash_exec_bg*/check/list не "диверсия" в смысле жалобы (это
    # разные РЕЖИМЫ запуска — sync/async, не 5 способов одного и того же),
    # остаются все.
    "bash_exec", "bash_exec_bg", "bash_exec_bg_check", "bash_exec_bg_list",
    # read — read_file_range один: путь + опциональный диапазон строк,
    # ближайший аналог Claude Code Read(file_path, offset, limit). Никаких
    # read_file/read_text_file/read_multiple_files рядом.
    "read_file_range",
    # grep — один поиск по содержимому, без search_symbols/search_code_semantic.
    "search_code",
    # glob — один поиск по имени файла, без search_files/list_directory/
    # directory_tree/project_tree (листинг директории — bash_exec("ls ...")).
    "find_files_by_name",
    # write — write_file (целиком) + точечные replace_lines/insert_lines/
    # copy_lines, БЕЗ edit_file. Живой прогон: qwen3-coder:30b на edit_file
    # регулярно проваливал byte-for-byte oldText-совпадение после
    # нескольких правок подряд (файл на диске расходился с тем, что модель
    # "помнила" о нём) — 2-3 провала "Could not find exact match", потом
    # сдавалась и переписывала файл целиком через write_file всё равно, но
    # только после нескольких потерянных раундов на пустые попытки.
    # replace_lines/insert_lines не страдают тем же — они адресуются по
    # номеру строки, а не по повторению существующего текста, так что тот
    # же класс провала для них не воспроизводится; убирать их вместе с
    # edit_file было бы перебором — точечная правка нужна для больших
    # файлов, где переписывать всё целиком через write_file дорого и
    # рискованно (шанс невольно потерять кусок при пересборке по памяти).
    "write_file", "replace_lines", "insert_lines", "copy_lines",
    # web
    "web_search", "fetch", "analyze_image",
    # ask_user — HITL, не тул с диверсией смысла.
}

# Знание о проекте (get_knowledge/update_knowledge) и память о пользователе
# (update_memory/list_memory) — отдельная от чтения/записи КОДА способность,
# не подпадает под жалобу "слишком много читающих/пишущих тулов".
_KNOWLEDGE_TOOL_NAMES = {"get_knowledge", "update_knowledge", "update_memory", "list_memory"}

# Генеративные тулы гейтятся ОТДЕЛЬНО через settings.gen_agent_tools
# (config.py:build_mcp_connections) — optimized_tools не должен вычитать то,
# что пользователь сам явно включил другим тумблером.
_GENERATION_TOOL_NAMES = {
    "generate_image", "edit_image", "unload_image_gen_model",
    "generate_music", "unload_music_gen_model",
    "generate_3d_model", "animate_3d_model", "generate_texture_for_model",
}

OPTIMIZED_TOOL_NAMES: frozenset[str] = frozenset(
    _CORE_TOOL_NAMES | _KNOWLEDGE_TOOL_NAMES | _GENERATION_TOOL_NAMES | {"ask_user"}
)


def build_optimized_tools(tools: list) -> tuple[list, dict]:
    """Фильтрует уже загруженный и обёрнутый список тулов (agent_builder.py:
    _get_tools — dedupe/verify-reminder/snapshot-обёртки уже применены к
    ЭТИМ же объектам по их родным именам, так что достаточно просто выбрать
    подмножество, ничего пересобирать не нужно) под OPTIMIZED_TOOL_NAMES.
    Возвращает (tools, tools_by_name) — та же форма, что _get_tools()."""
    filtered = [t for t in tools if t.name in OPTIMIZED_TOOL_NAMES]
    return filtered, {t.name: t for t in filtered}
