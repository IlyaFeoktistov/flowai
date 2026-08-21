"""
"Оптимизированный" набор тулов — settings.optimized_tools (см.
settings.py/ui/tui/settings.py). Только урезание списка, БЕЗ переименования:
тулы остаются под своими родными MCP-именами — self_heal.py,
config.py:TOOLS_REQUIRING_APPROVAL, ask_user_tool.py, compaction.py и весь
остальной код, matching по имени тула, продолжают работать без единой
правки.

Смысл: и Analyzer, и Coder, и voice_mode-агент (когда он всё же получает
тулы) в пути основного агента видят ВСЕ доступные тулы сразу, независимо от задачи
(см. router.py/roles.py — ровно то, чего у основного агента НЕТ и что уже
чинит новый пайплайн через per-request классификацию needs_project/
needs_shell). file_ops_server.py's консолидация (read_file/write_file/
edit_file/grep_search/glob_search вместо дюжины почти-дублей — filesystem
MCP-сервер + code_search_server.py + fs_extra_server.py) и удаление
отдельных git-тулов (git status/diff/log/commit/checkout идут через
bash("git ...") — отдельного git-сервера в проекте больше нет вообще, не
только в этом урезанном режиме) уже закрыли большую часть исходной жалобы
сами по себе — этот тумблер теперь в основном сужает генеративные тулы
(когда gen_agent_tools включён отдельным тумблером).

Действует и в новом пайплайне (mcp_agent/roles.py:_apply_optimized_filter).
"""

# Один тул на "смысл", без дублей — see module docstring.
_CORE_TOOL_NAMES = {
    # bash — bash_bg*/check/list не "диверсия" в смысле жалобы (это
    # разные РЕЖИМЫ запуска — sync/async, не 5 способов одного и того же),
    # остаются все.
    "bash", "bash_bg", "bash_bg_check", "bash_bg_list",
    # read/grep/glob — file_ops_server.py, уже по одному тулу на смысл, без
    # почти-дублей, ничего сужать не нужно.
    "read_file", "grep_search", "glob_search",
    # write — write_file (целиком) и edit_file (уникальная подстрока) —
    # ровно два инструмента с разным назначением, не почти-дубли друг друга.
    "write_file", "edit_file",
    # web
    "web_search", "fetch", "web_read", "analyze_image",
    # guide_server.py — static self-description, no generative/read/write
    # overlap with anything else here.
    "flowai_guide",
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


def build_optimized_tools(tools: list, extra_names: frozenset[str] = frozenset()) -> tuple[list, dict]:
    """Фильтрует уже загруженный и обёрнутый список тулов (agent_builder.py:
    _get_tools — dedupe/verify-reminder/snapshot-обёртки уже применены к
    ЭТИМ же объектам по их родным именам, так что достаточно просто выбрать
    подмножество, ничего пересобирать не нужно) под OPTIMIZED_TOOL_NAMES.
    extra_names — имена тулов плагинов (mcp_agent/plugins.py), неизвестные
    заранее и потому не входящие в статичный OPTIMIZED_TOOL_NAMES; без
    этого тумблер optimized_tools молча вырезал бы любой плагинский тул.
    Возвращает (tools, tools_by_name) — та же форма, что _get_tools()."""
    names = OPTIMIZED_TOOL_NAMES | extra_names
    filtered = [t for t in tools if t.name in names]
    return filtered, {t.name: t for t in filtered}
