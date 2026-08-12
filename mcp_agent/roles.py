"""
Реестр ролей пайплайна Router -> Analyzer -> Planner -> Coder -> Verifier
(mcp_agent/pipeline.py) — единый источник правды, какой роли какие тулы,
approval-список и лимиты положены. Router здесь не описан: он вообще не
получает MCP-тулов (см. mcp_agent/router.py — casual/snippet-ответ строится
как create_agent(model, tools=[], ...), тот же рецепт, что и voice_mode в
agent_builder.py).

Тулы роли — НЕ статичное множество на роль (это и была исходная проблема,
см. router.py про переход на флаги): ниже — capability-группы (что тул
ДЕЛАЕТ: читает проект, гоняет shell, пишет) и функции-компоновщики
(investigator_tools/planner_tools/executor_tools/coder_tools/
verifier_tools), собирающие конкретный набор из needs_project/needs_shell
(флаги router.py) на каждый вызов. pipeline.py вызывает нужную функцию с
текущими флагами и передаёт результат в agent_builder.py:_get_role_agent
как явный tool_names — роль больше не диктует ЕДИНСТВЕННЫЙ возможный набор
тулов, только промпт/бюджет попыток (ROLE_RECURSION_LIMIT/ROLE_MAX_ATTEMPTS
ниже, они не зависят от флагов, только от роли)."""
from mcp_agent.config import TOOLS_REQUIRING_APPROVAL
from mcp_agent.model_config import (
    ANALYZER_MAX_ATTEMPTS,
    ANALYZER_RECURSION_LIMIT,
    CODER_MAX_ATTEMPTS,
    CODER_RECURSION_LIMIT,
    PLANNER_MAX_ATTEMPTS,
    PLANNER_RECURSION_LIMIT,
    QUICK_FIX_MAX_ATTEMPTS,
    QUICK_FIX_RECURSION_LIMIT,
    VERIFIER_MAX_ATTEMPTS,
    VERIFIER_RECURSION_LIMIT,
)

# Явный allowlist по ИМЕНИ тула, а не "всё, что не в TOOLS_REQUIRING_APPROVAL"
# — так подмножество не меняется молча, если кто-то добавит новый
# read-only-с-виду тул, не подумав про инвестигатор/delegate (тот же
# принцип, что раньше был только в delegate_tool.py).
_PROJECT_READ_TOOLS = {
    # filesystem (@modelcontextprotocol/server-filesystem)
    "read_file", "read_text_file", "read_multiple_files",
    "list_directory", "directory_tree", "search_files", "get_file_info",
    # code_search_server.py — project_tree предпочтительнее directory_tree
    # на реальном проекте (исключает vendor/node_modules/.venv и т.п.,
    # см. её докстринг), но саму directory_tree не убираем — на маленьких
    # подкаталогах без vendor-мусора разницы нет, а модель уже про неё
    # знает из промпта.
    "search_code", "read_file_range", "find_files_by_name", "search_symbols", "project_tree",
    # git (mcp-server-git) — только чтение состояния
    "git_status", "git_diff", "git_diff_staged", "git_diff_unstaged", "git_log", "git_show",
    # lsp_server.py
    "lsp",
    # rag_server.py — семантический поиск, не reindex/remember_url (запись)
    "search_code_semantic", "search_dialog_history", "search_external_sources",
    # knowledge/memory — только чтение
    "get_knowledge", "list_memory",
}

# ОТКЛЮЧЕНЫ по умолчанию (не удалены — MCP-серверы их всё равно
# регистрируют, _get_tools() их всё равно грузит, просто ни один компоновщик
# ниже их больше не включает): тонкие обёртки над ОДНОЙ известной shell-
# командой (`git status`, `git log`, `git show`, `ls`, `find -name`, `stat`)
# без структурной пользы дедикейтед-тула — не правят файлы (нет риска
# "путает строки", который оправдывает read_file_range/replace_lines), не
# нуждаются в песочнице (это чтение состояния, не запись), не участвуют в
# дедупе read_history построчно. bash_exec справляется с той же самой
# командой ровно так же надёжно — лишний почти-дубль в списке тулов только
# повышает шанс путаницы при выборе (см. живой баг search_files vs
# find_files_by_name). Исключаются из состава там, где bash_exec
# гарантированно доступен как замена (investigator/planner/executor — с тех
# пор как _SHELL_TOOLS там безусловны, см. investigator_tools; Verifier — у
# него _SHELL_TOOLS всегда были безусловны) — Coder их не теряет НИКОГДА, у
# него bash_exec нет в принципе (verifier_tools про "верификация целиком у
# Verifier"), и без него это была бы чистая потеря возможностей, не замена.
# Если понадобится вернуть — просто убрать вычитание там, где оно применено.
_THIN_WRAPPER_TOOLS = {
    "git_status", "git_log", "git_show",
    "list_directory", "find_files_by_name", "get_file_info",
}

# bash_exec/bash_exec_bg* всё равно в TOOLS_REQUIRING_APPROVAL (config.py) —
# approval-диалог перед реальным запуском команды не меняется от того, кто
# именно из ролей её просит и был ли этот тул у роли "предусмотрен" изначально.
_SHELL_TOOLS = {"bash_exec", "bash_exec_bg", "bash_exec_bg_check", "bash_exec_bg_list"}

# Всегда доступны независимо от флагов — read-only, дешёвые, почти никогда
# не вредны как контекст. analyze_image (vision_server.py) добавлен сюда,
# а не в generation-тулы (image_gen/music/gen_model, гейтятся отдельно
# settings.gen_agent_tools, см. config.py:build_mcp_connections) — он
# ЧИТАЕТ картинку (для скриншотов/сгенерённых ассетов), а не создаёт её,
# так что засорения тул-листа тем же смыслом, что у генеративных тулов,
# здесь нет; было ранее вообще не подключено ни к одной роли — исправлено.
_WEB_TOOLS = {"web_search", "fetch", "analyze_image"}

# Пишущие/git-мутирующие тулы.
_WRITE_TOOLS = {
    "write_file", "edit_file", "create_directory", "move_file",
    "replace_lines", "insert_lines", "copy_lines",
    "delete_path", "restore_deleted_path", "list_deleted_paths",
    "git_restore_file", "restore_file_snapshot", "list_file_snapshots",
    "git_commit", "git_add", "git_reset", "git_create_branch", "git_checkout", "git_branch",
}


def _project_read_tools(has_shell: bool) -> set[str]:
    """_THIN_WRAPPER_TOOLS вычитается ТОЛЬКО когда у той же роли ЕСТЬ
    bash_exec (has_shell=true) — тогда замена (`bash_exec("git status")`)
    гарантированно доступна, реальной потери возможностей нет. Роли БЕЗ
    bash_exec (executor_tools, coder_tools) держат их полностью — там это
    единственный способ получить git_status/list_directory/
    find_files_by_name/get_file_info, не почти-дубль лишнего варианта."""
    return _PROJECT_READ_TOOLS - _THIN_WRAPPER_TOOLS if has_shell else set(_PROJECT_READ_TOOLS)


def investigator_tools(needs_project: bool) -> set[str]:
    """Read/observe-only набор — используется и когда инвестигатор питает
    Planner дальше (needs_change+ambiguous), и когда его саммари сразу
    становится финальным ответом (не needs_change, старое kind="explain") —
    в обоих случаях НИКТО не идёт проверять его работу отдельным взглядом
    после, так что самому иметь bash_exec тут безопасно и нужно.

    _SHELL_TOOLS (bash_exec) — БЕЗУСЛОВНЫ, больше не зависят от needs_shell.
    Живой инцидент: needs_shell — это классификация ДО единого вызова тула,
    та же квантованная модель, что и основной чат, ошибается — investigator
    с needs_shell=false, но реально нуждавшийся в команде, застревал в
    ретраях без единого способа восстановиться В ЭТОМ ЖЕ раунде (эскалация
    "дай мне другой набор тулов и попробуй снова" — отдельная, куда более
    инвазивная фича, трогающая retry-цикл stage_runner.py). approval-диалог
    на каждый реальный вызов bash_exec (TOOLS_REQUIRING_APPROVAL) уже
    защищает не хуже, чем отсутствие тула в схеме — так что цена ошибки
    needs_shell в любую сторону теперь минимальна, а не фатальна.
    needs_shell остаётся флагом РОУТЕРА (решает, входить ли в investigator
    вообще вместо остаться в casual — см. pipeline.py), просто больше не
    параметр этой функции."""
    names = set(_WEB_TOOLS) | set(_SHELL_TOOLS)
    if needs_project:
        names |= _project_read_tools(has_shell=True)
    return names


def planner_tools(needs_project: bool) -> set[str]:
    # ask_user — единственный "пишущий" тул Planner'а, обязателен для
    # согласования плана перед тем, как отдать его Coder'у.
    return investigator_tools(needs_project) | {"ask_user"}


def executor_tools(needs_project: bool) -> set[str]:
    """Investigate+write в ОДНОЙ стадии — для needs_change с
    change_is_ambiguous=false (старое kind="quick_fix"): читает, что нужно,
    и сразу правит, без отдельного Planner/ask_user. pipeline.py форсит
    needs_project=true всякий раз, когда needs_change=true (правка ЭТОГО
    проекта не бывает без его чтения), так что на практике этот набор
    всегда включает _PROJECT_READ_TOOLS.

    БЕЗ bash_exec — в отличие от investigator_tools, не наследует
    _SHELL_TOOLS. Живой баг: quick_fix унаследовал безусловный bash_exec от
    investigator_tools (эта функция раньше звалась изнутри неё), а
    собственный промпт (_quick_fix_system_prompt) всё ещё прямо говорил "you
    do NOT have bash_exec" — тот же класс несоответствия схема/промпт, что
    уже чинили для Analyzer. Смысловая причина держать его БЕЗ shell та же,
    что у coder_tools(): после quick_fix ВСЕГДА идёт Verifier (pipeline.py
    запускает verifier после coder_role независимо от того, "coder" это или
    "quick_fix") — самопроверка исполнителя противоречит смыслу отдельной
    непредвзятой проверки."""
    names = set(_WEB_TOOLS) | _WRITE_TOOLS
    if needs_project:
        names |= _project_read_tools(has_shell=False)
    return names


def coder_tools() -> set[str]:
    """Двустадийный путь (после согласованного Planner'ом плана) — всегда
    полный проектный read+write, но НИКОГДА bash_exec независимо от флагов:
    верификация целиком у Verifier (см. verifier_tools) — Coder не должен
    иметь возможность сам себя "проверить".

    mark_plan_step_current — только здесь: единственная стадия, у которой
    вообще есть пронумерованный plan_steps-чек-лист над футером (см.
    pipeline.py/ui/app.py) — Planner/Analyzer/Verifier/quick_fix его не
    рисуют, им нечего отмечать текущим."""
    return _project_read_tools(has_shell=False) | _WEB_TOOLS | _WRITE_TOOLS | {"mark_plan_step_current"}


def verifier_tools() -> set[str]:
    """Безусловно, не зависит от флагов исходного запроса — проверка правки
    всегда нуждается и в актуальном состоянии проекта, и в способности
    реально что-то запустить (тесты/линтер), независимо от того, что было
    needs_shell у САМОГО запроса. _THIN_WRAPPER_TOOLS вычитается — bash_exec
    здесь безусловно есть, так что реальной потери возможностей нет."""
    return _project_read_tools(has_shell=True) | _WEB_TOOLS | _SHELL_TOOLS


def filter_tools(names: set[str], tools: list) -> list:
    """Фильтрует уже загруженный общий список тулов (agent_builder.py:
    _get_tools, один подъём MCP-серверов на процесс) под конкретный набор
    имён — вызывающий код (pipeline.py) сам решает набор через функции
    выше, roles.py больше не хранит единственное статичное соответствие
    роль->тулы."""
    return [t for t in tools if t.name in names]


def approval_tools(names: set[str]) -> list[str]:
    """Пересечение набора тулов с общим TOOLS_REQUIRING_APPROVAL (config.py)
    — не отдельный список на роль/набор (риск разъехаться с общим), а
    вычисление от него."""
    return [name for name in TOOLS_REQUIRING_APPROVAL if name in names]


# Фиксированный набор без shell — путь легаси монолитного агента
# (delegate_tool.py: сабагент delegate, вызывается из mcp_agent/agent.py, а
# не из pipeline.py) не участвует в флагах router.py вообще, поэтому не
# может композировать набор per-call — используем тот же (project read +
# web, БЕЗ shell), что и раньше был в ANALYZER_TOOL_NAMES перед переходом
# на композицию. Собран напрямую (не через investigator_tools — та с тех
# пор как _SHELL_TOOLS у неё безусловны, всегда возвращает bash_exec, а
# delegate's собственный системный промпт до сих пор явно говорит "no
# shell") — этот набор держит то утверждение верным, никакого
# поведенческого изменения тут нет, в отличие от investigator_tools выше.
# _THIN_WRAPPER_TOOLS НЕ вычитаются — без bash_exec замены им тут нет.
LEGACY_INVESTIGATION_TOOL_NAMES = set(_WEB_TOOLS) | _PROJECT_READ_TOOLS

ROLE_RECURSION_LIMIT: dict[str, int] = {
    "analyzer": ANALYZER_RECURSION_LIMIT,
    "planner": PLANNER_RECURSION_LIMIT,
    "coder": CODER_RECURSION_LIMIT,
    "verifier": VERIFIER_RECURSION_LIMIT,
    "quick_fix": QUICK_FIX_RECURSION_LIMIT,
}

ROLE_MAX_ATTEMPTS: dict[str, int] = {
    "analyzer": ANALYZER_MAX_ATTEMPTS,
    "planner": PLANNER_MAX_ATTEMPTS,
    "coder": CODER_MAX_ATTEMPTS,
    "verifier": VERIFIER_MAX_ATTEMPTS,
    "quick_fix": QUICK_FIX_MAX_ATTEMPTS,
}
