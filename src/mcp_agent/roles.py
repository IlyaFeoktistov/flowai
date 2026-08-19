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
import settings
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
from mcp_agent.optimized_tools import OPTIMIZED_TOOL_NAMES

# Явный allowlist по ИМЕНИ тула, а не "всё, что не в TOOLS_REQUIRING_APPROVAL"
# — так подмножество не меняется молча, если кто-то добавит новый
# read-only-с-виду тул, не подумав про инвестигатор/delegate (тот же
# принцип, что раньше был только в delegate_tool.py). read_file/grep_search/
# glob_search/list_deleted_paths — file_ops_server.py, все помечены
# "read_only" в его TOOL_PERMISSIONS (см. его модульный докстринг про
# required_permission) — заменяют разом filesystem MCP-сервер + старые
# code_search_server.py/fs_extra_server.py: никакого отдельного тула для
# листинга директории — glob_search("**/*", path) достаточен для того, чтобы
# увидеть структуру проекта, отдельный tree-тул не нужен.
# Никакого отдельного git-тула вообще (mcp-server-git/git_extra_server.py —
# УБРАНЫ, 2026-08-14: "зачем git-тулы, есть же bash") — git status/diff/log/
# show идут через bash (Analyzer/Planner's read-only allowlist уже пускает
# их, см. agent_builder.py:_is_read_only_bash_command), git-мутации
# (commit/checkout/reset/...) — тоже через bash, под тем же approval, что и
# любая другая команда (config.py:TOOLS_REQUIRING_APPROVAL).
_PROJECT_READ_TOOLS = {
    "read_file", "grep_search", "glob_search", "list_deleted_paths",
    # lsp_server.py
    "lsp",
    # rag_server.py — семантический поиск, не reindex/remember_url (запись)
    "search_code_semantic", "search_dialog_history", "search_external_sources",
    # rag_server.py — структурный (не семантический) обзор своей же истории:
    # список сессий и полный транскрипт одной из них
    "list_episodic_sessions", "read_episodic_session",
    # knowledge/memory — только чтение
    "get_knowledge", "list_memory",
}

# bash/bash_bg* всё равно в TOOLS_REQUIRING_APPROVAL (config.py) —
# approval-диалог перед реальным запуском команды не меняется от того, кто
# именно из ролей её просит и был ли этот тул у роли "предусмотрен" изначально.
_SHELL_TOOLS = {"bash", "bash_bg", "bash_bg_check", "bash_bg_list"}

# Всегда доступны независимо от флагов — read-only, дешёвые, почти никогда
# не вредны как контекст. analyze_image (vision_server.py) добавлен сюда,
# а не в generation-тулы (image_gen/music/gen_model, гейтятся отдельно
# settings.gen_agent_tools, см. config.py:build_mcp_connections) — он
# ЧИТАЕТ картинку (для скриншотов/сгенерённых ассетов), а не создаёт её,
# так что засорения тул-листа тем же смыслом, что у генеративных тулов,
# здесь нет; было ранее вообще не подключено ни к одной роли — исправлено.
_WEB_TOOLS = {"web_search", "fetch", "analyze_image"}

# flowai_guide (guide_server.py) — static self-description, no side
# effects, no project/repo dependency at all, so it's unioned in wherever
# _WEB_TOOLS is rather than gated by any needs_project/needs_shell flag —
# a user asking "what are you"/"what can you do" mid-investigation
# shouldn't need a whole extra round just because the role's tool set
# didn't happen to include it.
_META_TOOLS = {"flowai_guide"}

# Пишущие тулы. write_file/edit_file/delete_path — file_ops_server.py,
# "workspace_write" в его TOOL_PERMISSIONS. Никакого отдельного
# move/create_directory: write_file создаёт недостающие родительские
# директории сама (см. её докстринг), а move — это либо write_file(new_path,
# <прочитанное>)+delete_path(old_path), либо bash. git_restore_file убран
# вместе с остальными git-тулами — откат к состоянию git теперь идёт через
# bash (роли, у которых оно есть) или restore_file_snapshot (свой пре-write
# снапшот, не завязанный на git).
_WRITE_TOOLS = {
    "write_file", "edit_file", "delete_path", "restore_deleted_path",
    "restore_file_snapshot", "list_file_snapshots",
}


def _project_read_tools(has_shell: bool) -> set[str]:
    return set(_PROJECT_READ_TOOLS)


def _apply_optimized_filter(names: set[str]) -> set[str]:
    """settings.optimized_tools (see optimized_tools.py's own docstring)
    used to be scoped to ONLY the legacy monolithic agent — "работает
    только когда 'новый пайплайн' ВЫКЛ" (see ui/tui/settings.py's toggle
    hint, updated alongside this). Direct instruction (2026-08-14): same
    toggle should narrow every new-pipeline role's tool set too, not just
    legacy's. The file_ops_server.py consolidation (read_file/write_file/
    edit_file/grep_search/glob_search replacing a dozen near-duplicate
    filesystem/code-search tools) already closes most of the original
    complaint on its own — this filter still matters for the remaining
    near-duplicates (every git_* read tool bound alongside bash, which
    already covers `git diff`/`git log` just as reliably).

    Intersects rather than replaces the capability-group union each
    composer function below builds — applied to that raw union BEFORE any
    role adds its own always-needed extra (planner's ask_user, coder's
    mark_plan_step_current), so those never get filtered out even though
    mark_plan_step_current isn't itself in OPTIMIZED_TOOL_NAMES (ask_user
    is, redundantly, but the union order makes it moot either way)."""
    return names & OPTIMIZED_TOOL_NAMES if settings.get("optimized_tools") else names


def investigator_tools() -> set[str]:
    """Read/observe-only набор — используется и когда инвестигатор питает
    Planner дальше (needs_change+ambiguous), и когда его саммари сразу
    становится финальным ответом (не needs_change, старое kind="explain") —
    в обоих случаях НИКТО не идёт проверять его работу отдельным взглядом
    после, так что самому иметь bash тут безопасно и нужно.

    _SHELL_TOOLS (bash) и _PROJECT_READ_TOOLS — ОБА БЕЗУСЛОВНЫ, не
    зависят от needs_shell/needs_project — тот же класс проблемы
    проявлялся и для needs_shell, и для needs_project по отдельности:
    needs_shell/needs_project — это классификация РОУТЕРОМ ДО единого
    вызова тула, та же квантованная модель, что и основной чат, ошибается
    — investigator с needs_project=false, но реально нуждавшийся в
    project-тулах, не просто "не мог восстановиться в ретраях" — получал
    ЯВНУЮ инструкцию в промпте (см. pipeline.py:_investigator_scope_note,
    до этого фикса) "у тебя нет этих тулов, даже не пытайся" и послушно
    отвечал отказом на вопрос вроде "какие параметры стоят у модели этого
    проекта" вместо того, чтобы прочитать конфиг. approval-диалог на
    bash (TOOLS_REQUIRING_APPROVAL) и то, что остальные project-read
    тулы вообще read-only без approval, защищают не хуже, чем отсутствие
    тула в схеме — так что цена ошибки needs_shell/needs_project в любую
    сторону теперь минимальна, а не фатальна. Оба флага остаются флагами
    РОУТЕРА (needs_shell решает, входить ли в investigator вообще вместо
    остаться в casual; needs_project влияет на _investigator_scope_note и
    is_final_answer — см. pipeline.py), просто больше не параметры этой
    функции."""
    return _apply_optimized_filter(set(_WEB_TOOLS) | _META_TOOLS | set(_SHELL_TOOLS) | _project_read_tools(has_shell=True))


def planner_tools() -> set[str]:
    # ask_user — единственный "пишущий" тул Planner'а, обязателен для
    # согласования плана перед тем, как отдать его Coder'у.
    return investigator_tools() | {"ask_user"}


def executor_tools(needs_project: bool) -> set[str]:
    """Investigate+write в ОДНОЙ стадии — для needs_change с
    change_is_ambiguous=false (старое kind="quick_fix"): читает, что нужно,
    и сразу правит, без отдельного Planner/ask_user. pipeline.py форсит
    needs_project=true всякий раз, когда needs_change=true (правка ЭТОГО
    проекта не бывает без его чтения), так что на практике этот набор
    всегда включает _PROJECT_READ_TOOLS.

    БЕЗ bash — в отличие от investigator_tools, не наследует
    _SHELL_TOOLS. Раньше эта функция вызывалась изнутри investigator_tools
    и потому безусловно наследовала bash оттуда, хотя собственный промпт
    (_quick_fix_system_prompt) прямо говорил "you do NOT have bash" — тот
    же класс несоответствия схема/промпт, что уже чинили для Analyzer.
    Смысловая причина держать его БЕЗ shell та же,
    что у coder_tools(): после quick_fix ВСЕГДА идёт Verifier (pipeline.py
    запускает verifier после coder_role независимо от того, "coder" это или
    "quick_fix") — самопроверка исполнителя противоречит смыслу отдельной
    непредвзятой проверки."""
    names = set(_WEB_TOOLS) | _META_TOOLS | _WRITE_TOOLS
    if needs_project:
        names |= _project_read_tools(has_shell=False)
    return _apply_optimized_filter(names)


def coder_tools() -> set[str]:
    """Двустадийный путь (после согласованного Planner'ом плана) — всегда
    полный проектный read+write, но НИКОГДА bash независимо от флагов:
    верификация целиком у Verifier (см. verifier_tools) — Coder не должен
    иметь возможность сам себя "проверить".

    mark_plan_step_current — только здесь: единственная стадия, у которой
    вообще есть пронумерованный plan_steps-чек-лист над футером (см.
    pipeline.py/ui/app.py) — Planner/Analyzer/Verifier/quick_fix его не
    рисуют, им нечего отмечать текущим."""
    raw = _project_read_tools(has_shell=False) | _WEB_TOOLS | _META_TOOLS | _WRITE_TOOLS
    return _apply_optimized_filter(raw) | {"mark_plan_step_current"}


def verifier_tools() -> set[str]:
    """Безусловно, не зависит от флагов исходного запроса — проверка правки
    всегда нуждается и в актуальном состоянии проекта, и в способности
    реально что-то запустить (тесты/линтер), независимо от того, что было
    needs_shell у САМОГО запроса. _THIN_WRAPPER_TOOLS вычитается — bash
    здесь безусловно есть, так что реальной потери возможностей нет."""
    return _apply_optimized_filter(_project_read_tools(has_shell=True) | _WEB_TOOLS | _META_TOOLS | _SHELL_TOOLS)


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
# пор как _SHELL_TOOLS у неё безусловны, всегда возвращает bash, а
# delegate's собственный системный промпт до сих пор явно говорит "no
# shell") — этот набор держит то утверждение верным, никакого
# поведенческого изменения тут нет, в отличие от investigator_tools выше.
# _THIN_WRAPPER_TOOLS НЕ вычитаются — без bash замены им тут нет.
LEGACY_INVESTIGATION_TOOL_NAMES = set(_WEB_TOOLS) | _META_TOOLS | _PROJECT_READ_TOOLS

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
