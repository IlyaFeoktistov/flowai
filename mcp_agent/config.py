"""
Конфигурация MCP-серверов и permission-маппинга для нового агента.

Источники серверов:
  filesystem  — @modelcontextprotocol/server-filesystem (официальный, npm)
  git         — mcp-server-git (официальный, PyPI)
  fetch       — mcp-server-fetch (официальный, PyPI) — замена tools/read_page.py
  bash_exec   — свой (готового с нужной permission-гранулярностью нет)
  web_search  — свой (готового под self-hosted SearXNG нет)
  memory      — свой (готовый community "Memory"-сервер — другая модель
                данных, knowledge graph, не совместимая с нашим форматом)
  knowledge   — свой (структурированная база знаний о ПРОЕКТЕ, отдельно
                от memory — там плоские факты о ПОЛЬЗОВАТЕЛЕ)
  rag         — свой (семантический поиск по коду/докам, истории диалогов
                и сохранённым внешним страницам — эмбеддинги + свой
                stdlib-векторный индекс, готового под этот формат нет)
  git_extra   — свой (git-операции, которых нет в mcp-server-git: тот
                умеет откатывать только ВЕТКИ (git_checkout), а не отдельный
                файл — см. git_extra_server.py про живой инцидент, который
                это выявил)
  fs_extra    — свой (filesystem-server не умеет удалять вообще ничего —
                delete_path закрывает это "мягким" удалением через корзину,
                а не permanent rm)
  lsp         — свой (семантическая навигация по коду через настоящий
                Language Server Protocol — goToDefinition/findReferences/
                hover/documentSymbol/etc. — вместо grep-угадайки
                search_symbols; см. lsp_server.py про то, какие языковые
                сервера установлены и почему)
  vision      — свой (analyze_image через отдельную vision-модель Ollama,
                settings.vision_model — chat_model эту роль не выполняет)
  music       — свой (generate_music через MusicGen/HF transformers,
                CPU-only — не соревнуется за VRAM с Ollama/SDXL)
  gen_model   — свой (generate_3d_model/animate_3d_model/
                generate_texture_for_model — image-to-3D + риг + анимация +
                перегенерация текстуры на готовом mesh, см.
                gen3d/pipeline.py; сам сервер лёгкий
                и живёт в этом же venv, но шеллится в ТРИ отдельных
                venv/сервиса (vendor/hunyuan3d-2gp, vendor/unirig,
                vendor/animato) — их несовместимые torch/CUDA-сборки не
                дают поставить это всё в один venv, см. setup.py)

Permission-политика (сознательное упрощение относительно текущего
tools/confirm.py, где даже read_file/list_dir спрашивали подтверждение):
только ПИШУЩИЕ/ИСПОЛНЯЮЩИЕ операции требуют approval. Чтение (read_file,
list_directory, git_status, git_diff, git_log, web_search, fetch) идёт без
диалога — так ведёт себя большинство современных coding-агентов, и это
единственная причина реального риска (запись/удаление/выполнение
команд/git-мутации), а не сам факт чтения.
"""
import os
import shlex
import sys
import tempfile

import settings

# ВАЖНО: это каталог УСТАНОВКИ flowAI (где лежит сам config.py), а не место,
# откуда пользователь запустил `flowai`. Раньше repo_path по умолчанию
# резолвился именно сюда — из-за этого агент всегда "видел" исходники самого
# flowAI, даже когда пользователь запускал flowai из совершенно другого
# проекта (например, PHP-репозитория). Оставлен только как явное имя для
# документации/дебага, НЕ используется как дефолт repo_path — см. ниже.
_FLOWAI_INSTALL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOG_DIR = os.path.join(tempfile.gettempdir(), "flowai-mcp-logs")


def _via_shell(command: str, args: list[str], log_name: str) -> tuple[str, list[str]]:
    """Каждый MCP-сервер пишет свои внутренние логи в stderr — Python `mcp`
    SDK логирует входящие запросы (mcp/server/lowlevel/server.py), Node
    filesystem-сервер пишет стартовый баннер и Roots-warning прямо
    console.error(). stderr наследуется от родителя (нашего TUI-процесса) и
    лезет прямо в Rich/prompt_toolkit рендер, ломая экран.

    langchain_mcp_adapters не даёт перенастроить это через конфиг — его
    stdio_client() жёстко использует sys.stderr, никакого errlog= в
    StdioConnection нет. Редиректим на уровне самого запуска подпроцесса
    стандартным shell '>': трогаем только stderr, stdout не трогаем — там
    живёт сам MCP JSON-RPC протокол, его нельзя портить."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_path = os.path.join(_LOG_DIR, f"{log_name}.log")
    full_cmd = " ".join(shlex.quote(p) for p in [command, *args])
    return "bash", ["-c", f"exec {full_cmd} 2>{shlex.quote(log_path)}"]

# Тулы, требующие подтверждения пользователя перед выполнением.
TOOLS_REQUIRING_APPROVAL = [
    # filesystem — пишущие
    "write_file", "edit_file", "create_directory", "move_file",
    # git — мутирующие состояние репозитория
    "git_commit", "git_add", "git_reset", "git_create_branch", "git_checkout",
    # свои — update_knowledge НЕ здесь: это текстовый факт о ПРОЕКТЕ,
    # который легко переписать, не деструктивная операция как остальные в
    # этом списке — approval на него был лишним барьером, из-за которого
    # модель (и без того ненадёжно вспоминающая об этом тулe вообще, см.
    # mcp_agent/knowledge.py) звала его за всю историю проекта 2 раза.
    "bash_exec", "bash_exec_bg", "update_memory", "generate_image", "edit_image", "generate_music", "remember_url",
    "generate_3d_model", "animate_3d_model", "generate_texture_for_model",
    "git_restore_file", "restore_file_snapshot",
    "delete_path", "restore_deleted_path", "replace_lines", "copy_lines",
    # insert_lines was missing here — same write risk as replace_lines/
    # copy_lines, but was slipping through with zero approval prompt while
    # its siblings all asked. Live bug: a Coder round inserted 26 lines
    # (including a duplicated import block, see the content-duplication
    # guard in fs_extra_server.py) into the user's repo with no confirmation
    # at all, then asked for approval on the very next replace_lines call —
    # same file, same round, inconsistent gate.
    "insert_lines",
]


def build_mcp_connections(repo_path: str | None = None) -> dict:
    # flowai-лончер запускает venv-python по абсолютному пути БЕЗ `cd` —
    # os.getcwd() честно отражает директорию, откуда пользователь реально
    # вызвал команду (его собственный проект), а не место установки flowAI.
    repo_path = repo_path or os.getcwd()
    py = sys.executable
    venv_bin = os.path.dirname(py)

    # mcp-server-git/mcp-server-fetch — консольные скрипты, установленные pip
    # в .venv/bin/ рядом с самим интерпретатором. Голое имя команды работает
    # только если .venv/bin есть в PATH (например, после `source
    # .venv/bin/activate`) — а лончер flowai запускает venv-python по
    # абсолютному пути БЕЗ активации venv и без правки PATH, так что бинарник
    # не находится (ENOENT). Резолвим абсолютный путь тем же способом, что
    # уже используется для sys.executable ниже.
    mcp_server_git = os.path.join(venv_bin, "mcp-server-git")
    mcp_server_fetch = os.path.join(venv_bin, "mcp-server-fetch")

    # `python -m mcp_agent.servers.X` резолвит пакет через sys.path, в который
    # Python неявно добавляет ТЕКУЩУЮ ДИРЕКТОРИЮ подпроцесса (а не место, где
    # физически лежит пакет). Пока cwd совпадал с каталогом установки flowAI,
    # это работало случайно; теперь cwd = repo_path пользователя (см. ниже),
    # и "-m" не находит mcp_agent → сервер падает при старте ("unhandled
    # errors in a TaskGroup" — subprocess умирает почти сразу после запуска).
    # Запуск по абсолютному пути к файлу делает резолвинг независимым от cwd.
    servers_dir = os.path.join(_FLOWAI_INSTALL_DIR, "mcp_agent", "servers")

    def _own_server(name: str) -> str:
        return os.path.join(servers_dir, f"{name}_server.py")

    raw_servers = {
        # Живой инцидент: пользователь попросил модель выйти за пределы
        # repo_path в соседний проект — filesystem-сервер отказал жёстко
        # ("Access denied"), ни единого шанса спросить разрешения, а модель
        # вместо честного "не могу выйти за пределы проекта" 30 минут
        # бесцельно копалась в НЕПРАВИЛЬНОМ (текущем) репозитории.
        # Прямое решение пользователя: ЧТЕНИЕ разрешено везде, до корня "/",
        # без единого approval-вопроса (риск читать — намного ниже риска
        # писать); ЗАПИСЬ вне repo_path остаётся под approval — см.
        # mcp_agent/ask_user_tool.py:_OutOfProjectWriteApprovalMiddleware,
        # она перехватывает каждый пишущий вызов ДО того, как он дойдёт до
        # этого сервера, и спрашивает через ask_permission, если целевой
        # путь вне repo_path. "/" здесь покрывает и repo_path — второй
        # аргумент избыточен, но оставлен явно как документация того, что
        # исходная (узкая) граница раньше была именно им.
        "filesystem": ("npx", ["-y", "@modelcontextprotocol/server-filesystem", repo_path, "/"]),
        "git": (mcp_server_git, ["-r", repo_path]),
        "fetch": (mcp_server_fetch, []),
        "bash_exec": (py, [_own_server("bash_exec")]),
        "web_search": (py, [_own_server("web_search")]),
        "memory": (py, [_own_server("memory")]),
        "knowledge": (py, [_own_server("knowledge")]),
        "rag": (py, [_own_server("rag")]),
        "code_search": (py, [_own_server("code_search")]),
        "image_gen": (py, [_own_server("image_gen")]),
        "vision": (py, [_own_server("vision")]),
        "music": (py, [_own_server("music")]),
        "gen_model": (py, [_own_server("gen_model")]),
        "git_extra": (py, [_own_server("git_extra")]),
        "fs_extra": (py, [_own_server("fs_extra")]),
        "lsp": (py, [_own_server("lsp")]),
    }

    # gen_agent_tools (settings.py) gates the AGENT's (LLM tool-calling)
    # access to generation tools, not the /gen /music /gen_model slash
    # commands — those import tools/image_gen.py, tools/gen_model.py, and
    # mcp_agent.servers.music_server directly in-process (cli.py) and never
    # go through this connections dict at all. "vision" (analyze_image) and
    # "web_search"/"fetch" are deliberately NOT in this set — they're
    # read-only and useful regardless of whether the model is coding or
    # generating media (see roles.py's always-on tool set). Dropping these
    # entries entirely (not just filtering the tool NAMES downstream) skips
    # spawning their MCP subprocesses altogether — real startup time/VRAM
    # savings, not just a smaller tool schema.
    if not settings.get("gen_agent_tools"):
        for name in ("image_gen", "music", "gen_model"):
            raw_servers.pop(name, None)

    connections = {}
    for name, (command, args) in raw_servers.items():
        wrapped_command, wrapped_args = _via_shell(command, args, name)
        connections[name] = {
            "transport": "stdio",
            "command": wrapped_command,
            "args": wrapped_args,
            "cwd": repo_path,
        }
    return connections
