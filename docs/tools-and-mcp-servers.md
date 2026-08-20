# Тулы и MCP-серверы

## Как это устроено

Почти каждый тул, доступный модели, — свой MCP-сервер (отдельный
подпроцесс), зарегистрированный в `mcp_agent/config.py:
build_mcp_connections()`. Исключения — тулы, которым нужен прямой доступ
к TUI-процессу (не выживают в изолированном подпроцессе):

- `ask_user`, `mark_plan_step_current` — `mcp_agent/ask_user_tool.py`
- `list_file_snapshots`/`restore_file_snapshot` — общее SQLite-хранилище
  снимков (`mcp_agent/snapshots.py`)
- D&D-тулы (`dnd_*`) — `mcp_agent/dnd_tools.py`, свои для каждой игровой
  сессии (замкнуты на `game_id`)
- `delegate` (`mcp_agent/delegate_tool.py`), `web_read`
  (`mcp_agent/web_read_tool.py`) — нужен уже поднятый `model` этого хода
  (переиспользуют резидентную модель, без второй загрузки весов), а не
  что-то, что можно поднять отдельным подпроцессом до выбора модели

Список серверов (все — свои реализации под этот конкретный локальный
GPU/Ollama-стек; нигде не найдено готового community-сервера под нужную
форму данных):

| Сервер | Тулы | Зачем свой, а не готовый |
|---|---|---|
| `fetch` | (официальный `mcp-server-fetch`, PyPI) | — сырой markdown страницы, кусками по 5000 симв. с пагинацией (`start_index`). См. `web_read` ниже — обычно предпочтительнее |
| `bash` | `bash`, `bash_bg`, `bash_bg_check`, `bash_bg_list` | нужна своя permission-гранулярность (auto-approve по первому слову команды). `bash` при превышении `timeout=` больше не убивает процесс — автоматически переводит его в фон (тот же реестр, что `bash_bg`) и отдаёт `job_id` для `bash_bg_check`, ничего не теряя (`bash_server.py`) |
| `file_ops` | `read_file`, `write_file`, `edit_file`, `grep_search`, `glob_search`, `delete_path`, `restore_deleted_path`, `list_deleted_paths` | заменяет разом filesystem + старые code_search/fs_extra серверы; своя корзина (не безвозвратное удаление). `write_file`/`edit_file` требуют свежего `read_file` по этому пути перед записью — отказ, если путь не читался в этой сессии или изменился на диске с момента чтения (`file_ops_server.py:_require_fresh_read`); модели в ответ уходит только короткое подтверждение ("Updated ..."), сам diff — в `structuredContent` (MCP), которое видит только UI (`ui/stream.py`), не модель |
| `web_search` | `web_search` | под self-hosted SearXNG |
| `memory` | `update_memory`, `list_memory` | плоские факты о ПОЛЬЗОВАТЕЛЕ, персистентно между сессиями |
| `knowledge` | `update_knowledge`, `get_knowledge` | категоризированные знания о ПРОЕКТЕ (архитектура/решения/конвенции), отдельно от памяти о пользователе. `get_knowledge(query=...)` — free-text поиск подстрокой по всем category/key/value сразу, без необходимости знать точное имя категории заранее (`category=` — точный фильтр, как раньше) |
| `rag` | `search_code_semantic`, `search_dialog_history`, `list_episodic_sessions`, `read_episodic_session`, `remember_url`, `search_external_sources` | семантический поиск (эмбеддинги + свой векторный индекс) по коду проекта, истории диалогов и сохранённым внешним страницам. Код-индекс модель не строит явным вызовом тула (нет такого тула вообще) — но он и не требует ручного `/reindex` для старта: каждый успешный `read_file`/`write_file`/`edit_file` тихо обновляет индекс ТОЛЬКО для этого файла в фоне (`mcp_agent/plugin_hooks.py:_auto_reindex_file`, никогда не блокирует сам тул-вызов). `/reindex [путь ...]` (`cli.py`, см. [commands.md](commands.md)) остаётся для полного/точечного прохода по всему дереву разом. Пока НИ один файл ещё не тронут — индекс пуст, и `search_code_semantic` прямо об этом говорит, переключая модель на `grep_search`/`glob_search`. Индекс всегда ЧАСТИЧНЫЙ (покрывает только уже тронутые файлы) — каждый результат несёт напоминание об этом, чтобы скудная/пустая выдача не читалась моделью как "этого нет в коде", только как "ещё не проиндексировано". Поддиректория, уже проиндексированная отдельно (например подпроект монорепо) — не переобходится повторным `/reindex`/автотриггером уровня выше, а подключается как живая ссылка (`rag/store.py:VectorStore.child_indexes`), поиск федеративно заходит и туда |
| `vision` | `analyze_image` | своя локальная vision-модель Ollama (`settings.vision_model`), отдельно от chat_model |
| `lsp` | `lsp` | настоящий Language Server Protocol (goToDefinition/findReferences/hover/...) вместо grep-угадайки |
| `guide` | `flowai_guide` | самоописание flowAI — см. ниже |
| `image_gen` | `generate_image`, `edit_image`, `unload_image_gen_model` | опционально, гейтится `gen_agent_tools` в `/settings` |
| `music` | `generate_music`, `unload_music_gen_model` | опционально, тот же тумблер |
| `gen_model` | `generate_3d_model`, `animate_3d_model`, `generate_texture_for_model` | опционально, тот же тумблер |

Отдельного git-тула нет вообще — `git status`/`diff`/`log`/`commit`/...
идут через `bash("git ...")`. Analyzer/Planner держат bash под read-only
allowlist (`agent_builder.py:_is_read_only_bash_command`), Verifier — под
denylist мутирующих команд.

## web_read

`mcp_agent/web_read_tool.py` — тул рядом с `fetch`, но не MCP-сервер: не
отдаёт сырую страницу модели, а фетчит URL (переиспользует HTML→markdown-
извлечение и robots.txt-проверку из уже установленного `mcp-server-fetch`,
без второй копии парсера), кладёт результат в ОТДЕЛЬНЫЙ, изолированный
вызов уже резидентной чат-модели вместе с конкретным вопросом (`question`)
и возвращает только её краткий ответ — сама страница (иногда десятки тысяч
символов, больше всего `OLLAMA_NUM_CTX`) в основной разговор не попадает
вообще. Самоочищающийся кэш по URL на 15 минут — повторный вопрос о той же
странице не идёт за ней в сеть заново. Добавлен во все роли пайплайна,
легаси-агент и `delegate` — везде, где уже есть `_WEB_TOOLS`/`fetch`.

## flowai_guide

`mcp_agent/servers/guide_server.py` — статический тул-самоописание:
модель зовёт его, когда пользователь спрашивает "что ты такое"/"что ты
умеешь", вместо того чтобы гадать. Не дублирует `/help` — это ориентир
по архитектуре, а не точный справочник команд.

## Кто какие тулы видит

`mcp_agent/roles.py` — НЕ статичная таблица "роль → тулы". Вместо этого
capability-группы (`_PROJECT_READ_TOOLS`, `_WRITE_TOOLS`, `_SHELL_TOOLS`,
`_WEB_TOOLS`, `_META_TOOLS`, ...) и функции-компоновщики
(`investigator_tools`/`planner_tools`/`coder_tools`/`verifier_tools`),
собирающие конкретный набор из флагов роутера (`needs_project`,
`needs_shell`) на каждый ход. `_META_TOOLS` (сейчас — только
`flowai_guide`) и плагинские тулы (см. [plugins.md](plugins.md))
доступны безусловно всем ролям пайплайна — не завязаны ни на один флаг.

`settings.optimized_tools` — переключатель в `/settings`, урезающий
список до одного тула на "смысл" (без переименования) —
`mcp_agent/optimized_tools.py:OPTIMIZED_TOOL_NAMES`.

## TOOLS_REQUIRING_APPROVAL

`mcp_agent/config.py` — список тулов, которые всегда спрашивают
подтверждение перед выполнением (мутирующие/исполняющие: `bash`,
`write_file`, `edit_file`, `delete_path`, `generate_image`, ...).
Читающие тулы (`read_file`, `grep_search`, `web_search`, `fetch`, ...)
идут без диалога — таково прямое решение пользователя (риск чтения
низкий по сравнению с записью).

## Добавить новый встроенный тул

Свой домен инструментов — свой MCP-сервер в `src/mcp_agent/servers/`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_server")

@mcp.tool()
async def my_tool(arg: str) -> str:
    """Description for the model — English only (see CLAUDE.md)."""
    return f"result for {arg}"

if __name__ == "__main__":
    mcp.run()
```

Зарегистрировать в `mcp_agent/config.py`'s `raw_servers` dict, и (если
тул нужен в новом пайплайне, не только легаси-агенту) добавить его имя в
подходящую capability-группу в `roles.py`. Если тул мутирует
состояние — добавить имя в `TOOLS_REQUIRING_APPROVAL`.

Для стороннего инструмента, который не должен жить в самом репозитории
flowAI — плагин, см. [plugins.md](plugins.md).
