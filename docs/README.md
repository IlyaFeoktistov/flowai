# Документация flowAI

Каждая тема — в своём файле:

- [architecture.md](architecture.md) — общее устройство приложения:
  вход (`cli.py`/`main.py`), пайплайн Router→Analyzer→Planner→Coder→
  Verifier против основного агента, `roles.py`'s композиция тулов по роли,
  self-heal цикл (`stage_runner.py`), `BuildCache`.
- [tools-and-mcp-servers.md](tools-and-mcp-servers.md) — как устроены
  тулы модели: MCP-серверы-подпроцессы, реестр
  `config.py:build_mcp_connections`, как тул становится видимым роли.
- [models.md](models.md) — какие модели используются, как выбирается
  дефолт, `expert_streaming`, `OLLAMA_NUM_CTX`, переключение через
  `/settings`.
- [persistence.md](persistence.md) — что и где хранится на диске:
  SQLite в `~/.local/share/flowai/`, память, `/clean`, логи,
  сгенерированные файлы.
- [generative-features.md](generative-features.md) — картинки, музыка,
  3D-модели, голос: как вызываются (слэш-команда против тула модели),
  куда сохраняются результаты.
- [dnd-mode.md](dnd-mode.md) — изолированный режим настольной ролевой
  игры (`/dnd`): отдельный агент, своё хранилище, свои мидлвари.
- [commands.md](commands.md) — полный список встроенных слэш-команд и
  как они соотносятся с командами плагинов/скилов.
- [plugins.md](plugins.md) — как грузятся глобальные плагины
  (`plugins/`) и per-project скилы/хуки (`.flowai/skills`,
  `.flowai/hooks`): слэш-команды, MCP-серверы, hooks.
- [development.md](development.md) — layout `src/`, установка для
  разработки, запуск тестов, живые прогоны, отладка.
- [web-ui.md](web-ui.md) — веб-интерфейс (`web_morda/` + `src/main.py`):
  REST `/api/v1/*`, WS-протокол событий, сессии, permission-мост.

Начать стоит с [architecture.md](architecture.md), если незнакомо
общее устройство приложения, или прямо с нужной темы, если ищется
что-то конкретное.
