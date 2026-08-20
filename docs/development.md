# Разработка

## Layout

Все исходники — под `src/` физически на диске (`src/cli.py`,
`src/mcp_agent/...`), но **не** `src/flowai/...` — `pyproject.toml`'s
`sources = ["src"]` стрипает префикс `src/` при сборке
колеса/editable-установки, так что на `sys.path`/в импортах модули
видны плоско (top-level `import cli`, `import mcp_agent`, ...), как и
раньше. Поменялось только физическое расположение файлов, не сам layout
пакета.

`vendor/`, `generated/`, `scripts/`, `venv-tts/`, `.env` остались на
своих местах — реальные каталоги корня репозитория, не переехали
вместе с `src/`.

`sources = ["src"]` — `hatchling` (не `setuptools`: тот пытается
выполнить корневой `setup.py` как часть сборки пакета, а он тут —
отдельный bootstrap-скрипт под опциональные ML-подсистемы, не
упаковщик; `hatchling` вообще не трогает `setup.py`).

`[tool.hatch.build.targets.wheel].include` в `pyproject.toml` — только
"main"-набор, покрывающий `requirements.txt` (базовый чат обязателен,
всё остальное опционально). Тяжёлые части (веса image-gen/whisper/
tts/gen3d) ставятся отдельно через `python3 setup.py --only ...` в свои
изолированные venv (конфликтующие версии torch/CUDA между ними), не
через `pip install .`.

## Установка для разработки

```bash
python3 setup.py --dry-run   # что скрипт возьмётся делать, без единой реальной команды
python3 setup.py --only main # минимум для базового чата
python3 setup.py             # всё (main + image-gen + whisper + tts + gen3d-пайплайн + expert-streaming)
```

Полный список системных пререквизитов и `--only`-компонентов — в
README.md's "Установка".

Запуск: `./flowai` (лончер сам находит `.venv`, активировать вручную не
нужно) — под капотом `exec .venv/bin/python src/cli.py "$@"`. Или
`python3 src/cli.py` после `source .venv/bin/activate`.

## Тесты

```bash
pytest                        # весь набор
pytest tests/test_plugins.py  # один файл
pytest -k dnd                 # по имени
```

Никакого `pytest.ini`/`setup.cfg` — только `tests/conftest.py` с общими
хелперами для синтетических LangChain-сообщений (`ai_message`/
`tool_message`/`write_round`) для тестов verdict/guidance-логики. Тесты
изолируются от реального `~/.local/share/flowai/` через monkeypatch
конкретных модулей (`storage.data_dir`, `plugins._REPO_ROOT` и т.д.) —
не через `FLOWAI_DATA_DIR` глобально.

Не импортировать `cli.py` в тестах — он рвёт stdout/stderr-teardown
pytest'а (переопределяет `sys.stdout`/`sys.stderr` при импорте; см.
`ui/error_reporting.py`, куда специально вынесен
`install_background_exception_handler()`, чтобы не тянуть `cli.py`
транзитивно).

## Живые прогоны (не через pytest)

Модели на этом железе (характеристики — см. корневой `README.md`, раздел
«Что умеет» → «Ограничения») по умолчанию рассуждают (`reasoning`/`think`) —
для короткого одноходового ручного теста (`src/mcp_agent/run_cli.py`
или прямой вызов) это превращает быстрый ответ в минуты генерации без
всякой пользы, раз никто не читает эту цепочку рассуждений. Всегда
выключать thinking на живых прогонах:
`ChatOllama(..., reasoning=False)` или `"think": false` в сырых
Ollama API-вызовах.

## Отладка

- `DEBUG=1` (переменная окружения) — единственная настройка, для
  которой env приоритетнее сохранённого в `/settings` значения
  (временный флаг на один конкретный прогон, см. `settings.py`).
- `/doctor` — живая проверка Ollama/модели/MCP-серверов/хранилища из UI,
  включая сколько инстансов моделей сейчас реально резидентно и сколько
  каждый занимает (учитывает и expert-streaming — свой процесс вне
  демона Ollama, `ollama ps` его одного не покажет).
- Диагностические логи ходов — `data_dir()/run-logs/*.jsonl`, см.
  [persistence.md](persistence.md).

## Версионирование

`pyproject.toml`'s `version` и `version.py`'s `__version__` держатся в
синхроне вручную — нет единого источника истины, при релизе поправить
оба.

## Живые комментарии в коде

В `mcp_agent/*.py` много комментариев вида «на реальном прогоне
случилось X → отсюда решение Y» рядом с константой/архитектурным
решением, которое из этого следует (см. README.md). Меняя такую
настройку — сначала читать комментарий рядом: там объяснено, какую
проблему она решает и на что был похож сбой без неё, а не просто
"здравый смысл предполагает другое число".
