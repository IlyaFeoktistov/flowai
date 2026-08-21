# Архитектура

flowAI — терминальный CLI-чат, использующий локальные LLM (нейросети)
через Ollama (плюс experimental форк llama.cpp для MoE expert-streaming,
см. [models.md](models.md)). Весь исходный код — под `src/` (плоский layout,
физически лежит под `src/`, но `pyproject.toml`'s `sources = ["src"]`
strip'ает префикс при сборке — снаружи всё выглядит как раньше:
`import cli`, `import mcp_agent`, ...).

## Точки входа

- `src/cli.py` — интерактивный терминальный чат (prompt_toolkit + Rich),
  основной способ использования.
- `src/main.py` — FastAPI-бэкенд веб-интерфейса (`web_morda/`):
  REST `/api/v1/*` + WS `/api/v1/ws/chat` (событийный поток), тот же
  движок, что и у CLI — см. [web-ui.md](web-ui.md).
- `src/mcp_agent/run_cli.py` — раннер для одноразовых вызовов без TUI
  (`python3 src/mcp_agent/run_cli.py "задача"`), полезен для живых
  тестов агента без интерактивного ввода.

## Два пути обработки хода

Каждое сообщение пользователя роутится в один из двух путей —
переключатель `pipeline_mode` в `/settings` (по умолчанию ВКЛ):

```
cli.py / main.py
        │
        ▼
pipeline_mode=ВКЛ (дефолт)              pipeline_mode=ВЫКЛ (основной агент)
        │                                        │
        ▼                                        ▼
mcp_agent/pipeline.py                   mcp_agent/agent.py:stream_chat
Router → Analyzer → Planner →           монолитный self-heal цикл —
Coder → Verifier                        исследование+запись+проверка
        │                               в ОДНОМ раунде, с финальным
        │                               LLM-судьёй (semantic_check)
        └────────────────┬──────────────────────┘
                          ▼
         mcp_agent/stage_runner.py:run_stage — общий self-heal движок
         (recursion-limit/context-overflow/ResponseError-восстановление,
         разбор утёкшей tool-call разметки, punt-to-user rescue,
         дайджест-ретраи) — один и тот же код крутит и каждую стадию
         пайплайна, и весь ход основного агента целиком.
```

Путь основного агента (`agent.py`) остаётся ЕДИНСТВЕННЫМ путём для голосового
режима (`voice_mode`) — у пайплайна нет голосовой ветки.

## Пайплайн (дефолтный путь)

1. **Router** (`mcp_agent/router.py:classify_intent`) — классифицирует
   входящее сообщение по 4 независимым да/нет осям: `needs_project`,
   `needs_shell`, `needs_change`, `change_is_ambiguous`. Не тратит
   тулы — быстрый JSON-вызов той же модели с `format="json"`.
   - Если ничего не требуется и включён `casual_answers_enabled` —
     прямой ответ без единого тула (`answer_casual`, `tools=[]`).
   - Иначе — попадает в Analyzer.
2. **Analyzer** — read-only исследование. Если `needs_change=false` (чистый
   вопрос) — его сводка это и есть финальный ответ. Если правка нужна и
   однозначна (`change_is_ambiguous=false`) — сразу combined-стадия
   **quick_fix** (исследование+правка в одном раунде, без Planner).
3. **Planner** — превращает находки Analyzer'а в пронумерованный план,
   подтверждает его у пользователя через `ask_user` (обязательно, единый
   тул с настоящим UI-диалогом, не текстовый вопрос).
4. **Coder** — выполняет план (пишет/правит файлы), без bash — не должен
   уметь сам себя "проверить".
5. **Verifier** — независимая проверка (bash разрешён, ЕДИНСТВЕННАЯ
   роль без write-тулов рядом с bash). Может отправить обратно к Coder
   на доработку (до `CODER_VERIFIER_MAX_ROUNDS`).

Каждая роль — свой `recursion_limit`/`max_attempts`
(`mcp_agent/roles.py:ROLE_RECURSION_LIMIT`/`ROLE_MAX_ATTEMPTS`), свой
набор тулов (композиция из capability-групп в `roles.py`, не статичная
таблица роль→тулы — см. [tools-and-mcp-servers.md](tools-and-mcp-servers.md)),
свой промпт (`mcp_agent/prompts.py`).

## Сборка агента

`mcp_agent/agent_builder.py` — единая точка сборки и для основного агента
(`_build_agent`), и для каждой роли пайплайна (`_build_role_agent`):

- модель (Ollama или expert-streaming backend, см.
  [models.md](models.md)) — кешируется, пересобирается на лету при
  смене `chat_model` в `/settings`, без перезапуска MCP-подпроцессов;
- тулы — загружаются из MCP-серверов (`mcp_agent/config.py:
  build_mcp_connections`), кешируются на весь процесс отдельно от
  модели (дорого поднимать заново на каждый ход);
- общий middleware-стек (`_base_agent_middleware`) — approval-диалоги,
  дедуп повторных тул-вызовов, guard-мидлвари, hooks плагинов (см.
  [plugins.md](plugins.md)) — плюс роль-специфичные добавки (например
  `_VerifierNoSelfFixMiddleware` только для Verifier).

## Self-heal

`stage_runner.py:run_stage` — общий движок ретраев для каждой
стадии/хода основного агента: если раунд не прошёл verdict-проверку (детерминированную
или через LLM-судью), строится дайджест уже сделанного и раунд повторяется
с урезанным контекстом, до `max_attempts`. Отдельно обрабатывает:
`GraphRecursionError`, реальный context-overflow от backend'а, утёкшую в
текст tool-call разметку (парсит и выполняет напрямую), и "punt-to-user" —
если модель вместо тула задаёт вопрос текстом, открывается настоящий
`ask_user`-диалог вместо того, чтобы жечь попытку на "вызови тул как надо"
(с капом `MAX_SELF_HEAL_ASKS`, чтобы не зациклиться).

## Голосовой ход

```
Alt+R → ui/audio.py:record_from_mic → transcribe (faster-whisper)
      → текст в поле ввода → обычный ход агента (путь основного агента) → ответ
      → (если voice_mode) ui/audio.py:speak → venv-tts subprocess (Chatterbox)
      → воспроизведение
```

## D&D-режим

Полностью отдельный от пайплайна чат-режим — см.
[dnd-mode.md](dnd-mode.md).
