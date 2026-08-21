# Веб-интерфейс (web_morda/ + src/main.py)

Локальный веб-фронтенд (`web_morda/`, React+TS+Vite) поверх того же
агента, что и `cli.py` — не отдельная реализация, а ещё один потребитель
уже существующего `on_event`-потока (`mcp_agent/pipeline.py`/
`mcp_agent/agent.py`) и permission-моста (`tools/confirm.py`), которые
`ui/app.py`'s терминальный TUI и так использует.

## Запуск

```bash
cd src && uvicorn main:app --reload --ws-ping-interval 20 --ws-ping-timeout 300
make run_web   # из корня репозитория — фронтенд (web_morda) + SearXNG detached
```

`make run_web` (см. `Makefile`) — `docker compose up -d searxng` (фоном,
переживает Ctrl+C) + `cd web_morda && npm run dev` (на переднем плане,
Ctrl+C останавливает только его). Бэкенд отдельно, командой выше — его
намеренно не включили в `make run_web`, чтобы `--reload`/логи uvicorn не
мешались в одном терминале с логами Vite.

## Структура фронтенда (`web_morda/src/`) — Feature-Sliced Design

```
app/        — точка сборки: App.tsx (композиция), styles/global.css (токены/reset), App.css
widgets/    — sidebar (сайдбар целиком), chat-panel (лента + TurnView)
features/   — send-message, pick-folder, view-doctor, check-updates,
              clean-storage, view-usage, manage-memory, view-plugins,
              edit-settings — один слайс = одно самостоятельное действие
              пользователя, каждый со своим ui/ (+ api/, если ходит в сеть)
entities/   — chat (типы хода/сообщений + useChatSocket), session (список
              сессий), project (текущая папка) — доменные модели + их API
shared/     — ui (Icons, Modal, kit.css — общие .btn/.code-block/...),
              api (client.ts — общий fetch-хелпер), lib (markdown.tsx)
```

Импорты: `@/...` (алиас на `src/`, настроен и в `tsconfig.app.json`, и в
`vite.config.ts` — держать оба в синхроне) для всего, что пересекает
границу слайса; внутри одного слайса (например `features/view-doctor/ui/`
→ `features/view-doctor/api/`) — обычные относительные `./`/`../`. Каждый
слайс отдаёт наружу только то, что реально нужно, через свой `index.ts`
(публичный API слайса) — соседние слои не лезут во внутренние файлы друг
друга напрямую.

`--ws-ping-timeout 300` — не опционально: холодный старт локальной модели
(`expert_streaming.py`'s `ensure_running` поднимает llama-server
подпроцесс и синхронно поллит его health-check `time.sleep`'ом, блокируя
event loop) или просто медленная генерация на слабом железе может занять
десятки секунд — с дефолтным таймаутом 20с uvicorn считает сокет мёртвым
и рвёт соединение прямо посреди хода.

## REST API — `/api/v1/*`

Все REST-маршруты и WS-эндпоинт версионированы одним префиксом
(`/api/v1`, см. `router = APIRouter(prefix="/api/v1")` в `main.py`) —
кроме `/health` (неверсионированная liveness-проба). Бампать префикс до
`/api/v2` есть смысл только при обратно несовместимом изменении REST-
контракта или wire-протокола событий, если у web_morda появятся клиенты,
которых нельзя одновременно обновить.

| Маршрут | Что |
|---|---|
| `GET /health` | liveness-проба, неверсионирован |
| `GET/POST /api/v1/project` | текущая рабочая папка процесса / сменить её (`os.chdir`, отклоняется 409, пока идёт ход) |
| `GET /api/v1/browse?path=` | список подпапок для UI выбора папки (не системный диалог — папка листается изнутри бэкенда) |
| `GET /api/v1/sessions` | список сессий (из `episodic_messages`) — id, превью, время |
| `GET /api/v1/sessions/{id}` | полная история сессии (только role=user/assistant — см. "Что НЕ сохраняется" ниже) |
| `GET /api/v1/doctor` | = `/doctor` в CLI |
| `POST /api/v1/update` | = `/update` |
| `GET/POST /api/v1/clean` | GET — отчёт без удаления, POST `{scope}` — реально чистит |
| `GET /api/v1/usage` | = `/usage` (персистентные totals, не live-счётчик текущего хода) |
| `GET/DELETE /api/v1/memory`, `DELETE /api/v1/memory/facts/{i}`, `DELETE /api/v1/memory/knowledge` | = `/memory` |
| `GET /api/v1/plugins` | = `/plugin` |
| `POST /api/v1/reindex` `{targets?}` | = `/reindex` |
| `GET/POST /api/v1/settings` | сырой dict `settings._state` — без /settings-меню, просто key/value |

Rich-разметка (`[bold]`/`[green]`/`[/]`) из `doctor.py`/`update.py`/
`clean.py`/`plugins.py` (эти модули пишутся для терминала) вырезается на
сервере (`main.py:_plain`, `rich.text.Text.from_markup(...).plain`) —
фронтенд получает чистый текст.

## WS-протокол — `/api/v1/ws/chat[?session_id=...]`

Без `session_id` в query — открывает НОВУЮ сессию (id приходит первым
событием `session_started`). С `session_id` — грузит историю из
`episodic_messages` и ПРОДОЛЖАЕТ ту же сессию (тот же файл, тот же
принцип, что `episodic/writer.py:resume_session`).

Клиент → сервер:
- `{"type": "user_message", "text": "..."}`
- `{"type": "permission_response", "id": "...", "answer": "y"|"a"|"n"}`
- `{"type": "ask_user_response", "id": "...", "answer": "..."}`

Сервер → клиент — ретранслирует as-is весь `on_event`-поток пайплайна/
основного агента (`answer_start/chunk/end`, `thinking_start/chunk/end`,
`tool_start/tool_arg_chunk/tool_end`, `stage_changed`, `plan_steps`/
`plan_step_done`, `stats`, `done`, `mid_turn_injected` — см.
[architecture.md](architecture.md) за их смыслом), плюс свои:
- `{"type": "session_started", "session_id": "..."}` — сразу после connect
- `{"type": "permission_request", "id", "action", "detail"}` — то же, что
  показывает `ui/app.py`'s permission-диалог (bash/запись файла и т.п.)
- `{"type": "ask_user_request", "id", "question", "options", "recommended"}`
  — модель сама спросила пользователя (тул `ask_user`)
- `{"type": "turn_complete", "session_id"}` — ход дописан в БД, можно
  разблокировать инпут
- `{"type": "error", "message"}` — например, попытка отправить сообщение,
  пока уже идёт другой ход на этом же процессе

`permission_request`/`ask_user_request` реализованы `web/bridge.py:
WebBridge` — единственная новая точка интеграции с ядром: `tools/
confirm.py` уже умеет отдавать эти диалоги произвольному `_app`
(`connect_app()`), `ui/app.py`'s curses-TUI и `WebBridge` — два равноправных
потребителя одного и того же контракта, ядро агента не тронуто.

`main.py`'s `ws_chat` держит ДВЕ конкурентные корутины на соединение —
`receive_loop` (читает входящие фреймы) и `process_turns` (гоняет ходы) —
не одну последовательную. Это принципиально: пока `process_turns` сидит
внутри `stream_fn`, ожидая ответ на `permission_request` через
`WebBridge`-future, `receive_loop` должен продолжать читать сокет, иначе
`permission_response` от кнопки в браузере физически не будет прочитан до
конца хода — то есть никогда, раз сам ход этого ответа и ждёт. Разрыв
соединения (`WebSocketDisconnect` из `receive_loop`) отменяет
`process_turns` явно (`asyncio.Task.cancel()`), а не оставляет его висеть:
без этого закрытая посреди permission-диалога вкладка держала бы
`_turn_lock` до перезапуска процесса, блокируя вообще ВСЕ последующие ходы
на этом сервере.

## Один ход за раз — на весь процесс

Как и терминальный `cli.py` (там это гарантирует сама природа
interactive stdin), сервер разрешает только один ОДНОВРЕМЕННЫЙ ход на
весь процесс (`main.py`'s `_turn_lock`), даже если открыто несколько
вкладок/сессий сразу — второе сообщение получит `{"type": "error", ...}`,
пока первое не отдаст `turn_complete`. Аналогично `/api/v1/project` (смена
рабочей папки) отклоняется 409, пока идёт ход — папка одна на процесс, как
`os.getcwd()` в CLI.

Bash/write-подтверждения (`ask_permissions` в `/settings`) — тоже
процесс-глобальные (`tools/confirm.py`'s `_always_approve`/
`_approved_actions`), не per-session: "разрешить всегда" в одной вкладке
действует и на следующий ход из другой. `_reset_session()` вызывается
один раз на WS-подключение (новая вкладка = новый "терминал"), не на
каждое сообщение — иначе "разрешить всегда" забывалось бы после первого
же ответа.

## Что НЕ сохраняется между перезагрузками

`episodic_messages` хранит только финальный текст хода (роли
`user`/`assistant`) — то же самое ограничение, что и в CLI (см.
[persistence.md](persistence.md)). Живая детализация хода (thinking/
tool-calls/промежуточные стадии пайплайна) видна во фронтенде только
ПОКА вкладка открыта в момент генерации — при перезагрузке страницы или
повторном открытии старой сессии из списка слева видны только финальные
реплики, без вложенной трассировки. Это ограничение источника данных, не
временное упрощение фронтенда.
