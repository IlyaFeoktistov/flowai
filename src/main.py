"""FastAPI backend for web_morda/ — bridges the same on_event stream and
tools/confirm.py permission hooks that ui/app.py's terminal TUI consumes,
over a websocket, instead of reimplementing any agent logic here.

Every route (REST and the /ws/chat websocket) lives under /api/v1 — see
`router` below — except /health, which stays unversioned by infra
convention. Bump the prefix to /api/v2 for a breaking change to the REST
shapes or the on_event wire protocol, once web_morda actually has clients
that need the old one kept alive; nothing does yet, so there's no v1/v2
routing split to maintain here until then.

Run: `make run_web` from the repo root (backend + web_morda + SearXNG,
see Makefile/docs/web-ui.md) — or manually:
`.venv/bin/uvicorn main:app --reload --ws-ping-interval 20 --ws-ping-timeout 300`
from `src/` (the `.venv/bin/` prefix matters: a bare `uvicorn` with no
active venv can resolve to a system install missing this project's
dependencies). Generous ws ping timeout — a cold local-model load, or a
slow response on weak hardware, can legitimately block the event loop
for tens of seconds; see expert_streaming.py's ensure_running, which
polls with a blocking time.sleep — the default 20s ping timeout drops
the socket mid-turn otherwise.
"""
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from rich.text import Text
from starlette.background import BackgroundTask

import settings as app_settings  # noqa: E402
import usage as usage_mod  # noqa: E402
import memory_admin  # noqa: E402
from ui import audio as ui_audio  # noqa: E402
from ui import images as ui_images  # noqa: E402
import clean as clean_mod  # noqa: E402
from doctor import run_doctor  # noqa: E402
from update import run_update  # noqa: E402
from episodic import EpisodicWriter  # noqa: E402
from mcp_agent import plugins  # noqa: E402
from mcp_agent.agent import stream_chat as main_stream_chat  # noqa: E402
from mcp_agent.pipeline import stream_chat as pipeline_stream_chat  # noqa: E402
from mcp_agent import prompts  # noqa: E402
from rag.index_code import reindex_code_from_disk  # noqa: E402
from tools.confirm import connect_app as connect_confirm_app, _reset_session  # noqa: E402
from web.bridge import WebBridge  # noqa: E402
from web.sessions_store import get_session, list_sessions, next_seq, save_title  # noqa: E402
from mcp_agent.router import generate_session_title  # noqa: E402

# Переключает mcp_agent/prompts.py's math_notation_rule на LaTeX-инструкцию —
# см. её докстринг: терминал не умеет рендерить формулы, web_morda умеет
# (KaTeX), поэтому один и тот же system-промпт зависит от того, кто сейчас
# ведёт процесс.
prompts.set_web_mode(True)

# GET /project ниже отдаёт os.getcwd() как "текущий проект" — тот же принцип,
# что и cli.py (папка, откуда запущен процесс). Для терминала это осмысленно
# (пользователь сам cd'ит перед запуском), а для веба — нет: make run_web
# делает `cd src && uvicorn ...`, так что без этого chdir дефолтом был бы
# repo's src/, чисто случайность обвязки запуска, а не осмысленный выбор
# проекта. Домашняя папка — нейтральный старт, дальше пользователь выбирает
# реальный проект через folder-picker (POST /project, тоже os.chdir).
os.chdir(os.path.expanduser("~"))

app = FastAPI(title="Flowio AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Один агент-процесс — один ход одновременно, тот же принцип, что и в
# терминальном cli.py (там это гарантируется самой природой interactive
# stdin, здесь — явным логом, раз клиентов может быть несколько вкладок).
_turn_lock = asyncio.Lock()

# /health остаётся неверсионированным (инфра-конвенция для liveness-проб) —
# всё остальное, включая WS, версионируется через один префикс разом,
# чтобы протокол событий/REST-контракт можно было развести по /api/v2 в
# будущем, не трогая уже подключившихся клиентов на /api/v1.
router = APIRouter(prefix="/api/v1")


def _plain(markup_text: str) -> str:
    """doctor/clean/update/plugins reports are built with Rich console
    markup ([bold]/[green]/[/]...) for the terminal — meaningless outside
    it, so strip it down to plain text for the web UI."""
    return Text.from_markup(markup_text).plain


@app.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/project")
async def get_project():
    # home — не для навигации, только чтобы фронтенд мог показать "~/..."
    # вместо полного пути в узкой кнопке сайдбара (см. Sidebar.tsx).
    return {"path": os.getcwd(), "home": os.path.expanduser("~")}


class ProjectPath(BaseModel):
    path: str


@router.post("/project")
async def set_project(body: ProjectPath):
    if _turn_lock.locked():
        return JSONResponse({"error": "a turn is in progress"}, status_code=409)
    path = os.path.abspath(os.path.expanduser(body.path))
    if not os.path.isdir(path):
        return JSONResponse({"error": "not a directory"}, status_code=400)
    os.chdir(path)
    return {"path": path}


@router.get("/browse")
async def browse(path: str | None = None):
    base = Path(path or os.getcwd()).expanduser().resolve()
    if not base.is_dir():
        return JSONResponse({"error": "not a directory"}, status_code=400)
    try:
        dirs = sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    except PermissionError:
        dirs = []
    return {"path": str(base), "parent": str(base.parent), "dirs": dirs}


@router.get("/sessions")
async def sessions_endpoint():
    return list_sessions()


@router.get("/sessions/{session_id}")
async def session_detail(session_id: str):
    return get_session(session_id)


@router.get("/doctor")
async def doctor_endpoint():
    return {"report": _plain(await run_doctor())}


@router.post("/update")
async def update_endpoint():
    return {"report": _plain(await run_update())}


@router.get("/clean")
async def clean_report_endpoint():
    return {"report": _plain(clean_mod.run_clean(None))}


class CleanBody(BaseModel):
    scope: str


@router.post("/clean")
async def clean_apply_endpoint(body: CleanBody):
    return {"report": _plain(clean_mod.run_clean(body.scope))}


@router.get("/usage")
async def usage_endpoint():
    return usage_mod.totals()


@router.get("/memory")
async def memory_get_endpoint():
    return {
        "facts": memory_admin.get_facts(),
        "knowledge": [
            {"category": c, "key": k, "value": v} for c, k, v in memory_admin.get_knowledge()
        ],
    }


@router.delete("/memory")
async def memory_clear_all_endpoint():
    return memory_admin.clear_all()


@router.delete("/memory/facts/{index}")
async def memory_delete_fact_endpoint(index: int):
    return {"deleted": memory_admin.delete_fact(index)}


class KnowledgeKey(BaseModel):
    category: str
    key: str


@router.delete("/memory/knowledge")
async def memory_delete_knowledge_endpoint(body: KnowledgeKey):
    return {"deleted": memory_admin.delete_knowledge_entry(body.category, body.key)}


@router.get("/plugins")
async def plugins_endpoint():
    return {"report": _plain(plugins.describe_installed(os.getcwd()))}


class ReindexBody(BaseModel):
    targets: list[str] | None = None


@router.post("/reindex")
async def reindex_endpoint(body: ReindexBody):
    return await reindex_code_from_disk(os.getcwd(), targets=body.targets)


@router.get("/models")
async def models_endpoint():
    """Installed Ollama models — same source /doctor uses (`ollama list`
    via the ollama package's AsyncClient), for the /settings model
    selector's <select> options."""
    import ollama
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = await ollama.AsyncClient(host=host).list()
        return {"models": [m.model for m in resp.models]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


@router.post("/upload_image")
async def upload_image_endpoint(image: UploadFile = File(...)):
    """Attach-file chip for an image — reuses ui/images.py's store_image()
    (the same store CLI's /img and clipboard-paste use), returning a
    '[Image-N]' placeholder. The composer inlines that bare placeholder into
    the outgoing message text; process_turns resolves it to the real
    on-disk path (resolve_image_paths) right before the model sees it, the
    same point cli.py does it — never for a mid-turn injection either,
    matching cli.py's own _enqueue (see ws_chat's comment)."""
    data = await image.read()
    b64 = base64.b64encode(data).decode()
    placeholder = ui_images.store_image(b64)
    return {"placeholder": placeholder}


@router.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Speech-to-text for the mic button — reuses ui/audio.py's transcribe()
    (faster-whisper), the same STT the CLI's voice mode uses; only the
    RECORDING side there is WSL/Windows-specific (powershell.exe + MCI),
    transcribe() itself just takes a path and is portable. The browser does
    the actual recording (MediaRecorder) and uploads the resulting blob
    here instead of flowai capturing the mic itself.

    transcribe() is a blocking, CPU-bound faster-whisper call — run via
    asyncio.to_thread, not awaited directly, or it would stall the event
    loop (same class of bug as expert_streaming.ensure_running, see
    docs/web-ui.md) for the whole transcription, blocking every other
    request/websocket on this process meanwhile."""
    suffix = Path(audio.filename or "").suffix or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        text = await asyncio.to_thread(ui_audio.transcribe, tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"text": text}


class SpeakBody(BaseModel):
    text: str


@router.post("/speak")
async def speak_endpoint(body: SpeakBody):
    """Text-to-speech for the voice-mode orb — reuses ui/audio.py's
    synthesize_speech() (Chatterbox, via the separate venv-tts subprocess,
    see its own docstring on why a separate interpreter), the same TTS the
    CLI's voice_mode uses for `/talk`/spoken replies. Unlike ui/audio.py's
    own speak(), which synthesizes AND plays through Windows/MCI, this only
    synthesizes — playback happens in the browser's own <audio>, so the
    WSL-specific playback path never enters into it.

    Blocking subprocess.run (RTF ~4-5x on this CPU) — asyncio.to_thread for
    the same reason as /transcribe. The synthesized WAV is a temp file
    (synthesize_speech's own tempfile.mkstemp) — BackgroundTask deletes it
    once the response body has actually been sent, not before."""
    path = await asyncio.to_thread(ui_audio.synthesize_speech, body.text)
    if path is None:
        return JSONResponse({"error": "synthesis failed"}, status_code=502)
    return FileResponse(path, media_type="audio/wav", background=BackgroundTask(os.unlink, path))


@router.get("/settings")
async def settings_get_endpoint():
    return {k: v for k, v in app_settings._state.items() if not k.startswith("_")}


class SettingBody(BaseModel):
    key: str
    value: object


@router.post("/settings")
async def settings_set_endpoint(body: SettingBody):
    app_settings.set_value(body.key, body.value)
    return {"ok": True}


async def _generate_and_save_title(session_id: str, first_user_text: str) -> None:
    """Fire-and-forget background task (asyncio.create_task, never awaited
    inline — see its call site) — a short LLM call is still real latency,
    and nothing in the turn actually needs the title to be ready before
    turn_complete. The sidebar just won't show it until the NEXT listSessions()
    refresh (already happens on every turn completion, see App.tsx)."""
    title = await generate_session_title(first_user_text)
    if title:
        save_title(session_id, title)


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    _reset_session()  # новая вкладка/подключение — новый "терминал", как cli.py при старте

    session_id = ws.query_params.get("session_id")
    episodic = EpisodicWriter()
    if session_id:
        episodic.resume_session(session_id, next_seq(session_id))
        messages = [{"role": m["role"], "content": m["content"]} for m in get_session(session_id)]
    else:
        session_id = episodic.new_session()
        messages = []
    await ws.send_json({"type": "session_started", "session_id": session_id})

    bridge = WebBridge(ws.send_json)
    inbound: asyncio.Queue[str] = asyncio.Queue()
    # И current_turn_task, И mid_turn_queue живут на уровне ws_chat (не
    # внутри process_turns), потому что receive_loop должен их видеть —
    # "stop" отменяет current_turn_task, "user_message" во время активного
    # хода уходит в mid_turn_queue вместо inbound, если ход это поддерживает
    # (см. cli.py's собственный _mid_turn_queue — тот же приём, тут просто
    # веб-эквивалент). None у обоих значит "ходов сейчас нет".
    current_turn_task: asyncio.Task | None = None
    mid_turn_queue: asyncio.Queue | None = None

    async def send_safe(payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass

    # receive_loop и process_turns ДОЛЖНЫ идти конкурентно, не последовательно
    # (было именно так и это был баг): пока process_turns сидит внутри
    # stream_fn, ожидая ответ на permission_request/ask_user_request через
    # WebBridge, тот же самый цикл не может параллельно вызвать
    # ws.receive_text() ещё раз — сообщение с ответом пользователя так и
    # осталось бы непрочитанным в сокете до конца ХОДА, то есть навсегда,
    # раз сам ход ждёт именно этого ответа. user_message тоже идёт через
    # очередь, а не обрабатывается прямо в receive_loop — иначе тот же цикл
    # блокировался бы на await stream_fn(...) и не смог бы вычитывать
    # permission_response, которые могут прийти ПОКА этот же ход ещё идёт.
    async def receive_loop() -> None:
        nonlocal current_turn_task, mid_turn_queue
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            mtype = data.get("type")
            if mtype in ("permission_response", "ask_user_response"):
                bridge.resolve(data)
            elif mtype == "user_message":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                turn_running = current_turn_task is not None and not current_turn_task.done()
                if turn_running and mid_turn_queue is not None:
                    # Ход уже идёт и умеет мид-терн стир (основной агент,
                    # см. process_turns) — сообщение уйдёт МОДЕЛИ между
                    # шагами графа, не дожидаясь конца хода. Фронтенд узнает
                    # об этом по факту (событие mid_turn_injected из
                    # agent.py, когда _stream_round реально его заберёт),
                    # не по немедленному подтверждению отсюда.
                    await mid_turn_queue.put(text)
                else:
                    # Либо ходов сейчас нет (станет новым), либо идёт
                    # пайплайн-ход (мид-терн стир не поддерживает, см.
                    # agent.py/pipeline.py) — тогда ждёт своей очереди в
                    # inbound, как раньше.
                    await inbound.put(text)
            elif mtype == "stop":
                if current_turn_task is not None and not current_turn_task.done():
                    current_turn_task.cancel()

    async def process_turns() -> None:
        nonlocal current_turn_task, mid_turn_queue
        while True:
            text = await inbound.get()
            if not text:
                continue

            # Плейсхолдер [Image-N] -> реальный абсолютный путь ТОЛЬКО для
            # того, что реально идёт модели/в историю — эхо в turn_started
            # ниже нарочно шлёт ИСХОДНЫЙ текст с плейсхолдером, а не путь:
            # (1) фронтенду он и не нужен, (2) entities/chat's
            # displayTextByRawRef матчит по точно тому, что сам отправил.
            # cli.py делает это же разрешение только для НОВОГО хода, не
            # для мид-терн инъекции (см. receive_loop выше) — та же
            # асимметрия тут сознательно повторена.
            # Только для НОВОЙ сессии (не для возобновлённой из sidebar) —
            # messages стартует пустым только тогда, см. ws_chat выше
            # (resume_session грузит историю в messages сразу при коннекте).
            is_first_turn = len(messages) == 0

            resolved_text = ui_images.resolve_image_paths(text)
            messages.append({"role": "user", "content": resolved_text})
            episodic.append("user", resolved_text)
            await send_safe({"type": "turn_started", "text": text})

            use_main = app_settings.get("voice_mode") or not app_settings.get("pipeline_mode")
            stream_fn = main_stream_chat if use_main else pipeline_stream_chat
            mid_turn_queue = asyncio.Queue() if use_main else None

            # answer_seen — трекает, приходил ли хоть один настоящий
            # answer_chunk за этот ход. agent.py's stream_chat на recursion-
            # limit/context-overflow/generation-error (после исчерпания
            # попыток) не кидает исключение — он yield'ит готовое дружелюбное
            # сообщение и завершается штатно, ДО того, как модель вообще
            # произвела хоть один токен через on_event. CLI это видит (читает
            # выдачу генератора напрямую), а веб — нет: answer_start/chunk/end
            # это ЕДИНСТВЕННЫЙ канал, которым текст попадает в UI, и здесь он
            # просто не вызывается — ход тихо завершался бы пустым, без
            # единого слова и без ошибки.
            answer_seen = False

            async def on_event_wrapper(payload: dict) -> None:
                nonlocal answer_seen
                if payload.get("type") == "answer_chunk":
                    answer_seen = True
                await send_safe(payload)

            stream_kwargs = {"on_event": on_event_wrapper}
            if use_main:
                stream_kwargs["mid_turn_queue"] = mid_turn_queue

            async def run_turn():
                async with _turn_lock:
                    connect_confirm_app(bridge)
                    final = ""
                    async for chunk in stream_fn(messages, **stream_kwargs):
                        final = chunk
                    return final

            current_turn_task = asyncio.create_task(run_turn())
            try:
                final_text = await current_turn_task
                if not answer_seen and final_text:
                    await send_safe({"type": "answer_start"})
                    await send_safe({"type": "answer_chunk", "text": final_text})
                    await send_safe({"type": "answer_end"})
            except asyncio.CancelledError:
                # Остановлено кнопкой "стоп" — то, что модель уже успела
                # написать, уже дошло до фронтенда через answer_chunk;
                # здесь только фиксируем итог в истории/episodic и явно
                # сообщаем фронтенду, что это была отмена, а не обрыв связи.
                final_text = "⚠️ Ход остановлен пользователем."
                await send_safe({"type": "stopped"})
            except Exception as e:
                # cli.py's эквивалент (см. его же try/except вокруг
                # stream_chat) печатает ошибку и красиво завершает ход, не
                # роняя весь процесс — без этой ветки любое настоящее
                # исключение модели (сеть, 400 от локального сервера и
                # т.п., НЕ пойманное как context-overflow внутри
                # _stream_round) валило бы current_turn_task, а с ним и
                # весь process_turns/ws_chat: клиент получал бы голый
                # обрыв соединения без единого события, ход навсегда
                # оставался бы "не завершённым" в UI (мигающая точка), а
                # кнопка "стоп" пропадала бы просто как побочный эффект
                # ws.onclose — снаружи выглядело бы как рассинхрон между
                # кнопкой и индикатором, хотя причина одна.
                final_text = f"⚠️ Ошибка хода: {e}"
                await send_safe({"type": "error", "message": str(e)})
            finally:
                # Сообщение могло уйти в mid_turn_queue уже ПОСЛЕ того, как
                # ход фактически закончился (обычный конец, ошибка, стоп) —
                # _stream_round не успел дойти до следующей границы между
                # шагами графа и забрать его. Сливаем недошедшее обратно в
                # inbound, иначе оно потерялось бы молча (тот же приём, что
                # cli.py's finally-дренаж _mid_turn_queue).
                if mid_turn_queue is not None:
                    while not mid_turn_queue.empty():
                        await inbound.put(mid_turn_queue.get_nowait())
                mid_turn_queue = None
                current_turn_task = None

            messages.append({"role": "assistant", "content": final_text})
            episodic.append("assistant", final_text)
            if is_first_turn:
                asyncio.create_task(_generate_and_save_title(session_id, resolved_text))
            await send_safe({"type": "turn_complete", "session_id": session_id})

    receiver_task = asyncio.create_task(receive_loop())
    processor_task = asyncio.create_task(process_turns())
    try:
        done, pending = await asyncio.wait(
            {receiver_task, processor_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # И receive_loop (реальный disconnect), и process_turns (падение
        # хода) — обе ветки бесконечные, так что "завершилась" здесь всегда
        # значит "упала с исключением", не "закончила работу штатно".
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        # Отменяет process_turns, даже если он застрял на середине хода
        # (например ждёт ответа на permission_request, которого от
        # отключившегося клиента больше не будет) — иначе оборванная
        # вкладка держала бы _turn_lock до конца времён, блокируя вообще
        # ВСЕ будущие ходы на этом процессе.
        receiver_task.cancel()
        processor_task.cancel()
        # .cancel() только ПРОСИТ остановиться — сама отмена доставляется
        # асинхронно, на следующей же передаче управления циклу событий.
        # Без этого gather episodic.close() ниже мог выполниться РАНЬШЕ,
        # чем process_turns успевал дойти до своего except
        # asyncio.CancelledError (там же он и дописывает "остановлено
        # пользователем" в episodic/messages для кнопки "стоп", см. её
        # тело) — гонка "Cannot operate on a closed database", если
        # process_turns как раз был на середине episodic.append.
        await asyncio.gather(receiver_task, processor_task, return_exceptions=True)
        bridge.cancel_all()
        episodic.close()


app.include_router(router)
