"""FastAPI backend for web_morda/ — bridges the same on_event stream and
tools/confirm.py permission hooks that ui/app.py's terminal TUI consumes,
over a websocket, instead of reimplementing any agent logic here.

Every route (REST and the /ws/chat websocket) lives under /api/v1 — see
`router` below — except /health, which stays unversioned by infra
convention. Bump the prefix to /api/v2 for a breaking change to the REST
shapes or the on_event wire protocol, once web_morda actually has clients
that need the old one kept alive; nothing does yet, so there's no v1/v2
routing split to maintain here until then.

Run: uvicorn main:app --reload --ws-ping-interval 20 --ws-ping-timeout 300
(generous ws ping timeout — a cold local-model load, or a slow response on
weak hardware, can legitimately block the event loop for tens of seconds;
see expert_streaming.py's ensure_running, which polls with a blocking
time.sleep — the default 20s ping timeout drops the socket mid-turn
otherwise.)
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rich.text import Text

import settings as app_settings  # noqa: E402
import usage as usage_mod  # noqa: E402
import memory_admin  # noqa: E402
import clean as clean_mod  # noqa: E402
from doctor import run_doctor  # noqa: E402
from update import run_update  # noqa: E402
from episodic import EpisodicWriter  # noqa: E402
from mcp_agent import plugins  # noqa: E402
from mcp_agent.agent import stream_chat as main_stream_chat  # noqa: E402
from mcp_agent.pipeline import stream_chat as pipeline_stream_chat  # noqa: E402
from rag.index_code import reindex_code_from_disk  # noqa: E402
from tools.confirm import connect_app as connect_confirm_app, _reset_session  # noqa: E402
from web.bridge import WebBridge  # noqa: E402
from web.sessions_store import get_session, list_sessions, next_seq  # noqa: E402

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
    return {"path": os.getcwd()}


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
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            mtype = data.get("type")
            if mtype in ("permission_response", "ask_user_response"):
                bridge.resolve(data)
            elif mtype == "user_message":
                await inbound.put((data.get("text") or "").strip())

    async def process_turns() -> None:
        while True:
            text = await inbound.get()
            if not text:
                continue
            if _turn_lock.locked():
                await send_safe({"type": "error", "message": "a turn is already in progress"})
                continue

            messages.append({"role": "user", "content": text})
            episodic.append("user", text)

            use_main = app_settings.get("voice_mode") or not app_settings.get("pipeline_mode")
            stream_fn = main_stream_chat if use_main else pipeline_stream_chat

            async with _turn_lock:
                connect_confirm_app(bridge)
                final_text = ""
                async for chunk in stream_fn(messages, on_event=send_safe):
                    final_text = chunk

            messages.append({"role": "assistant", "content": final_text})
            episodic.append("assistant", final_text)
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
        bridge.cancel_all()
        episodic.close()


app.include_router(router)
