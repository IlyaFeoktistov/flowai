#!/usr/bin/env python3
import asyncio
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Принудительный UTF-8 — чинит ошибки при переключении терминалов
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# httpx логирует каждый HTTP-запрос на уровне INFO обычным stdlib logging —
# при первой загрузке модели, вызванной ПРЯМО В ЭТОМ процессе (/gen, /music,
# /music_gen — в отличие от MCP-тулов, у которых stdout/stderr подпроцесса
# редиректятся в лог-файл, см. mcp_agent/config.py:_via_shell), эти строки
# лезут прямо в живой TUI-рендер как мусор. huggingface_hub/transformers сюда
# НЕ входят — у них своя обёртка поверх logging, которая инициализируется
# лениво и тихо перетирает внешний setLevel; их глушим их же родными
# set_verbosity_error()/disable_progress_bar() прямо в месте загрузки модели
# (см. mcp_agent/servers/music_server.py:_load_model) — здесь это не сработало
# бы и заодно означало бы тянуть тяжёлый import transformers при каждом
# старте CLI ради тула, который не факт что вызовут.
for _noisy_logger in ("httpx", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

# Тот же класс проблемы, что у httpx/urllib3 logging выше, другой источник:
# langchain_ollama падает с json.loads на аргументах тул-колла (например
# regex-запрос search_code с "\|" внутри — не валидный JSON escape) и падает
# в ast.literal_eval как фолбэк (chat_models.py:_parse_json_string) — эта
# функция парсит строку через compile(text, "<unknown>", "eval", ...), и
# Python сам печатает "SyntaxWarning: invalid escape sequence '\|'" ПРЯМО В
# stderr, минуя curses. Живой прогон: именно так и произошло (лог
# episodic_messages подтверждает search_code с "\|" в запросе в этот момент)
# — предупреждение вклинилось в отрисовку и оставило на экране дублированный
# футер/обрывок строки. literal_eval при этом успешно парсит строку (просто
# предупреждает про неоднозначный escape) — подавляем именно этот класс
# предупреждения, не трогая настоящие SyntaxError/ValueError, которые
# по-прежнему поднимаются и обрабатываются вызывающим кодом.
warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence.*")

from dotenv import load_dotenv
load_dotenv()

from rich.markup import escape
from rich.panel import Panel

from mcp_agent.agent import stream_chat as _legacy_stream_chat
from mcp_agent import dnd_store
from mcp_agent.dnd_agent import dnd_stream_chat, reconcile_before_exit
from mcp_agent.debug_log import log_event
from mcp_agent.model_config import DEBUG
from mcp_agent.pipeline import stream_chat as _pipeline_stream_chat
from mcp_agent.snapshots import clear_session_file_snapshots
import expert_streaming
import model_lifecycle
from compress import compress_history
from episodic import EpisodicWriter
from memory import get_store, DEFAULT_USER
from rag import EMBED_MODEL, VectorStore
from rag.index_dialog import index_episodic_entry
from tools.confirm import _reset_session, connect_app as connect_confirm_app
from tools.image_gen import generate_image
from tools.gen_model import generate_3d_model, animate_3d_model, generate_texture_for_model
from gen3d.img_refs import resolve_ref, EXTS as _IMG_EXTS
from gen3d.model_refs import resolve_model
from mcp_agent.servers.music_server import generate_music
from ui.app import FlowAIApp
from ui.console import console, safe_write, connect_app
from ui.header import print_header
from ui.images import get_clipboard_image, load_image_file, store_image, resolve_image_paths, clear_store as _clear_images
from ui.paste_store import resolve_pastes, clear_store as _clear_pastes
from ui.suggestions import looks_like_question, suggest_reply
from utils.proc import kill_process_tree

def clear_store():
    _clear_images()
    _clear_pastes()
from ui.stream import StreamDisplay
from ui.tui.settings import settings_menu
from ui.tui.usage import usage_screen
import settings as _settings
import storage
import usage as _usage

_reset_session()

_STATS: dict = {
    "messages":    0,
    "tokens_in":   0,
    "tokens_in_content": 0,
    "tokens_out":  0,
    "duration_ms": 0,
    "gen_duration_ms": 0,
    "tools_called": 0,
}


_SHELL_COMMAND_TIMEOUT = 60
_SHELL_COMMAND_MAX_OUTPUT = 5000


async def _run_shell_command(command: str) -> tuple[str, str, int | None]:
    """"!command" input (see _handle_input) — runs it directly (not through
    the model's own bash_exec MCP tool, no permission prompt: the user typed
    it themselves). Returns (stdout, stderr, returncode); returncode is None
    if the command was killed after _SHELL_COMMAND_TIMEOUT.

    stdin is left alone (inherited from this process' real terminal, NOT
    DEVNULL like bash_exec_server.py's own bash_exec) — this is a command
    the USER typed themselves specifically to run it directly, so a program
    that reads stdin should be able to. What it can't do is show a live
    prompt: stdout/stderr are captured wholesale and only printed after the
    command finishes (see _handle_input's use of the return value) — a
    program that prints "Enter a number:" and waits looks INSTANTLY frozen,
    with nothing on screen to explain why, even before the timeout below
    has any chance to fire.

    Live bug (20260812): a compiled Go binary that reads stdin
    (bufio.NewReader(os.Stdin)) was run this way, got no input (nothing
    typed reaches it — same terminal is also owned by prompt_toolkit's own
    input loop), and the OLD implementation here — asyncio.wait_for(proc.
    communicate(), timeout=...) — never actually fired its timeout: the
    process was still alive, unkilled, 269s after the coded 60s cap (ps
    evidence: both the /bin/sh -c wrapper AND the binary itself still
    running — proc.kill() sends SIGKILL, which cannot be caught/ignored, so
    if it had actually run, the wrapper would be dead). Root cause not
    fully pinned down (no live Python debugger available in this
    environment to inspect exactly where wait_for's cancellation got lost
    inside the full app — an isolated repro of just this pattern DID fire
    correctly), but wrapping a subprocess's own communicate() in wait_for
    and trusting its cancellation to reliably reach a blocked pipe read is
    exactly the kind of thing that's fragile to depend on. Replaced with a
    plain, independent watchdog task (asyncio.sleep + kill, no cancellation
    involved at all).

    Separately (found while testing the fix above): asyncio.subprocess.
    Process.kill() only signals the immediate `sh -c` wrapper — confirmed
    live that `/bin/sh -c` FORKS a real child for basically any external
    command (not just multi-command scripts; even a single bare command
    like "sleep 5" gets its own child PID here, not an exec() replacement).
    The child inherits the wrapper's own stdout/stderr pipe file
    descriptors, so if we kill only the wrapper, that orphaned child keeps
    holding the pipe's write end open — and proc.communicate() (what
    reads that pipe) does not see EOF until EVERY holder of the write end
    is gone. For a command being killed specifically because it never
    exits on its own, this means communicate() right after kill() can
    still hang forever even though the kill "succeeded". kill_process_tree
    (utils/proc.py) kills the whole spawned tree, not just the wrapper —
    verified live that communicate() then returns within the same tick."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return "", str(e), None

    timed_out = False

    async def _watchdog() -> None:
        nonlocal timed_out
        await asyncio.sleep(_SHELL_COMMAND_TIMEOUT)
        if proc.returncode is None:
            timed_out = True
            kill_process_tree(proc.pid)

    watchdog = asyncio.create_task(_watchdog())
    try:
        stdout_b, stderr_b = await proc.communicate()
    finally:
        watchdog.cancel()

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    if timed_out:
        # Not necessarily interactive-input specifically (could just be a
        # genuinely slow command) — worded as a hint, not a diagnosis, since
        # this tool has no way to actually tell the difference.
        note = (
            f"[killed — did not finish within {_SHELL_COMMAND_TIMEOUT}s. If "
            "it was waiting for input, this terminal's keyboard doesn't "
            "reach it live while it's captured this way — rerun it "
            "non-interactively instead (pipe the input in: `echo ... | "
            "cmd`, redirect from a file: `cmd < input.txt`, or use "
            "command-line flags/args instead of a stdin prompt). Output "
            "captured before it was killed is shown above, if any.]"
        )
        stderr = f"{stderr}\n{note}" if stderr else note
        return stdout, stderr, None
    return stdout, stderr, proc.returncode


def _format_bash_transcript(command: str, stdout: str, stderr: str, returncode: int | None) -> str:
    """English tags (see flowAI/CLAUDE.md: everything the model reads must
    be English) wrapping raw command output, which passes through verbatim
    whatever language/content it happens to have — same split as bash_exec's
    own MCP tool result, just for a human-typed "!command" instead of a
    model-issued tool call."""
    combined = "\n".join(x for x in (stdout.strip(), stderr.strip()) if x) or "(no output)"
    if len(combined) > _SHELL_COMMAND_MAX_OUTPUT:
        omitted = len(combined) - _SHELL_COMMAND_MAX_OUTPUT
        combined = combined[:_SHELL_COMMAND_MAX_OUTPUT] + f"\n...[TRUNCATED: {omitted} chars omitted]"
    exit_label = "killed (timeout)" if returncode is None else str(returncode)
    return (
        f"<bash-input>{command}</bash-input>\n"
        f"<bash-exit-code>{exit_label}</bash-exit-code>\n"
        f"<bash-output>\n{combined}\n</bash-output>"
    )


def _show_help() -> None:
    console.print(Panel(
        "[bold cyan]/gen[/] [dim]промпт[/]              — сгенерировать картинку напрямую\n"
        "[bold cyan]/img[/] [dim]путь.jpg[/]              — картинка с диска → [Image-N]\n"
        "[bold cyan]/paste[/]                   — картинка из буфера → [Image-N]\n"
        "[bold cyan]/music[/] [dim]промпт[/]            — потоковая музыка, /music ещё раз или Ctrl+C — стоп\n"
        "[bold cyan]/music_gen[/] [dim]промпт[/]        — один трек напрямую, без агента\n"
        "[bold cyan]/gen_model[/] [dim]промпт --rig --raw --lod N[/] — 3D-модель; --rig — со скелетом, --raw — без ретопологии, --lod N — ещё N моделей пониже полигонажем (_lod1.._lodN); @имя.png — картинка из img-refs/, несколько @ — батч (по модели на картинку, не одна общая)\n"
        "[bold cyan]/gen_model[/] [dim]--front @имя --left @имя --back @имя --right @имя[/] — ОДНА модель из нескольких ракурсов ОДНОГО объекта (Hunyuan3D-2mv). Ракурсы фиксированные, не произвольные: front — прямо на объект (0°), left — поворот на 90° влево, back — со спины (180°), right — поворот на 90° вправо. Нужен минимум front, для реального эффекта — front+left+back. Если референсы уже названы front.png/left.png/back.png — можно просто /gen_model @front @left @back, без флагов (определяется по именам)\n"
        "[bold cyan]/anim[/] [dim]описание движения[/]     — оживить последнюю риггованную модель, или /anim @имя ... — конкретную (Animato)\n"
        "[bold cyan]/gen_texture[/] [dim]@модель @картинка[/] — перегенерировать текстуру готовой модели по референсу, порядок аргументов не важен\n"
        "[bold cyan]/talk[/] [dim]текст[/]              — озвучить текст напрямую, без модели\n"
        "[bold cyan]/usage[/]                   — статистика токенов\n"
        "[bold cyan]/doctor[/]                  — проверка Ollama/модели/MCP-серверов/хранилища\n"
        "[bold cyan]/settings[/]                — настройки моделей и GPU\n"
        "[bold cyan]/memory[/]                  — что помнит нейронка, точечное/полное удаление\n"
        "[bold cyan]/dnd[/]                     — D&D-режим: список сохранений / новая игра\n"
        "[bold cyan]/clear[/]                   — очистить историю\n"
        "[bold cyan]/help[/]                    — эта справка\n"
        "[bold cyan]![/] [dim]команда[/]                  — выполнить shell-команду, вывод сразу уйдёт нейронке\n"
        "[bold cyan]@[/] [dim]имя_файла[/]              — поиск файлов в текущей директории, автокомплит по имени/пути\n\n"
        "[dim]Alt+V — вставить картинку из буфера · Alt+R — голосовой ввод (Ctrl+C — стоп записи)[/]\n"
        "[dim]Ctrl+C во время ответа/музыки — остановить[/]",
        title="[bright_black]команды[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))
    console.print()


# .glb is what generate_3d_model/run_gen_model ever writes to generated/models/,
# but a literal (non-@) path passed to /gen_texture could point at a mesh from
# anywhere -- gltf/fbx/obj are the other formats gen3d/pipeline.py's convert()
# already round-trips via Blender, and texture_wrapper.py loads via trimesh,
# which handles all of these too.
_GEN_TEXTURE_MODEL_EXTS = (".glb", ".gltf", ".fbx", ".obj")


def _classify_gen_texture_token(token: str) -> tuple[str, Path] | tuple[None, None]:
    """/gen_texture takes its model and reference-image arguments in either
    order -- classify a single token as ("model", path) or ("image", path)
    by checking, in order: @name against generated/models/ (a model), then
    @name against img-refs/ (an image); a bare path is classified by its
    extension instead. Returns (None, None) if it can't be resolved/classified
    at all."""
    if token.startswith("@"):
        name = token[1:]
        p = resolve_model(name)
        if p is not None:
            return "model", p
        p = resolve_ref(name)
        if p is not None:
            return "image", p
        return None, None
    p = Path(token)
    if not p.is_file():
        return None, None
    suffix = p.suffix.lower()
    if suffix in _GEN_TEXTURE_MODEL_EXTS:
        return "model", p
    if suffix in _IMG_EXTS:
        return "image", p
    return None, None


async def _reload_recap(app: FlowAIApp) -> None:
    data = await get_store().load(DEFAULT_USER)
    recap = data.get("recap", "")
    app.set_recap(recap)


async def main() -> None:
    app = FlowAIApp()
    connect_app(app)
    connect_confirm_app(app)
    print_header(app)

    # Ollama keeps a model resident for OLLAMA_KEEP_ALIVE (2h, see
    # model_config.py) in its OWN long-running daemon, independent of this
    # process — so a voice/vision model loaded in a previous session can
    # still be sitting in VRAM here, competing with the heavy coding model,
    # even though this is a fresh launch with nothing recorded in-process
    # yet. Previously the only way to clear that was the manual "выгрузить
    # модели" button / toggling voice_mode off in /settings on every
    # launch. Runs in a background thread (unload_idle_models is
    # synchronous, see model_lifecycle.py) so a slow/unreachable Ollama
    # doesn't delay startup; every step inside is already best-effort/fail-
    # open, so nothing here needs its own try/except.
    asyncio.create_task(asyncio.to_thread(model_lifecycle.unload_idle_models))

    messages: list[dict] = []
    episodic = EpisodicWriter()
    episodic.new_session()
    _usage.new_session()
    # Индекс — НЕ внутри текущей директории (раньше был ./rag_index/dialog.json
    # — засорял git status проекта, в котором открыт flowai, неотслеживаемым
    # файлом); project_dir хранит его вне дерева проекта, но с тем же
    # per-project разделением, что и раньше (см. storage.py, rag_server.py
    # использует ту же функцию для этого же файла).
    dialog_store = VectorStore.load(
        str(storage.project_dir(os.getcwd(), "rag_index") / "dialog.json"), model=EMBED_MODEL
    )

    async def _index_dialog_bg(entry: dict) -> None:
        # Не await'ится в критическом пути хода — сбой эмбеддинга не должен
        # блокировать ответ пользователю, запись в SQLite уже надёжно сделана отдельно.
        try:
            await index_episodic_entry(entry, dialog_store)
            dialog_store.save()
        except Exception:
            pass  # эмбеддинг — best-effort обогащение, не источник истины
    display = StreamDisplay(_STATS, app=app)

    # Живой инцидент: агент запорол правку файла в другом проекте (write_file
    # переписал его почти пустым при попытке "откатить" себя), а разобраться
    # постфактум было нечем — episodic хранит только финальный текст хода,
    # ни один tool_call/tool_result нигде не сохраняется, только печатается в
    # консоль под DEBUG=1 и теряется вместе со скроллбеком терминала. При
    # DEBUG=1 пишем эти события в ту же episodic_messages-таблицу — кроме
    # answer_chunk/thinking_chunk (потоковые дельты; их полное содержимое и
    # так попадает в уже сохраняемый финальный текст хода, писать их
    # поштучно было бы только шумом без новой информации).
    _DEBUG_SKIP_EVENTS = {"answer_chunk", "thinking_chunk"}

    async def _on_event(event: dict) -> None:
        nonlocal _waiting_for_model
        if event.get("type") in ("tool_start", "answer_chunk"):
            # Живой баг: _waiting_for_model раньше сбрасывался только на
            # ПЕРВЫЙ yield stream_chat() — а stream_chat это async-генератор,
            # который на обычном ходу отдаёт ровно ОДИН yield с уже ГОТОВЫМ
            # финальным текстом (см. agent.py — реальные tool_start/answer_
            # chunk идут только через on_event, не через yield). Значит
            # _waiting_for_model оставался True на ВСЁ время, что модель
            # реально искала файлы/звала тулы — а не только "пока первый
            # токен ещё не пришёл", как задумано комментарием ниже. Любое
            # доп. сообщение, отправленное в это время, ошибочно считалось
            # "модель ещё не начала" и уходило в ветку cancel-and-combine
            # (см. _enqueue) — отменяя весь уже проделанный раунд тулов
            # вместо того, чтобы встать в очередь и продолжить ПОСЛЕ него.
            _waiting_for_model = False
        t = event.get("type", "event")
        if t not in _DEBUG_SKIP_EVENTS:
            payload = {k: v for k, v in event.items() if k != "type"}
            log_event(t, **payload)
            if DEBUG:
                episodic.append(t, json.dumps(payload, ensure_ascii=False, default=str))
        await display.on_event(event)

    # Smart-queue state: allows new input to amend the current request while
    # the model is still "thinking" (pre-generation phase).
    _waiting_for_model = False      # True between messages.append and first token
    _active_handle_task: asyncio.Task | None = None
    _current_text: str = ""         # raw text being processed in current turn
    _suppress_echo = False          # True when combined message already echoed via amendment hint
    # Мид-терн стир (как в Claude Code): сообщение, пришедшее ПОКА текущий
    # ход уже не в фазе "жду первого токена" (тот случай уже покрыт amend-
    # combine чуть ниже), кладётся сюда вместо _pending — mcp_agent/agent.py:
    # _stream_round подхватывает его между шагами графа (после того, как
    # текущий tool_call доведён до конца, перед следующим вызовом модели),
    # а не после конца всего хода. Пересоздаётся заново на каждый ход
    # (_handle_input) — None, когда ходит не легаси-агент (D&D/pipeline_mode/
    # idle), тогда _enqueue откатывается на старую очередь _pending.
    _mid_turn_queue: "asyncio.Queue[str] | None" = None
    _music_task: asyncio.Task | None = None   # running /music stream, if any
    _talk_speech = None   # SpeechStreamer for /talk, lazily created (see ui/audio.py)

    # D&D-режим (/dnd) — отдельный от основного кодинг/casual-чата: свой
    # список сообщений (_dnd_messages, не смешивается с messages — иначе
    # фэнтези-нарратив мешался бы с историей кодинг-сессии и наоборот) и
    # свой изолированный агент (mcp_agent/dnd_agent.py), НЕ проходящий через
    # _legacy_stream_chat/_pipeline_stream_chat вообще. Структурное состояние
    # игры (персонаж/локация/инвентарь/партия/факты) живёт в БД
    # (mcp_agent/dnd_store.py), не здесь — эти три переменные — только "какая
    # игра активна прямо сейчас в этой сессии терминала".
    _dnd_active = False
    _dnd_game_id: int | None = None
    _dnd_messages: list[dict] = []

    def _dnd_exit(reason: str) -> None:
        """Общий выход из режима — из команды /dnd exit И из второго Ctrl+C
        (см. ui/app.py:set_dnd_active/_interrupt). game_id/conversation
        захватываются ЛОКАЛЬНО до сброса состояния — если игрок сразу же
        начнёт новую игру (/dnd new), фоновая сверка ниже не должна вдруг
        начать писать в НОВУЮ (уже другую) _dnd_game_id."""
        nonlocal _dnd_active
        _dnd_active = False
        app.set_dnd_active(False)
        game_id = _dnd_game_id
        conversation = list(_dnd_messages)
        console.print(f"[dim]  🎲 вышел из D&D-режима ({reason}) — сверяю сохранение...[/]")

        async def _finish_save() -> None:
            try:
                await reconcile_before_exit(game_id, conversation)
                console.print("[dim]  🎲 прогресс сохранён[/]\n")
            except Exception:
                console.print(
                    "[dim]  🎲 не удалось довериться финальной сверке — "
                    "сохранено то, что успело записаться по ходу игры[/]\n"
                )

        asyncio.create_task(_finish_save())

    async def _run_dnd_turn(seed_text: str) -> None:
        """One full D&D turn (stream + save) against _dnd_messages/
        _dnd_game_id — factored out of _handle_input's normal per-message
        flow so /dnd new and /dnd <id> can trigger it directly with a
        synthetic seed_text right after entering the mode, instead of
        leaving the DM silent until the player types something first (live
        request: starting/resuming a game with no greeting/recap read as
        the mode not actually having started). Never echoes seed_text as a
        "You ›" line — _handle_input's own flow already echoes a REAL
        player message before calling this; a synthetic kickoff shouldn't
        be echoed as if the player typed it."""
        nonlocal _waiting_for_model
        _dnd_messages.append({"role": "user", "content": seed_text})

        t_wall = time.monotonic()
        stopped = False
        _received_tokens = False
        _waiting_for_model = True
        display.start(t_wall)

        _chapter_ended = False

        async def _dnd_on_event(event: dict) -> None:
            nonlocal _chapter_ended
            if event.get("type") == "dnd_chapter_ended":
                _chapter_ended = True
            await _on_event(event)

        try:
            async for chunk in dnd_stream_chat(_dnd_messages, _dnd_game_id, on_event=_dnd_on_event):
                _waiting_for_model = False
                _received_tokens = True
                await display.finalize_stream_text(chunk)
        except (KeyboardInterrupt, asyncio.CancelledError):
            stopped = True
            display.stop_speech()
        except Exception as e:
            err = str(e).encode("utf-8", errors="replace").decode("utf-8")
            safe_write("\n")
            console.print(f"[red] ✗ {err}[/]")
            _dnd_messages.pop()
            console.print()
            return
        finally:
            await display.finish()
            _waiting_for_model = False

        turn_tok_in, turn_tok_out, turn_tok_in_content = display.flush_stats(stopped=stopped)
        _usage.record(turn_tok_in, turn_tok_out, turn_tok_in_content)
        if stopped and _received_tokens:
            console.print("[dim]  ↩ остановлено[/]")

        _dnd_messages[-1] = {"role": "user", "content": _dnd_messages[-1]["content"]}
        if display.full_response:
            _dnd_messages.append({"role": "assistant", "content": display.full_response})
        elif stopped:
            _dnd_messages.pop()

        if _chapter_ended:
            _dnd_messages.clear()
            console.print("[dim]  🎲 глава завершена — начинаю следующую с сохранённого состояния[/]\n")

    await _reload_recap(app)

    async def _handle_input(raw: str) -> None:
        """Handle a single line of user input submitted from the TUI."""
        nonlocal _waiting_for_model, _current_text, _suppress_echo, _music_task, _talk_speech
        nonlocal _dnd_active, _dnd_game_id, _dnd_messages, _mid_turn_queue
        user_input = raw.strip()
        if not user_input:
            return

        # ── "!команда" — выполнить shell-команду напрямую, результат уйдёт
        # нейронке как обычное сообщение ──────────────────────────────────
        # user_input переписывается на транскрипт ДО команд/_current_text
        # ниже — дальше по функции это просто обычный ход, ничего другого
        # менять не нужно (resolve_pastes/resolve_image_paths на транскрипте
        # безвредны: в нём нет [Paste-N]/[Image-N]-плейсхолдеров, если сама
        # команда их не напечатала буквально).
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if not shell_cmd:
                console.print("[red] ✗ Пустая команда после \"!\"[/]\n")
                return
            console.print(f"\n[green bold] You ›[/] [dim]! {escape(shell_cmd)}[/]\n")
            stdout, stderr, returncode = await _run_shell_command(shell_cmd)
            if stdout:
                safe_write(stdout if stdout.endswith("\n") else stdout + "\n")
            if stderr:
                console.print(f"[red]{escape(stderr)}[/]")
            console.print()
            user_input = _format_bash_transcript(shell_cmd, stdout, stderr, returncode)
            _suppress_echo = True  # команда+вывод уже показаны выше как есть

        # ── Команды ───────────────────────────────────────────────────────
        # Checked BEFORE touching _current_text/_suppress_echo: commands can
        # now run concurrently with an in-flight chat turn (see _enqueue),
        # so they must never mutate the state that turn's amend-combine
        # logic relies on.
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd      = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""
        else:
            cmd = cmd_args = ""

        if cmd in ("/help", "/?"):
            _show_help()
            return

        if cmd == "/clear":
            messages.clear()
            clear_store()
            app.clear_output()
            print_header(app)
            return

        if cmd == "/settings":
            # curses takes over the terminal — run outside the TUI app
            from prompt_toolkit.application.run_in_terminal import run_in_terminal
            def _run_settings():
                settings_menu(lambda: print_header(app))
            await run_in_terminal(_run_settings, render_cli_done=False)
            return

        if cmd == "/memory":
            from prompt_toolkit.application.run_in_terminal import run_in_terminal
            from ui.tui.memory_view import memory_menu
            def _run_memory():
                memory_menu(lambda: print_header(app))
            await run_in_terminal(_run_memory, render_cli_done=False)
            return

        if cmd == "/usage":
            from prompt_toolkit.application.run_in_terminal import run_in_terminal
            def _run_usage():
                usage_screen(_STATS, _usage.totals(), lambda: print_header(app))
            await run_in_terminal(_run_usage, render_cli_done=False)
            return

        if cmd == "/doctor":
            from doctor import run_doctor
            console.print("[dim]  🩺 проверяю Ollama/модели/MCP/хранилище…[/]\n")
            report = await run_doctor()
            console.print(Panel(report, title="[bright_black]doctor[/]", border_style="bright_black", padding=(0, 2)))
            console.print()
            return

        if cmd == "/paste":
            b64 = get_clipboard_image()
            if not b64:
                console.print("[red] ✗ Буфер пуст или не содержит изображение[/]\n")
                return
            placeholder = store_image(b64)
            console.print(f"[dim]   ↳ 🖼  {placeholder} вставлено · используй {placeholder} в запросе[/]\n")
            return

        if cmd == "/gen":
            if not cmd_args:
                console.print("[red] ✗ Укажи промпт: /gen cute kittens[/]\n")
                return
            console.print(f"[dim]  🎨 генерирую: {cmd_args}[/]")
            result = await generate_image({"prompt": cmd_args})
            console.print(f"[bright_black]     ↳ {result}[/]\n")
            return

        if cmd == "/gen_model":
            if not cmd_args:
                console.print("[red] ✗ Укажи промпт, [Image-N]/путь, или @имя.png: /gen_model cute penguin --rig[/]\n")
                return
            rig = False
            raw_flag = False
            lod = 0
            arg_text = cmd_args
            m = re.search(r"(^|\s)--rig(\s|$)", arg_text)
            if m:
                rig = True
                arg_text = (arg_text[:m.start()] + arg_text[m.end():]).strip()
            m = re.search(r"(^|\s)--raw(\s|$)", arg_text)
            if m:
                raw_flag = True
                arg_text = (arg_text[:m.start()] + arg_text[m.end():]).strip()
            m = re.search(r"(^|\s)--lod[=\s]+(\d+)(\s|$)", arg_text)
            if m:
                lod = int(m.group(2))
                arg_text = (arg_text[:m.start()] + arg_text[m.end():]).strip()
            # [Image-N] placeholders aren't resolved on the slash-command path
            # (see resolve_image_paths' call site below, only reached for
            # non-command input) — resolve explicitly here too.
            arg_text = resolve_image_paths(arg_text)

            # --front/--left/--back/--right: ONE model fused from several
            # views of the same object (Hunyuan3D-2mv), NOT the @ref-batch
            # semantics below (those give independent models, one per image
            # — see that branch's own comment). Deliberately a different,
            # value-taking flag syntax rather than overloading bare @ref
            # tokens, whose meaning is already taken.
            mv_views: dict[str, str] = {}
            mv_error = False
            for view in ("front", "left", "back", "right"):
                m = re.search(rf"(^|\s)--{view}[=\s]+(\S+)(\s|$)", arg_text)
                if not m:
                    continue
                raw_val = m.group(2)
                if raw_val.startswith("@"):
                    p = resolve_ref(raw_val[1:])
                    if p is None:
                        console.print(f"[red] ✗ Не найдено в img-refs/: {raw_val}[/]\n")
                        mv_error = True
                        break
                    mv_views[view] = str(p)
                elif Path(raw_val).is_file():
                    mv_views[view] = raw_val
                else:
                    console.print(f"[red] ✗ Файл не найден: {raw_val}[/]\n")
                    mv_error = True
                    break
                arg_text = (arg_text[:m.start()] + arg_text[m.end():]).strip()
            if mv_error:
                return
            if mv_views:
                flags_desc = " ".join(f for f in ("--rig" if rig else "", "--raw" if raw_flag else "",
                                                   f"--lod {lod}" if lod else "") if f)
                console.print(f"[dim]  🧊 генерирую 1 модель из {len(mv_views)} ракурсов ({', '.join(mv_views)}) {flags_desc}...[/]")
                result = await generate_3d_model({**mv_views, "rig": rig, "raw": raw_flag, "lod": lod})
                console.print(f"[bright_black]     ↳ {result}[/]\n")
                return

            tokens = arg_text.split()
            at_tokens = [t for t in tokens if t.startswith("@")]
            flags_desc = " ".join(f for f in ("--rig" if rig else "", "--raw" if raw_flag else "",
                                               f"--lod {lod}" if lod else "") if f)

            if tokens and len(at_tokens) == len(tokens):
                # Auto-detect multi-view intent from the ref NAMES themselves
                # (e.g. @front @left @back) matching Hunyuan3D-2mv's own view
                # tags -- more discoverable than requiring the explicit
                # --front/--left/--back flags when refs are already named
                # this way (a very natural way to name multi-view source
                # images in the first place). Only fires when EVERY name is
                # one of front/left/back/right with no repeats -- narrow
                # enough not to misfire on an unrelated batch of images that
                # happen to share a name with a view tag.
                view_tags = {"front", "left", "back", "right"}
                names = [t[1:].rsplit(".", 1)[0].lower() for t in at_tokens]
                if len(names) == len(set(names)) and set(names) <= view_tags:
                    mv_views = {}
                    mv_error = False
                    for t, name in zip(at_tokens, names):
                        p = resolve_ref(t[1:])
                        if p is None:
                            console.print(f"[red] ✗ Не найдено в img-refs/: {t}[/]\n")
                            mv_error = True
                            break
                        mv_views[name] = str(p)
                    if mv_error:
                        return
                    console.print(f"[dim]  🧊 генерирую 1 модель из {len(mv_views)} ракурсов "
                                  f"({', '.join(mv_views)}), определено по именам файлов {flags_desc}...[/]")
                    result = await generate_3d_model({**mv_views, "rig": rig, "raw": raw_flag, "lod": lod})
                    console.print(f"[bright_black]     ↳ {result}[/]\n")
                    return

                # every token is an @ref, but names don't match view tags --
                # batch: one model per image, in order, not a single
                # multi-view-fused model (Hunyuan3D-2mini doesn't do that —
                # see 3dtodo.md/README).
                resolved: list[Path] = []
                missing: list[str] = []
                for t in tokens:
                    p = resolve_ref(t[1:])
                    (resolved if p else missing).append(p if p else t)
                if missing:
                    console.print(f"[red] ✗ Не найдено в img-refs/: {', '.join(missing)}[/]\n")
                    return
                console.print(f"[dim]  🧊 генерирую {len(resolved)} моделей {flags_desc}...[/]")
                for i, p in enumerate(resolved, 1):
                    console.print(f"[dim]     [{i}/{len(resolved)}] {p.name}[/]")
                    result = await generate_3d_model({"prompt_or_path": str(p), "rig": rig, "raw": raw_flag, "lod": lod})
                    console.print(f"[bright_black]        ↳ {result}[/]")
                console.print()
                return

            if len(tokens) == 1 and len(at_tokens) == 1:
                p = resolve_ref(at_tokens[0][1:])
                if p is None:
                    console.print(f"[red] ✗ Не найдено в img-refs/: {at_tokens[0]}[/]\n")
                    return
                arg_text = str(p)

            console.print(f"[dim]  🧊 генерирую 3D-модель {flags_desc}: {arg_text}[/]")
            result = await generate_3d_model({"prompt_or_path": arg_text, "rig": rig, "raw": raw_flag, "lod": lod})
            console.print(f"[bright_black]     ↳ {result}[/]\n")
            return

        if cmd == "/anim":
            if not cmd_args:
                console.print("[red] ✗ Укажи описание движения: /anim wave with the right arm (или /anim @имя wave...)[/]\n")
                return
            model_path = ""
            motion = cmd_args
            tokens = cmd_args.split(maxsplit=1)
            if tokens and tokens[0].startswith("@"):
                p = resolve_model(tokens[0][1:])
                if p is None:
                    console.print(f"[red] ✗ Не найдено в generated/models/: {tokens[0]}[/]\n")
                    return
                model_path = str(p)
                motion = tokens[1] if len(tokens) > 1 else ""
                if not motion:
                    console.print("[red] ✗ Укажи описание движения после @имени: /anim @penguin wave with the right arm[/]\n")
                    return
            console.print(f"[dim]  🕺 анимирую{f' {Path(model_path).name}' if model_path else ''}: {motion}[/]")
            result = await animate_3d_model({"model_path": model_path, "motion": motion})
            console.print(f"[bright_black]     ↳ {result}[/]\n")
            return

        if cmd == "/gen_texture":
            if not cmd_args:
                console.print("[red] ✗ Укажи модель и референс, в любом порядке: /gen_texture @penguin @new_skin.png[/]\n")
                return
            arg_text = resolve_image_paths(cmd_args)
            tokens = arg_text.split()
            if len(tokens) != 2:
                console.print("[red] ✗ Нужно ровно два аргумента — модель (.glb) и картинка-референс: /gen_texture @penguin @new_skin.png[/]\n")
                return

            classified = [_classify_gen_texture_token(t) for t in tokens]
            missing = [t for t, (role, _) in zip(tokens, classified) if role is None]
            if missing:
                console.print(f"[red] ✗ Не найдено ни как модель (generated/models/), ни как картинка (img-refs/): {', '.join(missing)}[/]\n")
                return

            roles = [role for role, _ in classified]
            if roles[0] == roles[1]:
                what = "моделями" if roles[0] == "model" else "картинками"
                console.print(f"[red] ✗ Оба аргумента распознаны как {what} — нужны и модель, и референс[/]\n")
                return

            model_path = next(p for role, p in classified if role == "model")
            image_path = next(p for role, p in classified if role == "image")
            console.print(f"[dim]  🎨 перегенерирую текстуру: {model_path.name} ← {image_path.name}[/]")
            result = await generate_texture_for_model({"model_path": str(model_path), "image_path": str(image_path)})
            console.print(f"[bright_black]     ↳ {result}[/]\n")
            return

        if cmd == "/music":
            from ui.music_stream import stream_music, stop as stop_music
            if _music_task is not None and not _music_task.done():
                stop_music()
                console.print("[dim]  🎵 останавливаю (доиграет текущий кусок)...[/]\n")
                return
            if not cmd_args:
                console.print("[red] ✗ Укажи промпт: /music calm lofi beat[/]\n")
                return

            def _on_status(msg: str) -> None:
                console.print(f"[dim]  🎵 {msg}[/]")

            async def _run_music() -> None:
                try:
                    await stream_music(cmd_args, on_status=_on_status)
                finally:
                    # stream_music теперь реально ждёт остановки фонового
                    # синтеза (см. ui/music_stream.py) — снимаем "активно"
                    # только тут, когда это правда так, а не сразу при вызове.
                    app.set_music_active(False)

            console.print(f"[dim]  🎵 запускаю поток: {cmd_args} · /music или Ctrl+C — стоп[/]")
            _music_task = asyncio.create_task(_run_music())
            app.set_music_active(True, stop_music)
            return

        if cmd == "/music_gen":
            if not cmd_args:
                console.print("[red] ✗ Укажи промпт: /music_gen calm lofi beat[/]\n")
                return
            console.print(f"[dim]  🎵 генерирую: {cmd_args}[/]")
            result = await generate_music(cmd_args)
            console.print(f"[bright_black]     ↳ {result}[/]\n")
            return

        if cmd == "/talk":
            if not cmd_args:
                console.print("[red] ✗ Укажи текст: /talk привет, как дела[/]\n")
                return
            from ui.audio import SpeechStreamer
            console.print("[dim]  🔊 говорю...[/]")
            if _talk_speech is None:
                _talk_speech = SpeechStreamer()
            for sentence in re.split(r"(?<=[.!?…])\s+", cmd_args):
                _talk_speech.feed(sentence)
            _talk_speech.finish()
            return

        if cmd == "/img":
            if not cmd_args:
                console.print("[red] ✗ Укажи путь: /img файл.jpg[/]\n")
                return
            img_path = cmd_args.split()[0]
            try:
                b64 = load_image_file(img_path)
                placeholder = store_image(b64)
                console.print(f"[dim]   ↳ 🖼  {placeholder} загружено: {img_path}[/]\n")
            except FileNotFoundError as e:
                console.print(f"[red] ✗ {e}[/]\n")
            return

        if cmd == "/dnd":
            # /dnd (голое) и /dnd new/<id> — единственные подкоманды, которые
            # ДОЛЖНЫ работать вне активного режима (это и есть точка входа);
            # /dnd exit и /dnd help — только пока режим уже активен, по
            # прямому запросу пользователя не засорять команду её же
            # внутренними подкомандами, когда снаружи она не при делах.
            sub, _, rest = cmd_args.partition(" ")
            sub = sub.lower()
            rest = rest.strip()

            if sub == "exit":
                if not _dnd_active:
                    console.print("[red] ✗ Ты не в D&D-режиме — начни: /dnd new[/]\n")
                    return
                _dnd_exit("команда /dnd exit")
                return

            if sub == "help":
                if not _dnd_active:
                    console.print("[red] ✗ /dnd help — только внутри активной игры. Сначала /dnd new или /dnd <id>[/]\n")
                    return
                console.print(Panel(
                    "[bold cyan]/inventory[/]              — инвентарь текущего персонажа\n"
                    "[bold cyan]/status[/]                 — кто ты, где, когда, погода, здоровье, кто рядом\n"
                    "[bold cyan]/facts[/]                  — что мастер запомнил (обещания, секреты, лор мира)\n"
                    "[bold cyan]/dnd exit[/]               — выйти из D&D-режима\n"
                    "[dim]Ctrl+C во время ответа мастера — остановить генерацию;\n"
                    "Ctrl+C, когда мастер молчит — выйти из D&D-режима[/]\n\n"
                    "[dim]Дальше просто пиши обычным текстом, что делает твой персонаж —\n"
                    "мастер сам спросит, если что-то нужно уточнить.[/]",
                    title="[bright_black]команды D&D-режима[/]",
                    border_style="bright_black", padding=(0, 2),
                ))
                console.print()
                return

            if sub == "new":
                name = rest or f"Игра от {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                gid = dnd_store.create_game(name)
                _dnd_active = True
                _dnd_game_id = gid
                _dnd_messages = []
                app.set_dnd_active(True, lambda: _dnd_exit("Ctrl+C"))
                console.print(
                    f"[dim]  🎲 новая игра «{name}» (id={gid}) · /dnd help — команды режима\n"
                    "     мастер сейчас пишет первый ответ — это может занять от секунд до нескольких минут, "
                    "в зависимости от модели и мощности компьютера[/]\n"
                )
                # Мастер сам начинает — живой запрос пользователя: раньше
                # после /dnd new он молчал, пока игрок не напишет первым,
                # что читалось как "режим не запустился". Синтетическая
                # затравка НЕ эхается как "You ›" — см. _run_dnd_turn.
                await _run_dnd_turn(
                    "(OUT-OF-CHARACTER: the player just started a brand "
                    "new game. Greet them briefly and begin character "
                    "creation — ask about race, per your instructions.)"
                )
                return

            if sub.isdigit():
                gid = int(sub)
                game = dnd_store.get_game(gid)
                if game is None:
                    console.print(f"[red] ✗ Сохранение с id={gid} не найдено — /dnd покажет список[/]\n")
                    return
                _dnd_active = True
                _dnd_game_id = gid
                _dnd_messages = []
                app.set_dnd_active(True, lambda: _dnd_exit("Ctrl+C"))
                char = f"{game['race'] or '?'}/{game['class'] or '?'}"
                where = game["location"] or "неизвестно где"
                console.print(
                    f"[dim]  🎲 продолжаю «{game['name']}» ({char}, {where}) "
                    f"· /dnd help — команды режима\n"
                    "     мастер сейчас пишет рекап — это может занять от секунд до нескольких минут, "
                    "в зависимости от модели и мощности компьютера[/]\n"
                )
                # Живой запрос пользователя: при загрузке сохранения мастер
                # должен сам напомнить, кто ты/где ты/что происходит, а не
                # молчать, пока игрок не спросит. Контекст (раса/класс/
                # локация/инвентарь/факты) мастер УЖЕ видит через context
                # note (dnd_agent.py:_inject_context) — тут только просим
                # его пересказать это игроку, а не создавать заново.
                await _run_dnd_turn(
                    "(OUT-OF-CHARACTER: the player just resumed this saved "
                    "game after being away. Welcome them back with a brief "
                    "recap — where they are, who they are, what's "
                    "currently happening — using the game state already "
                    "shown to you, then ask what they'd like to do next. "
                    "Do NOT re-run character creation, race/class are "
                    "already set.)"
                )
                return

            if sub == "delete":
                if not rest.isdigit():
                    console.print("[red] ✗ Укажи id: /dnd delete 3 (номер видно в /dnd list)[/]\n")
                    return
                del_id = int(rest)
                if _dnd_active and del_id == _dnd_game_id:
                    console.print("[red] ✗ Нельзя удалить активную игру — сначала /dnd exit[/]\n")
                    return
                game = dnd_store.get_game(del_id)
                if game is None:
                    console.print(f"[red] ✗ Сохранение с id={del_id} не найдено — /dnd list покажет список[/]\n")
                    return
                dnd_store.delete_game(del_id)
                console.print(f"[dim]  🎲 удалено: «{game['name']}» (id={del_id})[/]\n")
                return

            if sub and sub != "list":
                console.print(f"[red] ✗ Неизвестная подкоманда: /dnd {sub} — /dnd list, /dnd new [имя], /dnd <id>, /dnd delete <id>, /dnd exit[/]\n")
                return

            # /dnd и /dnd list — список сохранений (одно и то же)
            games = dnd_store.list_games()
            if not games:
                console.print("[dim]  🎲 сохранений пока нет — начни: /dnd new [имя][/]\n")
                return
            lines = [
                f"[cyan]{g['id']}[/] — {g['name']}  "
                f"[dim]({g['race'] or '?'}/{g['class'] or '?'}, {g['location'] or '?'}, обновлено {g['updated_at']})[/]"
                for g in games
            ]
            console.print(Panel(
                "\n".join(lines), title="[bright_black]сохранения /dnd[/]",
                border_style="bright_black", padding=(0, 2),
            ))
            console.print("[dim]  /dnd <id> — продолжить, /dnd new [имя] — новая игра, /dnd delete <id> — удалить[/]\n")
            return

        if cmd == "/inventory" and _dnd_active:
            items = dnd_store.get_inventory(_dnd_game_id)
            if not items:
                console.print("[dim]  🎒 инвентарь пуст[/]\n")
                return
            lines = [
                f"- {i['item']} x{i['qty']}" + (f" — {i['description']}" if i["description"] else "")
                for i in items
            ]
            console.print(Panel(
                "\n".join(lines), title="[bright_black]инвентарь[/]",
                border_style="bright_black", padding=(0, 2),
            ))
            console.print()
            return

        if cmd == "/status" and _dnd_active:
            game = dnd_store.get_game(_dnd_game_id)
            if game is None:
                console.print("[red] ✗ Игра не найдена[/]\n")
                return
            xp_to_next = dnd_store.xp_for_level(game["level"] + 1) - game["xp"]
            lines = [
                f"[bold]Персонаж:[/] {game['race'] or '?'} / {game['class'] or '?'}",
                f"[bold]Уровень:[/] {game['level']}  [dim]({game['xp']} XP, {xp_to_next} до следующего)[/]",
                f"[bold]Здоровье:[/] {game['health_status']}",
                f"[bold]Локация:[/] {game['location'] or '(не задана)'}",
                f"[bold]Дата/время:[/] {game['in_game_date'] or '?'}, {game['time_of_day'] or '?'}",
                f"[bold]Погода:[/] {game['weather'] or '(не задана)'}",
                f"[bold]Золото:[/] {game['gold']}",
            ]
            if game["current_threat"]:
                lines.append(f"[bold]Текущая угроза:[/] {game['current_threat']} (уровень {game['current_threat_level']})")
            injuries = dnd_store.get_injuries(_dnd_game_id)
            if injuries:
                listed = ", ".join(
                    i["description"] + (f" ({i['severity']})" if i["severity"] else "")
                    for i in injuries
                )
                lines.append(f"[bold]Травмы:[/] {listed}")
            party = dnd_store.get_party(_dnd_game_id)
            if party:
                names = ", ".join(f"{p['name']}" + (f" — {p['description']}" if p["description"] else "") for p in party)
                lines.append(f"[bold]Рядом с тобой:[/] {names}")
            else:
                lines.append("[bold]Рядом с тобой:[/] [dim]никого, путешествуешь один[/]")
            console.print(Panel(
                "\n".join(lines), title="[bright_black]состояние[/]",
                border_style="bright_black", padding=(0, 2),
            ))
            console.print()
            return

        if cmd == "/facts" and _dnd_active:
            facts = dnd_store.get_facts(_dnd_game_id, limit=100)
            if not facts:
                console.print("[dim]  📜 мастер пока ничего не запомнил[/]\n")
                return
            lines = [f"{i + 1}. {f}" for i, f in enumerate(facts)]
            console.print(Panel(
                "\n".join(lines), title="[bright_black]факты (старые сверху)[/]",
                border_style="bright_black", padding=(0, 2),
            ))
            console.print()
            return

        if cmd:
            console.print(f"[red] ✗ Неизвестная команда: {cmd}[/]  [dim](/help для справки)[/]\n")
            return

        _current_text = user_input
        skip_echo = _suppress_echo
        _suppress_echo = False

        # Expand paste placeholders — model and echo both get the real text
        model_input = resolve_pastes(user_input)
        # Same idea for [Image-N] — the agent's tools (analyze_image,
        # edit_image) run in separate MCP subprocesses that can't see this
        # process's in-memory image store, so the model needs a real,
        # openable file path instead of the placeholder text.
        model_input = resolve_image_paths(model_input)

        # 🔒 Защита: не добавлять пустые сообщения
        if not model_input or not model_input.strip():
            console.print("[red] ✗ Пустой запрос после обработки paste-плейсхолдеров[/]\n")
            return

        # ── Echo user message to output ────────────────────────────────────
        if not skip_echo:
            console.print(f"\n[green bold] You ›[/] {escape(model_input)}\n")

        # D&D-режим ведёт СВОЙ список сообщений (_dnd_messages) и свой
        # изолированный агент — _run_dnd_turn (определена выше, рядом с
        # _dnd_exit) делает всё то же самое (стрим/сохранение истории/
        # сброс на конце главы), что и обычная ветка ниже, просто под
        # dnd_stream_chat вместо кодинг-пайплайна и без episodic/RAG/recap
        # (это всё заточено под кодинг-сессию, для dnd не нужно).
        if _dnd_active:
            await _run_dnd_turn(model_input)
            return

        messages.append({"role": "user", "content": model_input})
        entry = episodic.append("user", model_input)
        asyncio.create_task(_index_dialog_bg(entry))

        # ── Стриминг ──────────────────────────────────────────────────────
        t_wall  = time.monotonic()
        stopped = False
        _received_tokens = False
        _waiting_for_model = True   # enter pre-generation phase
        display.start(t_wall)

        # Новый пайплайн (mcp_agent/pipeline.py) не участвует в voice_mode
        # (нет голосовой ветки) — та ветка всегда идёт через легаси-агент,
        # независимо от pipeline_mode. Проверяем оба флага НА КАЖДЫЙ ход
        # (не один раз при старте) — оба переключаются на лету через
        # /settings, и пользователь должен увидеть эффект сразу, без
        # перезапуска.
        use_legacy = _settings.get("voice_mode") or not _settings.get("pipeline_mode")
        stream_chat = _legacy_stream_chat if use_legacy else _pipeline_stream_chat
        # Мид-терн стир (см. _mid_turn_queue выше) — только у легаси-агента:
        # _pipeline_stream_chat раскладывает ход на стадии с СВОИМИ
        # thread_id per стадия/ретрай (mcp_agent/stage_runner.py), а не один
        # непрерывный тред — тот же приём "продолжить тем же thread_id с
        # новым HumanMessage" там не годится без отдельной проработки.
        _mid_turn_queue = asyncio.Queue() if use_legacy else None
        stream_kwargs = {"on_event": _on_event}
        if use_legacy:
            stream_kwargs["mid_turn_queue"] = _mid_turn_queue
        try:
            async for chunk in stream_chat(messages, **stream_kwargs):
                _waiting_for_model = False  # first token arrived
                _received_tokens = True
                await display.finalize_stream_text(chunk)
        except (KeyboardInterrupt, asyncio.CancelledError):
            stopped = True
            display.stop_speech()
        except Exception as e:
            err = str(e).encode("utf-8", errors="replace").decode("utf-8")
            safe_write("\n")
            console.print(f"[red] ✗ {err}[/]")
            messages.pop()
            console.print()
            return
        finally:
            await display.finish()
            _waiting_for_model = False
            # Ход мог закончиться (обычный конец, ошибка, Ctrl+C) ДО того,
            # как _stream_round дошёл до следующей ToolMessage-границы и
            # успел забрать сообщение из _mid_turn_queue — без этого дренажа
            # оно бы просто потерялось молча (пользователь уже увидел "📤
            # передам по ходу", но модель его так и не увидела бы). Сливаем
            # недошедшее в обычную _pending — сработает как новый ход.
            if _mid_turn_queue is not None:
                drained = False
                while not _mid_turn_queue.empty():
                    await _pending.put(_mid_turn_queue.get_nowait())
                    drained = True
                _mid_turn_queue = None
                if drained:
                    app.set_queue_size(_pending.qsize())

        turn_tok_in, turn_tok_out, turn_tok_in_content = display.flush_stats(stopped=stopped)
        _usage.record(turn_tok_in, turn_tok_out, turn_tok_in_content)
        if stopped and _received_tokens:
            console.print("[dim]  ↩ остановлено[/]")
        await _reload_recap(app)

        # Save to history without images
        messages[-1] = {"role": "user", "content": messages[-1]["content"]}
        if display.full_response:
            messages.append({"role": "assistant", "content": display.full_response})
            entry = episodic.append("assistant", display.full_response)
            asyncio.create_task(_index_dialog_bg(entry))
        elif stopped:
            messages.pop()

        # Voice mode speaks as text streams in (see ui/stream.py:
        # StreamDisplay._feed_speech/_flush_speech_round) — no post-hoc
        # single blocking speak() call here anymore, that used to mean
        # waiting for the whole response before synthesis even started.

        # If the AI ended its turn on a question, offer a one-Tab reply —
        # skipped when the user Ctrl+C'd (nothing was actually asked to
        # answer) or the model produced no real text.
        if not stopped and display.full_response and looks_like_question(display.full_response):
            async def _offer_suggestion(ai_text: str) -> None:
                reply = await suggest_reply(ai_text)
                if reply:
                    app.set_input_suggestion(reply)
            asyncio.create_task(_offer_suggestion(display.full_response))

        # ── Context compression ───────────────────────────────────────────
        if not stopped:
            tok_in = display.pending_stats.get("tokens_in", 0)
            limit  = _settings.get("context_limit")
            threshold = int(limit * _settings.get("compress_at"))
            if tok_in > threshold:
                def _notify(n_old: int, n_words: int) -> None:
                    console.print(
                        f"[dim]  ↯ контекст сжат: {n_old} сообщений → ~{n_words} слов[/]\n"
                    )
                messages[:] = await compress_history(messages, on_notify=_notify)
                await _reload_recap(app)

    # ── Sequential input queue ────────────────────────────────────────────
    _pending: asyncio.Queue[str] = asyncio.Queue()

    async def _enqueue(text: str) -> None:
        nonlocal _active_handle_task, _current_text, _suppress_echo, _mid_turn_queue
        if text.strip().startswith("/"):
            # Commands (/usage, /settings, /clear, ...) are local UI actions,
            # not chat text — they must run right away regardless of whether
            # a chat turn is mid-thought or mid-stream, without cancelling it
            # (unlike the amend-combine case below) and without waiting behind
            # it in _pending (unlike the plain-queue case below).
            #
            # Still registered via set_active_task -- confirmed live (real
            # pty-driven Ctrl+C test) that NOT doing this was the actual
            # reason Ctrl+C could never stop /gen_model: this branch used to
            # fire-and-forget the task with no tracking at all, so
            # ui/app.py's _interrupt always saw self._active_task as None
            # and fell through to the no-op buffer-reset branch, regardless
            # of how correct gen3d/pipeline.py's own cancel_event/subprocess-
            # kill plumbing was -- that code was simply never being reached.
            # clear_active_task (not set_active_task(None)) on completion --
            # live bug (user report): /settings blocks in its own curses
            # screen (run_in_terminal) for as long as it's open, which is
            # routinely LONGER than "near-instant" if opened while a chat
            # turn is still streaming. Closing it fires this task's
            # done-callback while the chat task is still running --
            # set_active_task(None) used to clobber that chat task's own
            # registration unconditionally, leaving Ctrl+C with nothing to
            # cancel for the rest of that turn. clear_active_task guards on
            # task identity so only the task that's actually still current
            # gets cleared.
            #
            # Live bug (user report, /usage — same root cause applies to any
            # command that blocks in its own screen: /settings, /memory):
            # clear_active_task above only stops this command's OWN
            # registration from wrongly nulling out a chat task that took
            # over in the meantime — it does NOT restore _active_task back
            # to the chat task that was already running BEFORE this command
            # started and is STILL running after it closes. set_active_task
            # above overwrote _active_task to point at THIS command's task
            # the whole time its screen was open; once that task finishes,
            # _active_task genuinely becomes unset for the rest of the
            # chat turn — Ctrl+C has nothing to cancel even though the
            # model is still working. Re-registering the still-running chat
            # task here (if any) restores it as Ctrl+C's target.
            task = asyncio.create_task(_handle_input(text))

            def _on_command_done(_, t=task):
                app.clear_active_task(t)
                if _active_handle_task is not None:
                    app.set_active_task(_active_handle_task)
            app.set_active_task(task)
            task.add_done_callback(_on_command_done)
            return
        if _active_handle_task is not None and _waiting_for_model:
            # Model is thinking but hasn't generated yet — cancel and combine.
            # Original message was already echoed; show only the amendment and
            # suppress the duplicate echo on the combined re-run.
            combined = _current_text + "\n" + text
            task = _active_handle_task
            _active_handle_task = None  # prevent double-cancel
            _current_text = combined
            _suppress_echo = True
            task.cancel()
            await _pending.put(combined)
            console.print(f"[dim]  +++ You › {escape(text)}[/]")
        elif _active_handle_task is not None and _mid_turn_queue is not None:
            # Живой фиче-запрос (Claude Code-style "steer mid-turn"): ход уже
            # прошёл фазу "жду первый токен" (амменд выше не подошёл бы) —
            # текущий тул/генерация НЕ прерывается, сообщение подхватится
            # между шагами графа (mcp_agent/agent.py:_stream_round), а не
            # ждёт конца всего хода в _pending, как раньше.
            await _mid_turn_queue.put(text)
            console.print(f"[dim]  📤 передам по ходу: {escape(text)}[/]")
        else:
            await _pending.put(text)
        app.set_queue_size(_pending.qsize())

    async def _processor() -> None:
        nonlocal _active_handle_task
        while True:
            text = await _pending.get()
            app.set_queue_size(_pending.qsize())
            task = asyncio.create_task(_handle_input(text))
            _active_handle_task = task
            app.set_active_task(task)
            try:
                await task
            except asyncio.CancelledError:
                pass  # Ctrl+C — already handled inside _handle_input
            except Exception as e:
                err = str(e).encode("utf-8", errors="replace").decode("utf-8")
                console.print(f"[red] ✗ {err}[/]\n")
            finally:
                # clear_active_task, not set_active_task(None) -- same race
                # as the "/"-command branch above (_enqueue): a command task
                # started WHILE this chat task was running (e.g. /settings,
                # tracked separately) may still be the current _active_task
                # when this chat task finishes first. Guard on identity so
                # this only clears ITS OWN registration.
                app.clear_active_task(task)
                _active_handle_task = None
                _pending.task_done()

    app.set_submit_callback(_enqueue)

    processor = asyncio.create_task(_processor())
    try:
        await app.run_async()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        processor.cancel()
        try:
            await processor
        except asyncio.CancelledError:
            pass
        # Снимки правок (list_file_snapshots/restore_file_snapshot) —
        # рабочее состояние текущего сеанса, не долгоживущая история вроде
        # memory/episodic — чистим при выходе, чтобы они не накапливались
        # вечно в общей базе между запусками.
        clear_session_file_snapshots()
        # expert-streaming (порт 8090) — не системный демон вроде Ollama,
        # это ПОДПРОЦЕСС этого самого flowai (см. expert_streaming.py) —
        # умирать вместе с ним, а не переживать выход. Живой баг (отчёт
        # пользователя): без этого он оставался висеть после Ctrl+D/выхода,
        # и следующий запуск flowai видел порт занятым "каким-то другим
        # процессом" (ensure_running's health-check guard, см. там же) —
        # ложный "не запустился", хотя это был осиротевший процесс от
        # предыдущего же запуска. No-op, если expert-streaming не включён/
        # не запускался в этой сессии (stop_server сам проверяет _proc).
        expert_streaming.stop_server()

    console.print("\n[dim]Пока![/]")


def run() -> None:
    """Sync entry point for the `flowai` console script (pyproject.toml:
    [project.scripts]) — setuptools' generated launcher calls a plain
    callable with no args, it doesn't know to asyncio.run() a coroutine
    itself. Same body as the __main__ block below, which stays so
    `python3 cli.py` keeps working unchanged."""
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.stdout.write("\n")


if __name__ == "__main__":
    run()
