"""
Выгрузка резидентных моделей из памяти/VRAM — ручная (кнопка "выгрузить
модели" в /settings, см. ui/tui/settings.py) и автоматическая (при
выключении voice_mode, см. settings.py:set_value — переключение обратно на
тяжёлую кодовую модель должно освобождать всё остальное, а не просто
подгружать её ПОВЕРХ уже занятой памяти).

Всё здесь синхронное на входе (обе точки вызова — curses-код: кнопка
в /settings и settings.py:set_value при выключении voice_mode), но часть
целей достаётся только через asyncio:
- Whisper (ui/audio.py), SDXL-пайплайн /gen (tools/image_gen.py) и
  MusicGen-копия /music/  /music_gen (mcp_agent/servers/music_server.py,
  импортированная напрямую в ЭТОТ процесс, см. ui/music_stream.py/cli.py)
  — обычные модуль-level глобалы В ЭТОМ процессе, их "выгрузка" — просто
  присвоение None + сборка мусора, никакого моста не нужно.
- Ollama-модели (chat/vision) живут в отдельном процессе (демон
  `ollama serve`), выгружаются через его же API (generate с keep_alive=0)
  синхронным клиентом — в отличие от agent_builder._evict_ollama_model,
  который живёт в async-мире агента и использует AsyncClient.
- SDXL/MusicGen-копии, резидентные в MCP-подпроцессах image_gen_server.py/
  music_server.py (те, что использует САМ агент через тулы generate_image/
  generate_music) — другой процесс, достать до них можно только вызовом их
  собственных unload_* MCP-тулов через уже установленное соединение
  (agent_builder._tools_cache), а это asyncio. _run_coro_blocking ниже —
  мост: настоящий блокирующий asyncio.run() здесь не подходит, потому что
  обе точки вызова уже исполняются НА потоке главного event loop
  (prompt_toolkit's run_in_terminal гонит curses-меню прямо на нём, без
  executor'а по умолчанию — asyncio.run() внутри уже бегущего loop'а того
  же потока бросает RuntimeError). Отдельный ОС-поток со своим независимым
  loop технически избегает ЭТОЙ ошибки, но не более глубокой проблемы:
  пока главный loop блокирован (синхронный curses-код в его стеке вызовов),
  никто не "прокручивает" его — включая фоновую задачу, которая читает
  stdout MCP-подпроцесса и раздаёт ответы ожидающим Future. Наш новый поток
  пишет запрос в тот же stdin, но ответ читать физически некому, пока
  главный loop не разблокируется — то есть звонок в это соединение с
  большой вероятностью просто провисит до таймаута, а не отработает. Отсюда
  жёсткий timeout ниже (не бесконечный t.join()) — это ЧЕСТНО best-effort
  попытка, которая в текущей архитектуре скорее всего ничего не сделает, а
  не гарантированная выгрузка; она не должна вешать /settings навечно, если
  не сработает. Прямая выгрузка Whisper/tools/image_gen/music_server ниже
  — вот что реально работает всегда, независимо от этого моста.
"""
import os
import threading

import settings


def unload_ollama_model(model_name: str) -> None:
    """Синхронный аналог agent_builder._evict_ollama_model — тот держит
    AsyncClient ради async-мира агента, а вызывающим здесь (curses-меню,
    settings.set_value) обычный клиент удобнее, чем поднимать loop."""
    if not model_name:
        return
    try:
        import ollama
        client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        client.generate(model=model_name, prompt="", keep_alive=0)
    except Exception:
        pass


_SUBPROCESS_BRIDGE_TIMEOUT = 5.0


def _run_coro_blocking(coro, timeout: float = _SUBPROCESS_BRIDGE_TIMEOUT):
    """Runs `coro` to completion on a FRESH event loop in a separate OS
    thread, blocking the caller for at most `timeout` seconds. Returns the
    result, or None on timeout/error — never raises, and never blocks
    forever (see the module docstring: the coroutine this actually carries
    is a best-effort attempt likely to time out in the current
    architecture, not a guaranteed call — an unbounded t.join() here would
    mean a failed attempt freezes /settings indefinitely instead of just
    doing nothing)."""
    result: dict = {}

    def _target():
        import asyncio
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None  # timed out — leave the daemon thread to die with the process
    if "error" in result:
        return None
    return result.get("value")


def unload_agent_subprocess_models() -> list[str]:
    """Bridges into the agent's own MCP subprocesses (image_gen_server.py/
    music_server.py) to unload the SDXL/MusicGen copies backing its own
    generate_image/generate_music tool calls — separate from this
    process's own /gen and /music(_gen) copies of the same models, which
    unload_idle_models handles directly below without any bridge. Silently
    returns [] if those subprocesses were never even started this session
    (see agent_builder._unload_subprocess_models), on any bridge failure,
    or on timeout — this is a best-effort memory reclaim, not a critical
    path, and given how it's called (see module docstring on
    _run_coro_blocking) it will typically time out and do nothing while
    /settings' curses menu is actually open. Left in because it costs
    nothing to attempt and may still work from call sites that aren't
    blocking the main loop."""
    try:
        from mcp_agent.agent_builder import _unload_subprocess_models
        return _run_coro_blocking(_unload_subprocess_models()) or []
    except Exception:
        return []


def unload_idle_models() -> list[str]:
    """Выгружает всё, что резидентно В ЭТОМ процессе (Whisper, /gen-пайплайн,
    MusicGen-копия /music) плюс всё, что резидентно в агентских MCP-
    подпроцессах (SDXL/MusicGen копии за generate_image/generate_music),
    плюс все Ollama-теги, отличные от ТЕКУЩЕЙ chat_model (voice_chat_model,
    vision_model) — то есть "всё кроме активной кодовой/чат-модели прямо
    сейчас", везде, куда можно дотянуться. Возвращает список человекочитаемых
    пунктов о том, что реально было выгружено — пустой список значит "и так
    ничего лишнего не висело"."""
    freed: list[str] = []

    from ui.audio import unload_whisper
    if unload_whisper():
        freed.append("распознавание речи (Whisper)")

    from tools.image_gen import unload_pipe
    if unload_pipe():
        freed.append("генерация картинок /gen (SDXL/FLUX)")

    from mcp_agent.servers.music_server import unload_model as _unload_music_direct
    if _unload_music_direct():
        freed.append("генерация музыки /music, /music_gen (MusicGen)")

    freed.extend(unload_agent_subprocess_models())

    current_chat = settings.get("chat_model")

    voice_chat = settings.get("voice_chat_model")
    if voice_chat and voice_chat != current_chat:
        unload_ollama_model(voice_chat)
        freed.append(f"голосовая модель Ollama ({voice_chat})")

    vision = settings.get("vision_model")
    if vision and vision != current_chat:
        unload_ollama_model(vision)
        freed.append(f"vision-модель Ollama ({vision})")

    return freed
