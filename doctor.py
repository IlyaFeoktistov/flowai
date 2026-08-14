"""
/doctor — единый health-check вместо разбросанных по коду и комментариям
ручных проверок ("надо посмотреть `ollama ps` вживую", см. CLAUDE.md про
_MEASURED_GPU_SHARE и ui/tui/settings.py). Один вызов агрегирует то, что
раньше нужно было проверять отдельно и вручную: жив ли Ollama-демон, стоит
ли выбранная модель, поднят ли expert-streaming backend (expert_streaming.py),
как модель реально разложена между GPU/CPU прямо сейчас, здоров ли
data_dir() (storage.py), и что вообще настроено (settings.py снапшот,
MCP-серверы из mcp_agent/config.py).

Каждая проверка — best-effort и независима от остальных: сбой одной (демон
недоступен, лишний бинарник не найден) не должен прятать результаты
остальных — оборачиваем каждую в try/except и продолжаем.
"""
import os
import shutil

import expert_streaming
import settings
import storage
from mcp_agent.config import build_mcp_connections

_OK, _WARN, _FAIL = "ok", "warn", "fail"
_ICON = {_OK: "[green]✓[/]", _WARN: "[yellow]⚠[/]", _FAIL: "[red]✗[/]"}


class _Check:
    def __init__(self, name: str, level: str, summary: str):
        self.name = name
        self.level = level
        self.summary = summary


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


async def _check_ollama_daemon() -> tuple[_Check, list | None]:
    """Возвращает саму проверку + список моделей из `ollama list` (или None,
    если демон недоступен) — переиспользуется следующей проверкой вместо
    повторного похода в сеть."""
    host = _ollama_host()
    try:
        import ollama
        client = ollama.AsyncClient(host=host)
        resp = await client.list()
        return _Check("Ollama-демон", _OK, f"отвечает на {host} ({len(resp.models)} моделей установлено)"), resp.models
    except ImportError:
        return _Check("Ollama-демон", _FAIL, "пакет ollama не установлен в этом окружении"), None
    except Exception as e:
        return _Check(
            "Ollama-демон", _FAIL,
            f"недоступен на {host}: {e} — проверь, что `ollama serve` запущен",
        ), None


def _check_chat_model(installed_models: list | None) -> _Check:
    chat_model = settings.get("chat_model")
    if installed_models is None:
        return _Check("Выбранная модель", _WARN, f"'{chat_model}' — не проверено (демон недоступен)")
    names = {m.model for m in installed_models}
    if chat_model in names:
        return _Check("Выбранная модель", _OK, f"'{chat_model}' установлена")
    return _Check(
        "Выбранная модель", _FAIL,
        f"'{chat_model}' не найдена в `ollama list` — `ollama pull {chat_model}` или смени модель в /settings",
    )


async def _check_live_gpu_split() -> _Check:
    """Аналог живого `ollama ps`, на который и так постоянно ссылаются
    комментарии в model_config.py/ui/tui/settings.py — вместо того чтобы
    просить пользователя вызвать это руками, отчёт делает это сам."""
    try:
        import ollama
        client = ollama.AsyncClient(host=_ollama_host())
        resp = await client.ps()
    except Exception as e:
        return _Check("Загруженные модели (VRAM)", _WARN, f"не удалось получить `ollama ps`: {e}")
    if not resp.models:
        return _Check("Загруженные модели (VRAM)", _WARN, "ни одна модель сейчас не резидентна в Ollama")
    parts = []
    for m in resp.models:
        size = m.size or 0
        size_vram = m.size_vram or 0
        gpu_pct = round(size_vram / size * 100) if size else 0
        parts.append(f"{m.model}: {size / (1024**3):.1f}GB · {gpu_pct}% GPU")
    return _Check("Загруженные модели (VRAM)", _OK, "; ".join(parts))


def _check_expert_streaming() -> _Check | None:
    """None (не проверка вообще, не FAIL) если фича выключена в /settings —
    "не настроено" не то же самое, что "настроено и сломано"."""
    if not settings.get("expert_streaming_enabled"):
        return None
    if not expert_streaming.is_built():
        return _Check(
            "Expert-streaming backend", _FAIL,
            f"expert_streaming_enabled=ВКЛ, но бинарник не собран ({expert_streaming.SERVER_BINARY})",
        )
    if not expert_streaming.is_running():
        return _Check(
            "Expert-streaming backend", _WARN,
            "собран, но не запущен в этом процессе (поднимется автоматически при первом ходе)",
        )
    healthy = expert_streaming._health_check(expert_streaming.DEFAULT_PORT)
    if healthy:
        return _Check("Expert-streaming backend", _OK, f"отвечает на порту {expert_streaming.DEFAULT_PORT}")
    return _Check(
        "Expert-streaming backend", _FAIL,
        f"процесс запущен, но /health на порту {expert_streaming.DEFAULT_PORT} не отвечает — см. {expert_streaming._LOG_PATH}",
    )


def _check_data_dir() -> _Check:
    data_dir = storage.data_dir()
    try:
        probe = data_dir / ".doctor_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        return _Check("Хранилище данных", _FAIL, f"{data_dir} недоступно для записи: {e}")

    db_path = data_dir / "flowai.db"
    db_size_mb = db_path.stat().st_size / (1024 ** 2) if db_path.is_file() else 0
    try:
        free_gb = shutil.disk_usage(data_dir).free / (1024 ** 3)
    except OSError:
        free_gb = None
    detail = f"{data_dir}, flowai.db {db_size_mb:.1f}MB"
    if free_gb is not None:
        detail += f", {free_gb:.1f}GB свободно на диске"
        if free_gb < 1:
            return _Check("Хранилище данных", _FAIL, detail + " — меньше 1GB, модели не скачаются/не догрузятся")
        if free_gb < 5:
            return _Check("Хранилище данных", _WARN, detail + " — меньше 5GB свободно")
    return _Check("Хранилище данных", _OK, detail)


def _check_mcp_servers() -> _Check:
    connections = build_mcp_connections(os.getcwd())
    missing = []
    for name, conn in connections.items():
        command = conn["command"]
        # command — либо абсолютный путь (свои mcp_agent/servers/*_server.py,
        # venv-скрипты вроде mcp-server-git), либо голое имя (npx) — резолвим
        # тем способом, который реально подходит каждому случаю.
        resolvable = os.path.isfile(command) if os.path.isabs(command) else shutil.which(command) is not None
        if not resolvable:
            missing.append(f"{name} ({command})")
    if missing:
        return _Check(
            "MCP-серверы", _FAIL,
            f"{len(connections) - len(missing)}/{len(connections)} настроены; не найден бинарник для: {', '.join(missing)}",
        )
    return _Check("MCP-серверы", _OK, f"{len(connections)} настроено: {', '.join(sorted(connections))}")


def _check_settings_snapshot() -> _Check:
    keys = ("chat_model", "num_ctx", "optimized_tools", "pipeline_mode", "voice_mode", "gen_agent_tools", "ask_permissions")
    parts = [f"{k}={settings.get(k)}" for k in keys]
    return _Check("Текущие настройки", _OK, ", ".join(parts))


async def run_doctor() -> str:
    ollama_check, installed_models = await _check_ollama_daemon()
    checks = [
        ollama_check,
        _check_chat_model(installed_models),
        await _check_live_gpu_split(),
        _check_expert_streaming(),
        _check_data_dir(),
        _check_mcp_servers(),
        _check_settings_snapshot(),
    ]
    checks = [c for c in checks if c is not None]

    ok = sum(1 for c in checks if c.level == _OK)
    warn = sum(1 for c in checks if c.level == _WARN)
    fail = sum(1 for c in checks if c.level == _FAIL)

    lines = [f"[bold]Doctor[/] — {ok} ок, {warn} предупреждений, {fail} ошибок\n"]
    for c in checks:
        lines.append(f"{_ICON[c.level]} [bold]{c.name}[/] — {c.summary}")
    return "\n".join(lines)
