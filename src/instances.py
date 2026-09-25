"""
/instances (cli.py, ui/tui/instances_view.py) и GET/DELETE /api/v1/instances
(main.py) — какие модели сейчас реально загружены, на каком бэкенде, сколько
памяти держат, и ручная выгрузка любой из них.

Два бэкенда:
- Ollama — демон `ollama serve`, список через его /api/ps (size/size_vram
  на каждую модель), выгрузка — generate с keep_alive=0.
- llama.cpp — наш форк vendor/llama-expert-streaming (expert_streaming.py),
  отдельный процесс llama-server. Его модель Ollama не видит вообще, так что
  без этого пункта "что загружено" было бы занижено ровно в дефолтной
  конфигурации (expert_streaming_enabled=ВКЛ). RAM — RSS процесса; VRAM —
  по-процессная разбивка nvidia-smi, если доступна, иначе остаток "занято на
  GPU минус то, что заявила Ollama" (под WSL2 nvidia-smi не отдаёт
  --query-compute-apps, см. doctor.py:_nvidia_smi_gpu_memory).

Всё синхронное: вызывается из curses-меню (поток главного loop'а, см.
model_lifecycle.py) и из FastAPI через asyncio.to_thread.
"""
import os
import subprocess

import expert_streaming

_GIB = 1024 ** 3


def _ollama_client():
    import ollama
    return ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


def _process_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_smi(query: list[str]) -> list[list[str]] | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]


def _vram_by_pid() -> dict[int, int]:
    rows = _nvidia_smi(["--query-compute-apps=pid,used_memory"]) or []
    result = {}
    for row in rows:
        try:
            result[int(row[0])] = int(row[1]) * 1024 * 1024
        except (ValueError, IndexError):
            continue
    return result


def _gpu_used_bytes() -> int | None:
    rows = _nvidia_smi(["--query-gpu=memory.used"])
    try:
        return int(rows[0][0]) * 1024 * 1024 if rows else None
    except (ValueError, IndexError):
        return None


def list_instances() -> tuple[list[dict], list[str]]:
    """(instances, errors). Каждый instance: {id, backend, model, ram_bytes,
    vram_bytes, vram_estimated, details} — id стабилен между вызовами и
    передаётся обратно в unload_instance. errors — человекочитаемые причины,
    почему какой-то бэкенд не удалось опросить (например, демон Ollama не
    запущен), чтобы пустой список не выглядел как "ничего не загружено"."""
    instances: list[dict] = []
    errors: list[str] = []

    ollama_vram_total = 0
    try:
        for m in _ollama_client().ps().models:
            size = m.size or 0
            size_vram = m.size_vram or 0
            ollama_vram_total += size_vram
            gpu_pct = round(size_vram / size * 100) if size else 0
            details = f"{gpu_pct}% на GPU"
            expires = getattr(m, "expires_at", None)
            if expires is not None:
                details += f", выгрузится в {expires.astimezone().strftime('%H:%M')}"
            instances.append({
                "id": f"ollama:{m.model}",
                "backend": "ollama",
                "model": m.model,
                "ram_bytes": max(0, size - size_vram),
                "vram_bytes": size_vram,
                "vram_estimated": False,
                "details": details,
            })
    except Exception as e:
        errors.append(f"Ollama недоступна: {e}")

    state = expert_streaming.live_state()
    if state:
        pid = state["pid"]
        vram = _vram_by_pid().get(pid)
        estimated = False
        if vram is None:
            used = _gpu_used_bytes()
            if used is not None:
                vram = max(0, used - ollama_vram_total)
                estimated = True
        instances.append({
            "id": f"llama.cpp:{pid}",
            "backend": "llama.cpp",
            "model": state.get("model_tag") or "?",
            "ram_bytes": _process_rss_bytes(pid),
            "vram_bytes": vram,
            "vram_estimated": estimated,
            "details": f"pid {pid}, порт {state.get('port')}, контекст {state.get('num_ctx')}",
        })

    return instances, errors


def unload_instance(instance_id: str) -> str:
    """Выгружает инстанс по id из list_instances. Возвращает сообщение для
    пользователя. Сбрасывает кэш собранных агентов: они держат клиента,
    привязанного к уже выгруженной модели/убитому порту llama-server, и без
    этого следующий ход ушёл бы в мёртвое соединение вместо перезагрузки."""
    backend, _, ref = instance_id.partition(":")
    if backend == "ollama":
        try:
            _ollama_client().generate(model=ref, prompt="", keep_alive=0)
        except Exception as e:
            return f"не удалось выгрузить {ref}: {e}"
        msg = f"выгружена {ref} (ollama)"
    elif backend == "llama.cpp":
        if not expert_streaming.stop_server(include_adopted=True):
            return "llama-server уже не запущен"
        msg = "llama-server остановлен"
    else:
        return f"неизвестный инстанс: {instance_id}"
    try:
        from mcp_agent.agent_builder import invalidate_agent_caches
        invalidate_agent_caches()
    except Exception:
        pass
    return msg


def format_bytes(n: int | None, estimated: bool = False) -> str:
    if n is None:
        return "—"
    return f"{'~' if estimated else ''}{n / _GIB:.1f} GB"
