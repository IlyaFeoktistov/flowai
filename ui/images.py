import base64
import re
import subprocess
import tempfile
from pathlib import Path

# Системный tmp, а не generated/uploads внутри проекта — вставленная из
# буфера картинка не обязана вообще существовать как файл где-либо (скопирована
# из браузера/скриншот, а не сохранена пользователем) и не является
# результатом работы этого проекта, как generate_image/edit_image — это
# входные, а не выходные данные, им место в tmp. Тот же паттерн, что уже
# использует mcp_agent/config.py:_LOG_DIR для логов MCP-серверов. Путь не
# зависит от os.getcwd() (у cli.py это repo_path ПОЛЬЗОВАТЕЛЯ, см.
# config.py) — analyze_image/edit_image (отдельные подпроцессы) получают
# абсолютный путь напрямую от модели и понятия не имеют, откуда запущен cli.py.
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "flowai-uploads"

# ── Хранилище вставленных изображений ────────────────────────────────────────
# Единственный источник истины — файл под _UPLOAD_DIR, без дублирующей
# base64-копии в памяти: analyze_image/edit_image всё равно читают только
# файл (отдельный подпроцесс), а resolve_image_paths ниже отличает реальный
# плейсхолдер от буквально введённого пользователем текста "[Image-99]"
# простой проверкой path.exists(), не требуя отдельного реестра ключей.
_counter = 0

_PATH_RE = re.compile(r'\[(Image-\d+)\]')


def store_image(b64: str) -> str:
    """Сохраняет base64-изображение на диск под generated/uploads/, возвращает
    плейсхолдер '[Image-N]'."""
    global _counter
    _counter += 1
    key = f"Image-{_counter}"

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (_UPLOAD_DIR / f"{key}.png").write_bytes(base64.b64decode(b64))

    return f"[{key}]"


def resolve_image_paths(text: str) -> str:
    """Заменяет каждый '[Image-N]' в тексте на абсолютный путь сохранённого
    файла — тот же паттерн, что ui/paste_store.py:resolve_pastes для текстовых
    вставок. Модель получает обычный путь как текст своего же сообщения и
    передаёт его тулам (analyze_image/edit_image) так же, как repo_path."""
    def _sub(m: re.Match) -> str:
        path = _UPLOAD_DIR / f"{m.group(1)}.png"
        return str(path) if path.exists() else m.group(0)
    return _PATH_RE.sub(_sub, text)


def clear_store() -> None:
    global _counter
    _counter = 0


# ── Получение изображения из буфера обмена ───────────────────────────────────

def get_clipboard_image() -> str | None:
    # WSL2/Windows: PowerShell -STA is required for WinForms Clipboard access.
    # Tries 'PNG' format first (Snipping Tool / Win+Shift+S), then CF_BITMAP fallback.
    try:
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$do=[System.Windows.Forms.Clipboard]::GetDataObject();"
            "if($do -ne $null){"
            "  if($do.GetDataPresent('PNG')){"
            "    $s=$do.GetData('PNG');"
            "    $ms=New-Object System.IO.MemoryStream;"
            "    $s.CopyTo($ms);"
            "    [Console]::WriteLine([Convert]::ToBase64String($ms.ToArray()))"
            "  } elseif([System.Windows.Forms.Clipboard]::GetImage() -ne $null){"
            "    $img=[System.Windows.Forms.Clipboard]::GetImage();"
            "    $ms=New-Object System.IO.MemoryStream;"
            "    $img.Save($ms,[System.Drawing.Imaging.ImageFormat]::Png);"
            "    [Console]::WriteLine([Convert]::ToBase64String($ms.ToArray()))"
            "  }"
            "}"
        )
        r = subprocess.run(
            ["powershell.exe", "-NonInteractive", "-STA", "-Command", ps],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            out = r.stdout.decode("utf-8-sig", errors="ignore").strip()
            if out:
                return out
    except Exception:
        pass

    # Linux: xclip
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            capture_output=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout:
            return base64.b64encode(r.stdout).decode()
    except Exception:
        pass

    return None


def copy_to_clipboard(text: str) -> bool:
    """Копирует текст в системный буфер обмена. Возвращает True при успехе."""
    try:
        # clip.exe (WSL/Windows) читает stdin в активной кодовой странице
        # консоли, а не как UTF-8 — любой не-ASCII байт (кириллица) превращается
        # в мусор. Буфер обмена Windows хранит текст в UTF-16LE, так что отдаём
        # clip.exe сразу UTF-16LE — это обходит переинтерпретацию кодовой страницы.
        r = subprocess.run(["clip.exe"], input=text.encode("utf-16-le"), timeout=3)
        if r.returncode == 0:
            return True
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), timeout=3,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass

    return False


def paste_image_from_clipboard() -> str | None:
    """Берёт изображение из буфера, сохраняет в store, возвращает '[Image-N]' или None."""
    b64 = get_clipboard_image()
    if not b64:
        return None
    return store_image(b64)


def load_image_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return base64.b64encode(p.read_bytes()).decode()
