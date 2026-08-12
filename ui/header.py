import io
import os
import sys
import settings
from rich.console import Console
from rich.panel import Panel

from version import __version__


def _ollama_gpu_badge() -> str:
    """Возвращает метку GPU/CPU для Ollama на основе `ollama ps`."""
    try:
        import ollama as _ol
        running = _ol.ps()
        models = getattr(running, "models", [])
        if not models:
            return ""
        m = models[0]
        # size_vram показывает сколько VRAM занято; если > 0 — GPU используется
        vram = getattr(m, "size_vram", 0) or 0
        total = getattr(m, "size", 0) or 1
        if vram > 0:
            pct = int(vram / total * 100)
            gb = vram / 1024 ** 3
            return f" [green]GPU {pct}% ({gb:.1f}GB)[/]"
        else:
            return " [yellow]CPU[/]"
    except Exception:
        return ""


def _display_cwd() -> str:
    """os.getcwd() с домашней директорией, свёрнутой в '~' — то же
    сокращение, что показывает большинство шелл-промптов."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd == home:
        return "~"
    if cwd.startswith(home + os.sep):
        return "~" + cwd[len(home):]
    return cwd


def _gen3d_badge() -> str:
    """Статус /gen_model и /anim — движки там фиксированные (Hunyuan3D-2GP,
    Animato), выбора модели как у чата/vision/imggen нет, но настройки,
    реально влияющие на результат (полигонаж, источник скиннинга, offload-
    профиль), и модель, которая пишет код анимации (тот же chat_model,
    что и в обычном чате — не отдельная), стоит показывать явно."""
    if not settings.get("gen3d_enabled"):
        return "[dim]выкл[/]"
    faces   = settings.get("gen3d_target_faces")
    skin    = settings.get("gen3d_skin_source")
    profile = settings.get("gen3d_hunyuan_profile")
    chat    = settings.get("chat_model")
    return (f"[green]ВКЛ[/] ({faces} полиг · {skin} · offload {profile})"
            f"  [bright_black]·[/]  [bright_black]анимация:[/] [yellow]{chat}[/]")


def _voice_badge() -> str:
    """STT-модель показывается всегда (Alt+R доступен независимо от
    voice_mode), TTS/новая chat-модель — только когда голосовой режим реально
    включён, чтобы не путать с обычным chat_model в первом сегменте шапки."""
    stt = settings.get("stt_model")
    if settings.get("voice_mode"):
        voice_model = settings.get("voice_chat_model")
        return f"[yellow]stt {stt}[/] · [green]озвучка ВКЛ ({voice_model})[/]"
    return f"[yellow]stt {stt}[/] · [dim]озвучка выкл[/]"


def print_header(app=None) -> None:
    chat    = settings.get("chat_model")
    vision  = settings.get("vision_model")
    imggen  = settings.get("image_gen_model")
    device  = settings.get("image_gen_device")

    device_badge = (
        f"[green]{device.upper()}[/]"
        if (device == "cuda" and settings.CUDA_AVAILABLE)
        else f"[dim]{device.upper()}[/]"
    )
    gpu_badge = _ollama_gpu_badge()
    voice_badge = _voice_badge()
    gen3d_badge = _gen3d_badge()

    # Capture rendered output into a string buffer
    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, highlight=False)

    cap.print()
    cap.print(Panel(
        f"[bold cyan]FlowAI[/] [bright_black]v{__version__}[/]"
        f"  [bright_black]│[/]  [bright_black]чат:[/] [yellow]{chat}[/]{gpu_badge}"
        f"  [bright_black]│[/]  [bright_black]vision:[/] [yellow]{vision}[/]"
        f"  [bright_black]│[/]  [bright_black]imggen:[/] [yellow]{imggen}[/] {device_badge}"
        f"  [bright_black]│[/]  [bright_black]voice:[/] {voice_badge}"
        f"  [bright_black]│[/]  [bright_black]3D:[/] {gen3d_badge}",
        title=f"[bright_black]{_display_cwd()}[/]",
        subtitle="[bright_black]/gen · /img · /paste · /usage · /settings · /help[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))
    cap.print()

    ansi_str = buf.getvalue()

    if app is not None:
        app.replace_header(ansi_str)
    else:
        try:
            sys.stdout.buffer.write(ansi_str.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            sys.stdout.write(ansi_str)
