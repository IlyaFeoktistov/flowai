"""/gen_model, /animate_model — CLI-direct path (used by cli.py, outside the
agent tool-calling pipeline), mirroring tools/image_gen.py's split from its
MCP counterpart (mcp_agent/servers/gen_model_server.py): this file is safe to
sprinkle console.print/TUI progress into because it always runs in-process in
the CLI; the MCP server runs in a separate subprocess whose stdout IS the
MCP JSON-RPC channel, so it must stay silent and call gen3d/pipeline.py
directly instead of going through here.
"""
import asyncio
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import settings
from tools.base import ok, fail
from tools.image_gen import generate_image
from ui.console import get_app, fmt_elapsed
from gen3d.pipeline import PipelineError, run_gen_model, run_gen_texture, lod_paths_for
from gen3d.animato_client import animate as _animato_animate

_SAVED_RE = re.compile(r"Image saved: (.+)$")

# Session-only (not persisted): the most recently generated model, so /anim
# can just take a motion description without repeating a path every time.
_last_model_path: str | None = None

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# run_gen_model's on_stage keys -> what the user sees. Hunyuan3D-2GP alone
# runs ~7-8 min (see 3dtodo.md) and rig/retopo/rebake each add real time on
# top -- without this, the TUI just goes silent for minutes after the initial
# "генерирую..." line, which reads as hung/done rather than still working.
_STAGE_LABELS = {
    "image":        "генерирую опорное изображение...",
    "mesh":         "генерирую меш и текстуру (Hunyuan3D-2GP, до ~8 мин)...",
    "retopo":       "ретопология (снижаю полигонаж)...",
    "rebake":       "перезапекаю текстуру на новую UV...",
    "pbr_ai":       "AI-оценка roughness/metallic (SuperMat: рендер ракурсов + оценка + проекция на UV, до ~8 мин)...",
    "rig_skeleton": "риг: строю скелет (UniRig)...",
    "rig_skin":     "риг: скиннинг...",
    "animate":      "анимирую (Animato + локальная модель)...",
    "texture":      "генерирую текстуру по референсу (Hunyuan3D-Paint, до ~6 мин)...",
}


class _StageSpinner:
    """Ticking status line for /gen_model & /anim's multi-minute, multi-stage
    pipelines -- same idea as tools/image_gen.py's _spin_download (TUI:
    app.set_stats(), plain terminal: \\r-overwritten line), but with a
    swappable stage label instead of a download progress bar. `set()` is
    called from the pipeline's executor thread (see run_gen_model's on_stage
    docstring), the ticker task reads it from the event loop -- a plain
    string attribute is enough here, no lock: CPython reference assignment is
    atomic and this is a UI label, not a correctness-sensitive value."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._t0 = time.monotonic()

    def set(self, label: str) -> None:
        self.label = label

    async def _run(self) -> None:
        app = get_app()
        i = 0
        try:
            while True:
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                elapsed = time.monotonic() - self._t0
                stat = f"\033[2m{frame} {self.label}  {fmt_elapsed(elapsed)}\033[0m"
                if app is not None:
                    app.set_stats(stat)
                else:
                    sys.stdout.buffer.write(f"\r{stat}".encode())
                    sys.stdout.buffer.flush()
                await asyncio.sleep(0.2)
                i += 1
        except asyncio.CancelledError:
            if app is not None:
                app.set_stats("")
            else:
                sys.stdout.buffer.write(b"\r\033[K")
                sys.stdout.buffer.flush()


@asynccontextmanager
async def _stage_spinner(initial_label: str):
    spinner = _StageSpinner(initial_label)
    task = asyncio.create_task(spinner._run())
    try:
        yield spinner
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _slug(text: str) -> str:
    return f"{hash(text) & 0xFFFFFF:06x}"


async def _resolve_image(prompt_or_path: str) -> str:
    """If prompt_or_path is an existing file, use it directly as the
    image-to-3D reference. Otherwise treat it as a text prompt and generate
    one via the existing generate_image tool (reused, not duplicated) --
    [Image-N] placeholders are already resolved to real paths by cli.py
    before this is called."""
    p = Path(prompt_or_path)
    if p.is_file():
        return str(p)
    result = await generate_image({"prompt": prompt_or_path})
    if result["status"] != "ok":
        raise PipelineError(f"couldn't generate a source image: {result['error']}")
    m = _SAVED_RE.search(result["output"])
    if not m:
        raise PipelineError(f"unexpected image_gen result: {result['output']}")
    return m.group(1)


async def generate_3d_model(args: dict) -> dict:
    """args: either "prompt_or_path" (single-view, text prompt or existing
    image), or one-to-four of "front"/"left"/"back"/"right" (multi-view,
    Hunyuan3D-2mv -- existing image files only, no text-prompt generation for
    these, see gen3d/hunyuan_wrapper.py's own docstring for the view
    convention) -- never both."""
    global _last_model_path
    if not settings.get("gen3d_enabled"):
        return fail("3D generation is disabled — enable it in /settings (\"gen_model включён\")")

    mv_views = {view: (args.get(view) or "").strip() for view in ("front", "left", "back", "right")}
    mv_views = {k: v for k, v in mv_views.items() if v}
    prompt_or_path = (args.get("prompt_or_path") or "").strip()

    if not mv_views and not prompt_or_path:
        return fail("No prompt or image path specified for 3D generation")
    if mv_views:
        for view, path in mv_views.items():
            if not Path(path).is_file():
                return fail(f"File not found ({view}): {path}")

    rig = bool(args.get("rig", False))
    raw = bool(args.get("raw", False))
    lod = int(args.get("lod", 0) or 0)
    pbr_ai = bool(args.get("pbr_ai", settings.get("gen3d_pbr_ai")))

    loop = asyncio.get_event_loop()
    cancel_event = threading.Event()
    app = get_app()

    image_path = None
    if mv_views:
        slug_source = "|".join(f"{k}={v}" for k, v in sorted(mv_views.items()))
        initial_label = "использую готовые ракурсы..."
    else:
        slug_source = prompt_or_path
        initial_label = _STAGE_LABELS["image"] if not Path(prompt_or_path).is_file() else "использую изображение..."

    try:
        async with _stage_spinner(initial_label) as spin:
            if not mv_views:
                image_path = await _resolve_image(prompt_or_path)

            def _on_stage(key: str) -> None:
                if key.startswith("lod"):
                    spin.set(f"генерирую LOD{key[3:]} (ретопология + перезапекание текстур)...")
                else:
                    spin.set(_STAGE_LABELS.get(key, key))

            spin.set(_STAGE_LABELS["mesh"] if not mv_views else
                      f"генерирую единую модель из {len(mv_views)} ракурсов (Hunyuan3D-2mv, до ~8 мин)...")
            # Registered only around the actual subprocess chain (not image
            # resolution above) -- Ctrl+C here kills gen3d/pipeline.py's
            # currently-running child process instead of leaving it to finish
            # in the background, see ui/app.py's set_gen3d_active docstring.
            if app is not None:
                app.set_gen3d_active(True, cancel_event.set)
            try:
                out_path = await loop.run_in_executor(
                    None,
                    lambda: run_gen_model(
                        image_path=image_path,
                        images=mv_views or None,
                        out_slug=_slug(slug_source),
                        rig=rig,
                        raw=raw,
                        target_faces=settings.get("gen3d_target_faces"),
                        profile=settings.get("gen3d_hunyuan_profile"),
                        skin_source=settings.get("gen3d_skin_source"),
                        lod=lod,
                        pbr_ai=pbr_ai,
                        on_stage=_on_stage,
                        cancel_event=cancel_event,
                    ),
                )
            finally:
                if app is not None:
                    app.set_gen3d_active(False)
        _last_model_path = str(out_path)
        msg = f"3D model saved: {out_path}"
        if lod:
            lod_list = ", ".join(str(p) for p in lod_paths_for(out_path, lod))
            msg += f"\nLOD models: {lod_list}"
        return ok(msg)
    except PipelineError as e:
        return fail(str(e))


async def animate_3d_model(args: dict) -> dict:
    global _last_model_path
    if not settings.get("gen3d_enabled"):
        return fail("3D generation is disabled — enable it in /settings (\"gen_model включён\")")
    model_path = (args.get("model_path") or "").strip() or _last_model_path
    motion = (args.get("motion") or "").strip()
    if not model_path:
        return fail("No model to animate — run /gen_model --rig first, or pass a path")
    if not motion:
        return fail("No motion description specified")

    src = Path(model_path)
    if not src.is_file():
        return fail(f"File not found: {model_path}")

    loop = asyncio.get_event_loop()

    try:
        async with _stage_spinner(_STAGE_LABELS["animate"]):
            out_path = await loop.run_in_executor(
                None,
                lambda: _animato_animate(
                    src, motion,
                    chat_model=settings.get("chat_model"),
                    out_slug=_slug(model_path + motion),
                ),
            )
        _last_model_path = str(out_path)
        return ok(f"Animated model saved: {out_path}")
    except PipelineError as e:
        return fail(str(e))


async def generate_texture_for_model(args: dict) -> dict:
    """/gen_texture: repaint an EXISTING mesh from a reference image, via
    Hunyuan3D-2GP's paint pipeline alone -- no shape generation involved."""
    global _last_model_path
    if not settings.get("gen3d_enabled"):
        return fail("3D generation is disabled — enable it in /settings (\"gen_model включён\")")
    model_path = (args.get("model_path") or "").strip()
    image_path = (args.get("image_path") or "").strip()
    if not model_path or not image_path:
        return fail("Both a mesh (.glb) and a reference image are required")

    mesh = Path(model_path)
    image = Path(image_path)
    if not mesh.is_file():
        return fail(f"File not found: {model_path}")
    if not image.is_file():
        return fail(f"File not found: {image_path}")

    loop = asyncio.get_event_loop()
    cancel_event = threading.Event()
    app = get_app()

    try:
        async with _stage_spinner(_STAGE_LABELS["texture"]):
            if app is not None:
                app.set_gen3d_active(True, cancel_event.set)
            try:
                out_path = await loop.run_in_executor(
                    None,
                    lambda: run_gen_texture(
                        str(mesh),
                        str(image),
                        out_slug=_slug(model_path + image_path),
                        profile=settings.get("gen3d_hunyuan_profile"),
                        cancel_event=cancel_event,
                    ),
                )
            finally:
                if app is not None:
                    app.set_gen3d_active(False)
        _last_model_path = str(out_path)
        return ok(f"Textured model saved: {out_path}")
    except PipelineError as e:
        return fail(str(e))
