"""
MCP-сервер: генерация и анимация 3D-моделей (Hunyuan3D-2GP + Blender +
UniRig + Animato), см. gen3d/pipeline.py и 3dtodo.md для полной истории
пайплайна и замеров VRAM.

В отличие от tools/gen_model.py (используется напрямую cli.py'шными
/gen_model, /animate_model, /gen_texture, в том же процессе, что и
tools/image_gen.py — может свободно печатать прогресс), этот файл —
отдельный подпроцесс, чей stdout — это сам MCP JSON-RPC протокол. Поэтому
generate_3d_model здесь принимает ТОЛЬКО путь к готовому изображению, а не
текстовый промпт: чтобы сгенерировать картинку с нуля, агент сам сначала
зовёт generate_image (другой подпроцесс, image_gen_server.py) —
переиспользуем существующий тул через композицию вызовов, а не дублируем
SDXL/FLUX pipeline-код здесь. generate_texture_for_model по той же причине
принимает только пути к уже существующим mesh и картинке.

Запуск: python3 -m mcp_agent.servers.gen_model_server
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import settings  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from gen3d.pipeline import PipelineError, run_gen_model, run_gen_texture, lod_paths_for  # noqa: E402
from gen3d.animato_client import animate as _animato_animate  # noqa: E402

mcp = FastMCP("gen_model")


def _slug(text: str) -> str:
    return f"{hash(text) & 0xFFFFFF:06x}"


@mcp.tool()
async def generate_3d_model(image_path: str = "", front: str = "", left: str = "", back: str = "",
                             right: str = "", rig: bool = False, raw: bool = False, lod: int = 0,
                             pbr_ai: bool | None = None) -> str:
    """Generate a textured 3D model (.glb), locally (Hunyuan3D-2GP + Blender
    retopology/texture rebake — an albedo texture, a tangent-space normal map
    AND an AO map capturing the detail decimation would otherwise discard,
    plus flat roughness/metallic factors — GPU, several minutes).

    Two mutually exclusive input modes:
    - image_path: single reference image (existing file — if you don't have
      one yet, call generate_image first and pass its output path here).
    - front/left/back/right: multiple views of the SAME object, fused into
      ONE consistent model (Hunyuan3D-2mv) instead of image_path's single
      view. Each is an EXISTING image file — no text-prompt generation for
      these, generate each view yourself first if needed. These are the
      model's own FIXED viewing angles, not arbitrary ones: front = camera
      facing the object head-on (0°), left = camera rotated 90° to see the
      object's left side, back = directly opposite front (180°), right = 90°
      the other way (270°). At least front is required; front+left+back is
      what actually gives a consistent multi-view result (matches the
      model's own reference examples) — right is rarely needed as a 4th.
      Adds its own extra GPU time for the multi-view checkpoint load.

    Set rig=True to also add a skeleton + skinning (UniRig + Blender
    Automatic Weights by default, a couple extra minutes) so the model is
    ready for animation via animate_3d_model. Set raw=True to skip
    retopology/texture-rebake and get Hunyuan3D-2GP's untouched output
    (hundreds of thousands of faces, NOT game-ready — for inspection only).
    Set lod=N to also generate N additional, progressively lower-poly
    variants as separate files (`{model}_lod1.glb` .. `{model}_lodN.glb`,
    each with its own baked albedo+normal), each level's face count halving
    the previous one's — useful for game engines' LOD groups. LOD variants
    are never rigged, even if rig=True. Set pbr_ai=True to replace the flat
    roughness/metallic factors with an AI-estimated metallicRoughnessTexture
    (SuperMat: renders 6 lit views of the model, estimates per-view
    roughness/metallic, projects them onto the mesh's real UV — a separate
    GPU-heavy subprocess, adds ~6-8 minutes; defaults to the gen3d_pbr_ai
    setting when not given, which itself defaults to on — requires
    vendor/supermat to be set up, see PipelineError if it isn't).
    Has no effect when raw=True, and is never applied to LOD variants."""
    if not settings.get("gen3d_enabled"):
        return "Error: 3D generation is disabled in /settings (\"gen_model включён\")"

    mv_views = {view: p for view, p in (("front", front), ("left", left), ("back", back), ("right", right)) if p}
    if mv_views and image_path:
        return "Error: pass either image_path or front/left/back/right, not both"
    if not mv_views and not image_path:
        return "Error: no image_path or front/left/back/right given"
    if mv_views:
        for view, path in mv_views.items():
            if not Path(path).is_file():
                return f"Error: file not found ({view}): {path}"
        slug_source = "|".join(f"{k}={v}" for k, v in sorted(mv_views.items()))
    else:
        src = Path(image_path)
        if not src.is_file():
            return f"Error: file not found: {image_path}"
        slug_source = image_path

    try:
        out_path = run_gen_model(
            image_path=image_path or None,
            images=mv_views or None,
            out_slug=_slug(slug_source),
            rig=rig,
            raw=raw,
            target_faces=settings.get("gen3d_target_faces"),
            profile=settings.get("gen3d_hunyuan_profile"),
            skin_source=settings.get("gen3d_skin_source"),
            lod=lod,
            pbr_ai=settings.get("gen3d_pbr_ai") if pbr_ai is None else pbr_ai,
        )
        msg = f"3D model saved: {out_path}"
        if lod:
            lod_list = ", ".join(str(p) for p in lod_paths_for(out_path, lod))
            msg += f"\nLOD models: {lod_list}"
        return msg
    except PipelineError as e:
        return f"Error: {e}"


@mcp.tool()
async def animate_3d_model(model_path: str, motion: str) -> str:
    """Animate an EXISTING rigged 3D model (.glb with a skeleton — the output
    of generate_3d_model called with rig=True) per a natural-language motion
    description (e.g. "wave with the right arm"), locally via Animato + the
    current chat model. `model_path` must already exist and already be
    rigged — this does not add a rig itself."""
    if not settings.get("gen3d_enabled"):
        return "Error: 3D generation is disabled in /settings (\"gen_model включён\")"
    src = Path(model_path)
    if not src.is_file():
        return f"Error: file not found: {model_path}"
    try:
        out_path = _animato_animate(
            src, motion,
            chat_model=settings.get("chat_model"),
            out_slug=_slug(model_path + motion),
        )
        return f"Animated model saved: {out_path}"
    except PipelineError as e:
        return f"Error: {e}"


@mcp.tool()
async def generate_texture_for_model(model_path: str, image_path: str) -> str:
    """Repaint an EXISTING 3D mesh (.glb) with a new texture generated from a
    reference image, locally (Hunyuan3D-2GP's paint pipeline alone, GPU, a
    few minutes) — skips shape generation entirely, so it's much faster than
    generate_3d_model and works on any mesh, not just one this tool produced.
    Both `model_path` and `image_path` MUST already exist. The mesh's UV is
    rebuilt from scratch (xatlas) as part of this — any existing UV layout is
    discarded, not reused."""
    if not settings.get("gen3d_enabled"):
        return "Error: 3D generation is disabled in /settings (\"gen_model включён\")"
    mesh = Path(model_path)
    if not mesh.is_file():
        return f"Error: file not found: {model_path}"
    image = Path(image_path)
    if not image.is_file():
        return f"Error: file not found: {image_path}"
    try:
        out_path = run_gen_texture(
            model_path,
            image_path,
            out_slug=_slug(model_path + image_path),
            profile=settings.get("gen3d_hunyuan_profile"),
        )
        return f"Textured model saved: {out_path}"
    except PipelineError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
