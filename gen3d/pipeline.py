"""Orchestrates the /gen_model pipeline: image-to-3D generation (Hunyuan3D-2GP)
-> retopology -> texture rebake -> optional rig (UniRig skeleton + Blender
Automatic Weights, or full UniRig skin model). Every heavy step runs as a
subprocess in its own vendor/*/venv or via the system Blender -- see
setup.py's module docstring for why they can't share flowAI's own venv.

All functions here are synchronous (blocking on subprocess.Popen via
_run_subprocess, not subprocess.run -- see that function's own docstring for
why) -- callers (tools/gen_model.py, mcp_agent/servers/gen_model_server.py)
wrap them in loop.run_in_executor, the same pattern tools/image_gen.py uses
for its own blocking pipeline calls. An optional cancel_event threaded through
every stage lets a caller actually kill the current child process early
(rather than just abandoning the awaiting coroutine, which leaves the child
running in the background) -- see PipelineCancelled's docstring.
"""
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

FLOWAI_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = FLOWAI_ROOT / "vendor"
BLENDER_SCRIPTS = Path(__file__).resolve().parent / "blender_scripts"
GENERATED_MODELS_DIR = FLOWAI_ROOT / "generated" / "models"

HUNYUAN_DIR = VENDOR_DIR / "hunyuan3d-2gp"
UNIRIG_DIR = VENDOR_DIR / "unirig"
ANIMATO_DIR = VENDOR_DIR / "animato"
SUPERMAT_DIR = VENDOR_DIR / "supermat"


class PipelineError(RuntimeError):
    """A pipeline stage failed -- message is meant to be shown to the user/model
    as-is (includes which stage and, where useful, the missing-setup hint)."""


class PipelineCancelled(PipelineError):
    """Raised when cancel_event fired while a subprocess was running -- the
    child was actually killed, not left running in the background. Ctrl+C
    during /gen_model used to only cancel the asyncio task awaiting this
    pipeline (see tools/gen_model.py) -- that unblocks the UI immediately but
    never reached the blocking subprocess.run() calls running inside a
    run_in_executor thread, so the Blender/Hunyuan3D-2GP/UniRig/SuperMat child
    process kept running to completion in the background regardless (same
    class of bug ui/music_stream.py already documents for /music). Callers
    don't need to catch this specially -- asyncio.CancelledError from the
    cancelled task unwinds the awaiting coroutine before this exception would
    even be observed; it exists so _run_subprocess has a way to stop the
    poll loop below and skip any further pipeline stages."""


def _free_gpu_for_subprocess() -> None:
    """Two other things can be sitting resident in VRAM when a Hunyuan3D-2GP/
    UniRig subprocess is about to start, on a card this pipeline was tuned
    against (6 GB, see 3dtodo.md's measurements -- all taken with nothing
    else resident):

    - Ollama's chat model -- a separate long-running daemon process, freed
      via its own HTTP API (keep_alive=0), reachable the same way regardless
      of which process calls this (CLI-direct or an MCP subprocess).
    - The /gen text-to-image pipeline (SDXL/FLUX) -- tools/image_gen.py's
      resident `_pipe` module global, built lazily on first use and NEVER
      unloaded on its own afterwards (see that module's unload_pipe()
      docstring). On the CLI-direct path (tools/gen_model.py calling
      run_gen_model right after tools/image_gen.generate_image resolved a
      text prompt into an image -- see tools/gen_model.py's _resolve_image)
      this is the SAME process, so importing and calling unload_pipe() here
      reaches the real thing. On the MCP path (gen_model_server.py), this
      subprocess never calls generate_image itself -- the agent's own
      generate_image tool call runs in the SEPARATE image_gen_server.py
      subprocess, whose resident pipe isn't reachable from here at all (no
      IPC between sibling MCP subprocesses); the import below is then a
      harmless no-op against this process's own always-empty pipe. Fixing
      that path needs the agent orchestration layer itself to unload
      image_gen_server's pipe before dispatching generate_3d_model/
      generate_texture_for_model, which is out of scope for this function.

    Without this, one CUDA context (Ollama's daemon, or this process's own
    SDXL/FLUX) can sit on VRAM the Hunyuan3D-2GP/UniRig subprocess needs,
    turning "risk of OOM" into a near-guaranteed one -- both reload lazily on
    next use, so unloading here costs a slower next call, not correctness."""
    try:
        import settings
        import model_lifecycle
        model_lifecycle.unload_ollama_model(settings.get("chat_model"))
    except Exception:
        pass
    try:
        from tools.image_gen import unload_pipe
        unload_pipe()
    except Exception:
        pass


def _blender_bin() -> str:
    path = shutil.which("blender")
    if not path:
        raise PipelineError(
            "Blender not found. Install the official build (NOT apt's package -- "
            "its Cycles bake is broken for skinned meshes, see 3dtodo.md): "
            "sudo snap install blender --classic"
        )
    return path


def _require_vendor(dir_: Path, component: str) -> None:
    if not (dir_ / "venv").exists() and not (dir_ / ".venv").exists() and component != "animato":
        raise PipelineError(f"{component} isn't set up yet. Run: python3 setup.py --only {component}")
    if component == "animato" and not dir_.exists():
        raise PipelineError("animato isn't set up yet. Run: python3 setup.py --only animato")


def _run_subprocess(cmd: list[str], timeout: int, cancel_event: "threading.Event | None" = None,
                     cwd: Path | None = None, env: dict | None = None,
                     error_label: str = "subprocess") -> None:
    """subprocess.run() replacement that can actually be killed early.
    subprocess.run(timeout=...) only raises AFTER the fact (the child keeps
    running the whole time regardless), and it has no way to check anything
    ELSE mid-flight -- polls proc.poll() in a loop instead so cancel_event can
    be checked in between, and kills the child the moment it's set rather than
    letting it run to completion in the background (see PipelineCancelled's
    own docstring for why that distinction matters).

    stdout/stderr go to real temp files, NOT stdout=PIPE/stderr=PIPE -- a pipe
    has a small fixed OS buffer (~64KB on Linux); a verbose child (Hunyuan3D-
    2GP's own model-loading logs alone are well past that) blocks on write()
    once it fills up, and nothing was draining the pipe between poll() calls
    below (only proc.communicate() at the very end would have) -- confirmed
    live: a real /gen_model run hung indefinitely with the child stuck in
    pipe_write (see /proc/<pid>/wchan), not actually making progress despite
    looking merely slow. Regular files have no such fixed-size wall."""
    import tempfile
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out_f, \
         tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err_f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out_f, stderr=err_f, text=True)
        start = time.monotonic()
        cancelled = False
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                proc.kill()
                break
            if time.monotonic() - start > timeout:
                proc.kill()
                proc.wait()
                raise PipelineError(f"{error_label} timed out after {timeout}s")
            time.sleep(0.3)
        proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read()
        stderr = err_f.read()
    if cancelled:
        raise PipelineCancelled(f"{error_label} cancelled")
    if proc.returncode != 0:
        raise PipelineError(f"{error_label} failed:\n{stdout[-2000:]}\n{stderr[-2000:]}")


def _run_blender_script(script: str, args: list[str], timeout: int,
                         cancel_event: "threading.Event | None" = None) -> None:
    cmd = [_blender_bin(), "--background", "--python", str(BLENDER_SCRIPTS / script), "--", *args]
    _run_subprocess(cmd, timeout, cancel_event=cancel_event, error_label=f"blender {script}")


def generate_mesh_and_texture(image_path: str, out_glb: Path, profile: int = 4, timeout: int = 1800,
                               cancel_event: "threading.Event | None" = None) -> None:
    """Image-to-3D via Hunyuan3D-2GP: raw mesh + baked albedo texture."""
    _require_vendor(HUNYUAN_DIR, "hunyuan3d-2gp")
    _free_gpu_for_subprocess()
    venv_python = HUNYUAN_DIR / "venv" / "bin" / "python"
    cmd = [
        str(venv_python), str(Path(__file__).resolve().parent / "hunyuan_wrapper.py"),
        "--vendor-dir", str(HUNYUAN_DIR),
        "--image", str(Path(image_path).resolve()),
        "--output", str(out_glb.resolve()),
        "--profile", str(profile),
    ]
    _run_subprocess(cmd, timeout, cancel_event=cancel_event, error_label="Hunyuan3D-2GP generation")


def generate_mesh_and_texture_mv(images: dict[str, str], out_glb: Path, profile: int = 4, timeout: int = 1800,
                                  cancel_event: "threading.Event | None" = None) -> None:
    """Multi-view image-to-3D via Hunyuan3D-2GP: several views of ONE object
    (keys: "front"/"left"/"back"/"right", Hunyuan3D-2mv's own fixed 0/90/180/
    270-degree convention -- not arbitrary angles) fused into a SINGLE shape,
    instead of generate_mesh_and_texture()'s independent-batch behavior for
    multiple images (see cli.py's own "@ref @ref @ref = batch" comment on why
    that's a real, separate limitation this sidesteps by using a different
    Hunyuan3D checkpoint + image processor -- see hunyuan_wrapper.py's own
    docstring for the mechanism)."""
    _require_vendor(HUNYUAN_DIR, "hunyuan3d-2gp")
    _free_gpu_for_subprocess()
    venv_python = HUNYUAN_DIR / "venv" / "bin" / "python"
    cmd = [
        str(venv_python), str(Path(__file__).resolve().parent / "hunyuan_wrapper.py"),
        "--vendor-dir", str(HUNYUAN_DIR),
        "--output", str(out_glb.resolve()),
        "--profile", str(profile),
    ]
    for view, path in images.items():
        cmd += [f"--{view}", str(Path(path).resolve())]
    _run_subprocess(cmd, timeout, cancel_event=cancel_event, error_label="Hunyuan3D-2GP multi-view generation")


def generate_texture_for_mesh(mesh_path: Path, image_path: str, out_glb: Path, profile: int = 4, timeout: int = 900,
                               cancel_event: "threading.Event | None" = None) -> None:
    """Image-to-texture ONLY via Hunyuan3D-2GP's paint pipeline: bakes a new
    texture (and UV -- gen3d/texture_wrapper.py's mesh_uv_wrap rebuilds it
    from scratch via xatlas, any existing UV on mesh_path is discarded) onto
    an EXISTING mesh, skipping shape generation entirely. mesh_path can be
    any mesh trimesh can load (.glb/.gltf/.obj/...), not just one this
    pipeline generated itself."""
    _require_vendor(HUNYUAN_DIR, "hunyuan3d-2gp")
    _free_gpu_for_subprocess()
    venv_python = HUNYUAN_DIR / "venv" / "bin" / "python"
    cmd = [
        str(venv_python), str(Path(__file__).resolve().parent / "texture_wrapper.py"),
        "--vendor-dir", str(HUNYUAN_DIR),
        "--mesh", str(Path(mesh_path).resolve()),
        "--image", str(Path(image_path).resolve()),
        "--output", str(out_glb.resolve()),
        "--profile", str(profile),
    ]
    _run_subprocess(cmd, timeout, cancel_event=cancel_event, error_label="Hunyuan3D-2GP texture generation")


def retopologize(in_glb: Path, out_glb: Path, target_faces: int = 15000, voxel_size: float = 0.01, timeout: int = 300,
                  cancel_event: "threading.Event | None" = None) -> None:
    """Voxel Remesh + iterative Decimate down to target_faces (see 3dtodo.md --
    Decimate alone plateaus well above target on raw marching-cubes meshes)."""
    _run_blender_script("retopo.py", [str(Path(in_glb).resolve()), str(Path(out_glb).resolve()),
                                       str(target_faces), str(voxel_size)], timeout, cancel_event=cancel_event)


def rebake_texture(source_glb: Path, target_glb: Path, out_glb: Path, texture_size: int = 2048, timeout: int = 600,
                    cancel_event: "threading.Event | None" = None) -> None:
    """Transfers source_glb's albedo AND surface detail (as a tangent-space
    normal map + an AO map) onto target_glb's new UVs (Cycles bake, no AI) --
    retopologize() throws away the original UVs and geometric detail alike;
    this is what puts the detail back as a texture instead. Also sets flat
    roughness/metallic factors (no per-pixel source for those yet -- see
    rebake_texture.py's own docstring). Three full bake passes now (was one
    before normal+AO were added), hence the higher default timeout."""
    _run_blender_script("rebake_texture.py", [str(Path(source_glb).resolve()), str(Path(target_glb).resolve()),
                                               str(Path(out_glb).resolve()), str(texture_size)], timeout,
                        cancel_event=cancel_event)


def estimate_material(mesh_glb: Path, out_glb: Path, num_views: int = 6, texture_size: int = 2048,
                       timeout_render: int = 300, timeout_estimate: int = 900, timeout_project: int = 300,
                       cancel_event: "threading.Event | None" = None) -> None:
    """AI roughness/metallic estimation (SuperMat, vendor/supermat) on top of an
    already-textured mesh (rebake_texture()'s own output -- needs its baked
    albedo to render a believable lit view from, see
    gen3d/blender_scripts/render_multiview.py's docstring). Three steps, each
    its own subprocess:

    1. render_multiview.py (system Blender) -- 6 lit views of mesh_glb + each
       camera's transform, into a temp dir.
    2. material_wrapper.py (vendor/supermat's own venv) -- SuperMat's
       multi-view UNet estimates albedo/roughness/metallic PER VIEW (not in
       the mesh's UV space -- see that script's own docstring for why only
       its multi-view mode is used at all here).
    3. project_material_to_uv.py (system Blender) -- bakes those per-view
       roughness/metallic estimates onto mesh_glb's REAL UV (nearest-camera-
       by-face-normal, no occlusion raycast -- see 3dtodo.md's own notes on
       this known v1 limitation), merges the result into mesh_glb's EXISTING
       material (keeps albedo/normal/AO, adds metallicRoughnessTexture in
       place of the flat ROUGHNESS_DEFAULT/METALLIC_DEFAULT factors).

    Runs GPU-heavy (SD2.1 UNet, same class of contention as Hunyuan3D-2GP/
    UniRig -- _free_gpu_for_subprocess() applies here too)."""
    _require_vendor(SUPERMAT_DIR, "supermat")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gen3d_material_") as tmp:
        tmp = Path(tmp)
        render_dir = tmp / "renders"
        render_dir.mkdir()
        _run_blender_script("render_multiview.py",
                             [str(Path(mesh_glb).resolve()), str(render_dir), str(num_views)],
                             timeout_render, cancel_event=cancel_event)

        _free_gpu_for_subprocess()
        material_dir = tmp / "material"
        material_dir.mkdir()
        venv_python = SUPERMAT_DIR / "venv" / "bin" / "python"
        cmd = [
            str(venv_python), str(Path(__file__).resolve().parent / "material_wrapper.py"),
            "--vendor-dir", str(SUPERMAT_DIR),
            "--input", str(render_dir),
            "--output-dir", str(material_dir),
            "--num-views", str(num_views),
            "--base-model", "sd2-community/stable-diffusion-2-1",
        ]
        _run_subprocess(cmd, timeout_estimate, cancel_event=cancel_event, error_label="SuperMat material estimation")

        _run_blender_script("project_material_to_uv.py",
                             [str(Path(mesh_glb).resolve()), str(render_dir / "cameras.json"),
                              str(material_dir), str(Path(out_glb).resolve()), str(texture_size)],
                             timeout_project, cancel_event=cancel_event)


def _strip_glb_extras(in_glb: Path, out_glb: Path, timeout: int = 60) -> None:
    """Pure-Python glTF JSON cleanup, no Blender needed -- see
    gen3d/blender_scripts/strip_glb_extras.py for why this is necessary.
    No cancel_event -- near-instant (JSON/binary surgery, no model/GPU work),
    not a meaningful cancellation target."""
    _run_subprocess(
        ["python3", str(BLENDER_SCRIPTS / "strip_glb_extras.py"), str(in_glb), str(out_glb)],
        timeout, error_label="strip_glb_extras",
    )


def _unirig_run(script: str, extra_args: list[str], timeout: int,
                 cancel_event: "threading.Event | None" = None) -> None:
    _require_vendor(UNIRIG_DIR, "unirig")
    _free_gpu_for_subprocess()
    import os
    venv_bin = UNIRIG_DIR / "venv" / "bin"
    # The bash scripts under launch/inference/ call bare `python` themselves --
    # they need UniRig's own venv first on PATH, not a wholesale env replacement
    # (that would drop HOME/etc. and break the HF cache lookup, CUDA libs, ...).
    env = {**os.environ, "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}"}
    cmd = ["bash", str(UNIRIG_DIR / "launch" / "inference" / script), *extra_args]
    _run_subprocess(cmd, timeout, cancel_event=cancel_event, cwd=UNIRIG_DIR, env=env,
                     error_label=f"UniRig {script}")


def generate_skeleton(mesh_glb: Path, out_fbx: Path, timeout: int = 600,
                       cancel_event: "threading.Event | None" = None) -> None:
    """UniRig skeleton prediction -- ~4 GB VRAM peak measured on a 6 GB card,
    see 3dtodo.md (much lighter than the README's stated 8 GB minimum: the
    mesh point-cloud encoder runs on CPU by default in UniRig's own config).

    mesh_glb/out_fbx are resolved to absolute here -- _unirig_run runs its
    subprocess with cwd=UNIRIG_DIR (the launch/inference/*.sh scripts assume
    it), so a relative path would resolve against the WRONG directory."""
    _unirig_run("generate_skeleton.sh", ["--input", str(Path(mesh_glb).resolve()),
                                          "--output", str(Path(out_fbx).resolve())], timeout,
                cancel_event=cancel_event)


def skin_with_unirig(skeleton_fbx: Path, mesh_glb: Path, out_glb: Path, timeout: int = 900,
                      cancel_event: "threading.Event | None" = None) -> None:
    """Full UniRig skin-prediction model -- more accurate but peaked at
    ~5.87 GB on a 6 GB card (only ~274 MB headroom, see 3dtodo.md). Prefer
    skin_with_auto_weights() unless accuracy on complex topology matters more
    than that margin."""
    skeleton_fbx = Path(skeleton_fbx).resolve()
    mesh_glb = Path(mesh_glb).resolve()
    out_glb = Path(out_glb).resolve()
    skin_fbx = skeleton_fbx.with_name("skin.fbx")
    _unirig_run("generate_skin.sh", ["--input", str(skeleton_fbx), "--output", str(skin_fbx)], timeout,
                cancel_event=cancel_event)
    merged = skin_fbx.with_name("merged.glb")
    _unirig_run("merge.sh", ["--source", str(skin_fbx), "--target", str(mesh_glb), "--output", str(merged)], 120,
                cancel_event=cancel_event)
    _strip_glb_extras(merged, out_glb)


def skin_with_auto_weights(skeleton_fbx: Path, out_glb: Path, timeout: int = 120,
                            cancel_event: "threading.Event | None" = None) -> None:
    """Blender's built-in heat-diffusion Automatic Weights instead of UniRig's
    skin model -- no extra VRAM/checkpoint, CPU-only, default skin source."""
    skeleton_fbx = Path(skeleton_fbx).resolve()
    out_glb = Path(out_glb).resolve()
    raw = skeleton_fbx.with_name("auto_weight_raw.glb")
    _run_blender_script("auto_weight.py", [str(skeleton_fbx), str(raw)], timeout, cancel_event=cancel_event)
    _strip_glb_extras(raw, out_glb)


def add_rig(mesh_glb: Path, out_glb: Path, skin_source: str = "auto_weights",
            on_stage: Callable[[str], None] | None = None,
            cancel_event: "threading.Event | None" = None) -> None:
    """skin_source: "auto_weights" (default) or "unirig"."""
    mesh_glb = Path(mesh_glb).resolve()
    out_glb = Path(out_glb).resolve()
    skeleton_fbx = mesh_glb.with_name("skeleton.fbx")
    if on_stage:
        on_stage("rig_skeleton")
    generate_skeleton(mesh_glb, skeleton_fbx, cancel_event=cancel_event)
    if skin_source == "unirig":
        if on_stage:
            on_stage("rig_skin")
        skin_with_unirig(skeleton_fbx, mesh_glb, out_glb, cancel_event=cancel_event)
    else:
        if on_stage:
            on_stage("rig_skin")
        skin_with_auto_weights(skeleton_fbx, out_glb, cancel_event=cancel_event)


def convert(in_path: Path, out_path: Path, timeout: int = 60,
            cancel_event: "threading.Event | None" = None) -> None:
    """.glb/.gltf/.fbx conversion in any direction -- e.g. Animato only
    accepts .fbx/.gltf/.obj, not .glb."""
    _run_blender_script("convert.py", [str(Path(in_path).resolve()), str(Path(out_path).resolve())], timeout,
                        cancel_event=cancel_event)


def run_gen_model(image_path: str | None = None, out_slug: str = "", rig: bool = False, raw: bool = False,
                   target_faces: int = 15000, profile: int = 4,
                   skin_source: str = "auto_weights", lod: int = 0, pbr_ai: bool = False,
                   images: dict[str, str] | None = None,
                   on_stage: Callable[[str], None] | None = None,
                   cancel_event: "threading.Event | None" = None) -> Path:
    """Full pipeline: generate -> (retopologize -> rebake texture, unless
    raw=True) -> (rig, if rig=True) -> (lod additional LOD variants, if
    lod>0). Exactly one of `image_path` (single-view) or `images` (multi-view,
    see generate_mesh_and_texture_mv()'s own docstring) must be given --
    `image_path` must already be a real image file, `images` a dict of
    {"front"/"left"/"back"/"right": path}; text-prompt-to-image is the
    CALLER's job (tools/gen_model.py reuses tools.image_gen.generate_image for
    that, no need to duplicate it here). Returns the final (LOD0/full-quality)
    .glb path under generated/models/.

    raw=True skips retopology/texture-rebake entirely -- the untouched
    Hunyuan3D-2GP output (hundreds of thousands of faces, see 3dtodo.md on
    why that's too dense for a game asset as-is). Useful to inspect the raw
    generation quality on its own, or to feed a different retopology tool
    downstream. Combinable with rig=True, but untested at that face count --
    UniRig's own extract step resamples internally (faces_target_count
    defaults to 50000), so it may just be slow rather than broken.

    lod: how many ADDITIONAL, progressively lower-poly variants to generate
    alongside the main model, each its own separate .glb -- not multiple LODs
    embedded in one file, glTF's MSFT_lod extension for that isn't supported
    by Blender's exporter or by trimesh, and separate files are what game
    engines' own LOD-group workflows expect anyway (e.g. Unity import: drop
    each in as its own mesh, wire into a LODGroup). Named
    `{out_slug}_lod1.glb` .. `{out_slug}_lod{lod}.glb`; `{out_slug}.glb`
    itself is always the full-quality model (LOD0), regardless of lod.
    Every LOD is retopologized+rebaked straight from the RAW Hunyuan3D-2GP
    output, not chained off the previous LOD (or off the main model when
    raw=True skipped its own retopo) -- so detail loss doesn't compound
    across levels. Each level's target_faces halves the previous one's
    (floored at 500 faces). Rigging is NOT repeated for LODs: a shared
    skeleton driving meshes of different topology needs its own skin-
    transfer step, which this pipeline doesn't have -- out of scope here,
    LOD meshes come out unrigged even when rig=True. pbr_ai is likewise NOT
    repeated for LODs -- it alone adds ~6-8 minutes, doing that per LOD level
    would defeat their own point (fast, lower-detail tiers).

    pbr_ai=True adds AI-estimated roughness/metallic (SuperMat, see
    estimate_material()'s own docstring) to the main model only, on top of
    the retextured mesh -- has no effect when raw=True (no retextured mesh to
    estimate from). Adds ~6-8 minutes (own GPU-heavy subprocess chain, same
    VRAM contention class as Hunyuan3D-2GP/UniRig).

    on_stage, if given, is called with a short stage key ("mesh", "retopo",
    "rebake", "pbr_ai", "rig_skeleton", "rig_skin", "lod1".."lod{lod}") right
    before each blocking step starts -- this whole function runs inside the
    CALLER's executor thread (see tools/gen_model.py), so callers use it to
    keep a TUI spinner/status line honest about which multi-minute step is
    actually running, instead of going silent for the ~7+ minutes between
    "started" and "done". The MCP subprocess path (gen_model_server.py) just
    doesn't pass one. "pbr_ai" covers estimate_material()'s entire multi-
    subprocess chain (render+estimate+project), not split into sub-stages --
    same granularity as "rebake" covering rebake_texture() as a whole.

    cancel_event, if given, is checked between/during every subprocess call
    (see _run_subprocess) -- setting it kills whichever child process is
    currently running instead of leaving it to finish in the background, and
    raises PipelineCancelled up through here, skipping every later stage (no
    per-stage cancellation checks needed beyond that -- an exception already
    unwinds the rest of this function). See tools/gen_model.py for how Ctrl+C
    wires up to this.
    """
    if bool(image_path) == bool(images):
        raise ValueError("run_gen_model needs exactly one of image_path or images")
    GENERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gen3d_") as tmp:
        tmp = Path(tmp)
        raw_glb = tmp / "raw.glb"
        if on_stage:
            on_stage("mesh")
        if images:
            generate_mesh_and_texture_mv(images, raw_glb, profile=profile, cancel_event=cancel_event)
        else:
            generate_mesh_and_texture(image_path, raw_glb, profile=profile, cancel_event=cancel_event)

        if raw:
            final = raw_glb
        else:
            retopo = tmp / "retopo.glb"
            if on_stage:
                on_stage("retopo")
            retopologize(raw_glb, retopo, target_faces=target_faces, cancel_event=cancel_event)

            textured = tmp / "textured.glb"
            if on_stage:
                on_stage("rebake")
            rebake_texture(raw_glb, retopo, textured, cancel_event=cancel_event)
            final = textured

            if pbr_ai:
                pbr_glb = tmp / "pbr.glb"
                if on_stage:
                    on_stage("pbr_ai")
                estimate_material(final, pbr_glb, cancel_event=cancel_event)
                final = pbr_glb

        if rig:
            rigged = tmp / "rigged.glb"
            add_rig(final, rigged, skin_source=skin_source, on_stage=on_stage, cancel_event=cancel_event)
            final = rigged

        out_path = GENERATED_MODELS_DIR / f"{out_slug}.glb"
        shutil.copy(final, out_path)

        for i in range(1, lod + 1):
            if on_stage:
                on_stage(f"lod{i}")
            lod_faces = max(500, target_faces // (2 ** i))
            lod_retopo = tmp / f"lod{i}_retopo.glb"
            retopologize(raw_glb, lod_retopo, target_faces=lod_faces, cancel_event=cancel_event)
            lod_textured = tmp / f"lod{i}_textured.glb"
            rebake_texture(raw_glb, lod_retopo, lod_textured, cancel_event=cancel_event)
            shutil.copy(lod_textured, GENERATED_MODELS_DIR / f"{out_slug}_lod{i}.glb")

        return out_path


def lod_paths_for(out_path: Path, lod: int) -> list[Path]:
    """Deterministic sibling-file names run_gen_model's lod= loop writes --
    lets callers report them without duplicating the naming convention."""
    return [out_path.with_name(f"{out_path.stem}_lod{i}{out_path.suffix}") for i in range(1, lod + 1)]


def run_gen_texture(mesh_path: str, image_path: str, out_slug: str, profile: int = 4,
                     on_stage: Callable[[str], None] | None = None,
                     cancel_event: "threading.Event | None" = None) -> Path:
    """Texture-only pipeline for /gen_texture: paint a new texture onto an
    existing mesh from a reference image, skipping shape generation
    (run_gen_model's "mesh" stage) entirely. Returns the final .glb path
    under generated/models/."""
    GENERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if on_stage:
        on_stage("texture")
    out_path = GENERATED_MODELS_DIR / f"{out_slug}.glb"
    generate_texture_for_mesh(Path(mesh_path), image_path, out_path, profile=profile, cancel_event=cancel_event)
    return out_path
