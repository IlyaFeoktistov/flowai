import asyncio
import os
import sys
import threading
from pathlib import Path
import settings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # убираем только на время скачки

from tools.base import ok, fail
from ui.console import console as _console, safe_write, get_app

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_dl_state: dict = {"bytes_done": 0, "bytes_total": 0, "rate": None, "files_done": 0, "files_total": 0}
_dl_lock = threading.Lock()
_sink = open(os.devnull, "w")


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class _TqdmCapture:
    """Перехватывает tqdm от HF Hub: пишет в /dev/null, прогресс кладёт в _dl_state."""
    _saved: dict = {}

    @classmethod
    def install(cls):
        import tqdm as _tm
        import tqdm.auto as _ta
        cls._saved = {"tqdm": _tm.tqdm, "auto": _ta.tqdm}
        real = _tm.tqdm

        class Capture(real):
            def __init__(self, *args, **kw):
                kw["file"] = _sink
                kw["dynamic_ncols"] = False
                super().__init__(*args, **kw)

            def display(self, msg=None, pos=None, *args, **kwargs):
                unit = getattr(self, "unit", "") or ""
                n = getattr(self, "n", 0) or 0
                total = getattr(self, "total", None)
                with _dl_lock:
                    if unit == "it":
                        _dl_state["files_done"] = n
                        if total is not None:
                            _dl_state["files_total"] = total
                    else:
                        _dl_state["bytes_done"] = n
                        if total is not None:
                            _dl_state["bytes_total"] = total
                        try:
                            _dl_state["rate"] = self.format_dict.get("rate")
                        except Exception:
                            pass
                return True

            def clear(self, nolock=False):
                pass

        _tm.tqdm = Capture
        _ta.tqdm = Capture
        try:
            import huggingface_hub.file_download as _hf
            _hf.tqdm = Capture
        except Exception:
            pass

    @classmethod
    def uninstall(cls):
        if not cls._saved:
            return
        import tqdm as _tm
        import tqdm.auto as _ta
        _tm.tqdm = cls._saved["tqdm"]
        _ta.tqdm = cls._saved["auto"]
        try:
            import huggingface_hub.file_download as _hf
            _hf.tqdm = cls._saved["tqdm"]
        except Exception:
            pass
        cls._saved.clear()


async def _spin_download(label: str) -> None:
    i = 0
    app = get_app()
    try:
        while True:
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            with _dl_lock:
                s = dict(_dl_state)

            stat = f"\033[2m{frame} {label}"
            total = s.get("bytes_total") or 0
            done  = s.get("bytes_done") or 0
            rate  = s.get("rate")
            files_done  = s.get("files_done") or 0
            files_total = s.get("files_total") or 0

            if total > 0:
                pct = done / total * 100
                stat += f"  {pct:.1f}%  {_fmt_size(done)}/{_fmt_size(total)}"
            if rate:
                stat += f"  ↓ {_fmt_size(rate)}/с"
            if files_total > 0:
                stat += f"  {files_done}/{files_total} файлов"
            stat += "\033[0m"

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

def _output_dir() -> Path:
    """Path.cwd() — the project the user has open right now (repo_path),
    not flowAI's own install directory: generated images belong next to
    the project they were generated for, same convention as every other
    "generated/" dir in this app (gen3d.pipeline.generated_models_dir(),
    music_server.py's OUTPUT_DIR)."""
    return Path.cwd() / "generated"

_pipe = None
_pipe_key: tuple = (None, None, None, None)  # (model, device, safety, steps)


def unload_pipe() -> bool:
    """Frees the resident SDXL/FLUX pipeline built by generate_image() above
    (the /gen command's own pipe — separate from the identically-shaped
    _pipe in mcp_agent/servers/image_gen_server.py, which lives in its own
    MCP subprocess and isn't reachable from here). Rebuilds lazily on the
    next /gen or generate_image call, same cost as the very first one this
    session. Returns True if something was actually unloaded. See
    model_lifecycle.unload_idle_models, the caller for both the manual
    /settings button and the automatic voice_mode-off sweep."""
    global _pipe, _pipe_key
    if _pipe is None:
        return False
    device = _pipe_key[1]
    _pipe = None
    _pipe_key = (None, None, None, None)
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()
    return True

_LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
_LIGHTNING_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
_LIGHTNING_CKPTS = {4: "sdxl_lightning_4step_unet.safetensors", 8: "sdxl_lightning_8step_unet.safetensors"}

_MODEL_SIZES = {
    "black-forest-labs/FLUX.1-schnell":           "~33 GB (однократно)",
    "black-forest-labs/FLUX.1-dev":               "~33 GB (однократно)",
    _LIGHTNING_REPO:                              "SDXL-base ~6.5 GB + Lightning UNet ~0.8 GB",
    "stabilityai/sdxl-turbo":                     "~6.5 GB",
    "stabilityai/stable-diffusion-xl-base-1.0":   "~6.5 GB",
    "stabilityai/stable-diffusion-2-1":           "~5.2 GB",
    "runwayml/stable-diffusion-v1-5":             "~4.0 GB",
}

def _is_flux(model: str) -> bool:
    return "FLUX" in model or "flux" in model


def _is_cached(repo: str, filename: str = "model_index.json") -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(repo, filename) is not None
    except Exception:
        return False


def _needs_download(model: str, steps: int) -> bool:
    if model == _LIGHTNING_REPO:
        ckpt = _LIGHTNING_CKPTS.get(steps, _LIGHTNING_CKPTS[4])
        return not _is_cached(_LIGHTNING_BASE) or not _is_cached(_LIGHTNING_REPO, ckpt)
    return not _is_cached(model)


def _build_lightning(device: str, dtype, steps: int):
    import warnings
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt = _LIGHTNING_CKPTS.get(steps, _LIGHTNING_CKPTS[4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unet = UNet2DConditionModel.from_config(_LIGHTNING_BASE, subfolder="unet").to(device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(_LIGHTNING_REPO, ckpt)))
        kwargs = {"unet": unet, "torch_dtype": dtype}
        if dtype.itemsize == 2:  # float16
            kwargs["variant"] = "fp16"
        pipe = StableDiffusionXLPipeline.from_pretrained(_LIGHTNING_BASE, **kwargs).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    return pipe, None


def _build_flux(model: str):
    import warnings
    import torch
    from diffusers import FluxPipeline

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, token=token)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe, None


def _build_pipeline(model: str, device: str, safety: bool, steps: int = 4):
    try:
        import warnings
        import torch
        # TRANSFORMERS_VERBOSITY (env, top of file) isn't enough on its own:
        # diffusers' auto_pipeline.py eagerly imports EVERY pipeline class it
        # supports (including one — z_image — that references a transformers
        # class deprecated in transformers 5.x) right at the `from diffusers
        # import AutoPipelineForText2Image` line below, before any
        # from_pretrained call ever runs — and transformers' own logging
        # module only applies the env var the FIRST time its root logger gets
        # configured, silently keeping whatever was set if something else
        # configured it first. Call its own function here, before that import
        # can trigger the warning (same fix already applied in
        # mcp_agent/servers/music_server.py:_load_model for the same class of
        # noise from MusicGen's own dependencies).
        from transformers.utils import logging as _tf_logging
        _tf_logging.set_verbosity_error()
        from huggingface_hub.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        return None, "FATAL: dependencies not installed. Tell user: pip install diffusers torch transformers accelerate safetensors"

    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        if model == _LIGHTNING_REPO:
            return _build_lightning(device, dtype, steps)

        if _is_flux(model):
            return _build_flux(model)

        kwargs = {"torch_dtype": dtype}
        if not safety:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
            pipe = pipe.to(device)
        return pipe, None
    except Exception as exc:
        try:
            from huggingface_hub.errors import GatedRepoError
            if isinstance(exc, GatedRepoError):
                return None, (
                    f"Model {model} is gated — a HuggingFace token is required.\n"
                    f"  1. Accept the license: https://huggingface.co/{model}\n"
                    f"  2. Get a token: https://huggingface.co/settings/tokens\n"
                    f"  3. Run: export HF_TOKEN=hf_your_token\n"
                    f"  Or switch the model in /settings to sdxl-turbo (no token required)"
                )
        except ImportError:
            pass
        raise


_ENHANCE_SYSTEM = """\
You are a high-precision image generation assistant.

Always generate visually coherent, physically plausible, highly detailed images.

Focus on:
- correct anatomy and proportions
- realistic lighting and shadows
- high-frequency details (skin, fabric, materials)
- strong composition and depth
- cinematic framing and camera awareness
- natural textures without plastic or AI artifacts

Interpret prompts literally and avoid hallucinating extra objects unless clearly implied.

If the prompt is ambiguous, choose the most realistic and compositionally strong interpretation.

Prioritize photographic realism unless a style is explicitly requested.

Output should look like a professional photograph or high-end digital artwork.

Rules:
- DO NOT change the core meaning or subject
- Output ONLY the enhanced prompt, no explanations, no quotes
- Keep it under 150 words
- Write in English\
"""


def _enhance_prompt_sync(prompt: str) -> str:
    try:
        import ollama
        resp = ollama.chat(
            model=settings.get("chat_model"),
            messages=[
                {"role": "system", "content": _ENHANCE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            stream=False,
            options={"num_gpu": 99, "temperature": 0.7},
        )
        enhanced = (resp.message.content or "").strip()
        return enhanced if enhanced else prompt
    except Exception:
        return prompt


async def _enhance_prompt(prompt: str, loop) -> str:
    return await loop.run_in_executor(None, _enhance_prompt_sync, prompt)


async def generate_image(args: dict) -> dict:
    global _pipe, _pipe_key

    prompt = args.get("prompt", "").strip()
    if not prompt:
        return fail("No prompt specified for generation")

    model = settings.get("image_gen_model")
    device = settings.get("image_gen_device")
    safety = settings.get("imggen_safety")
    steps = settings.get("imggen_steps")
    guidance = settings.get("imggen_guidance")
    width = settings.get("imggen_width")
    height = settings.get("imggen_height")
    prompt_prefix = settings.get("imggen_prompt_prefix") or ""
    negative_prompt = args.get("negative_prompt", "") or settings.get("imggen_negative_prompt") or ""
    enhance = settings.get("imggen_enhance_prompt")
    key = (model, device, safety, steps)

    tui = get_app() is not None
    loop = asyncio.get_event_loop()

    if enhance:
        prompt = await _enhance_prompt(prompt, loop)

    if _pipe is None or _pipe_key != key:
        downloading = _needs_download(model, steps)

        if tui:
            # TUI mode: suppress all output, just build silently.
            # The stream display already shows a spinner; we don't want raw writes
            # or tqdm bars corrupting the layout.
            _TqdmCapture.install()
            if downloading:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                with _dl_lock:
                    _dl_state.update({"bytes_done": 0, "bytes_total": 0,
                                      "rate": None, "files_done": 0, "files_total": 0})
            _pipe, err = await loop.run_in_executor(
                None, lambda: _build_pipeline(model, device, safety, steps)
            )
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            _TqdmCapture.uninstall()
        else:
            # CLI mode: show progress normally.
            size = _MODEL_SIZES.get(model, "")
            size_str = f"  [dim]{size}[/]" if size else ""
            if downloading:
                _console.print(f"\n[cyan]  📥 скачиваю[/] [yellow]{model}[/]{size_str}")
                _console.print("[dim]  Файлы скачиваются один раз, это займёт время[/]")
                with _dl_lock:
                    _dl_state.update({"bytes_done": 0, "bytes_total": 0,
                                      "rate": None, "files_done": 0, "files_total": 0})
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                _TqdmCapture.install()
                spin_label = "скачиваю"
            else:
                _console.print(f"\n[cyan]  ⚙ загружаю[/] [yellow]{model}[/]{size_str}")
                spin_label = "загружаю модель..."

            spinner = asyncio.create_task(_spin_download(spin_label))
            _pipe, err = await loop.run_in_executor(
                None, lambda: _build_pipeline(model, device, safety, steps)
            )
            spinner.cancel()
            try:
                await spinner
            except asyncio.CancelledError:
                pass

            if downloading:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                _TqdmCapture.uninstall()

            _console.print("[green]  ✓ модель готова[/]\n")

        if err:
            return fail(err)
        _pipe_key = key

    output_dir = _output_dir()
    output_dir.mkdir(exist_ok=True)

    final_prompt = f"{prompt_prefix}, {prompt}" if prompt_prefix else prompt
    use_negative = negative_prompt and not _is_flux(model)

    def _run():
        import warnings
        _TqdmCapture.install()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kwargs = dict(
                    prompt=final_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                )
                if use_negative:
                    kwargs["negative_prompt"] = negative_prompt
                result = _pipe(**kwargs)
        finally:
            _TqdmCapture.uninstall()
        img = result.images[0]
        out = output_dir / f"{hash(final_prompt) & 0xFFFFFF:06x}.png"
        img.save(out)
        return str(out)

    path = await loop.run_in_executor(None, _run)
    return ok(f"Image saved: {path}")
