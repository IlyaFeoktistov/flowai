"""
Кастомный MCP-сервер: генерация изображений (Stable Diffusion/FLUX через
diffusers), портировано из tools/image_gen.py.

Готового MCP-сервера под наш локальный GPU-pipeline (конкретные модели,
кэш, FLUX/Lightning-варианты) в реестре нет — свой.

Не портирована UI-специфичная часть tools/image_gen.py (прогресс-бар
скачивания через _TqdmCapture/app.set_stats) — она была привязана к
терминальному ui.app этого процесса, а MCP-сервер живёт в отдельном
подпроцессе без доступа к нему. Первое скачивание модели (разово, до
нескольких GB) идёт без визуального прогресса — не блокирующая проблема,
но при желании можно добавить через FastMCP Context.report_progress().

Запуск: python3 -m mcp_agent.servers.image_gen_server
"""
import asyncio
import os
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import settings  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("image_gen")

OUTPUT_DIR = Path(_PROJECT_ROOT) / "generated"

_pipe = None
_pipe_key: tuple = (None, None, None, None)

_LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
_LIGHTNING_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
_LIGHTNING_CKPTS = {4: "sdxl_lightning_4step_unet.safetensors", 8: "sdxl_lightning_8step_unet.safetensors"}


def _is_flux(model: str) -> bool:
    return "flux" in model.lower()


def _is_cached(repo: str, filename: str = "model_index.json") -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(repo, filename) is not None
    except Exception:
        return False


def _build_lightning(device: str, dtype, steps: int):
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt = _LIGHTNING_CKPTS.get(steps, _LIGHTNING_CKPTS[4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unet = UNet2DConditionModel.from_config(_LIGHTNING_BASE, subfolder="unet").to(device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(_LIGHTNING_REPO, ckpt)))
        kwargs = {"unet": unet, "torch_dtype": dtype}
        if dtype.itemsize == 2:
            kwargs["variant"] = "fp16"
        pipe = StableDiffusionXLPipeline.from_pretrained(_LIGHTNING_BASE, **kwargs).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    return pipe, None


def _build_flux(model: str):
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
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        return None, "FATAL: dependencies not installed (pip install diffusers torch transformers accelerate safetensors)"

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
            pipe = AutoPipelineForText2Image.from_pretrained(model, **kwargs).to(device)
        return pipe, None
    except Exception as exc:
        try:
            from huggingface_hub.errors import GatedRepoError
            if isinstance(exc, GatedRepoError):
                return None, (
                    f"Model {model} is gated — a HuggingFace token is required. "
                    f"Accept the license at https://huggingface.co/{model}, "
                    "get a token at https://huggingface.co/settings/tokens, "
                    "then export HF_TOKEN=..., or switch to sdxl-turbo (no token needed)."
                )
        except ImportError:
            pass
        raise


@mcp.tool()
async def unload_image_gen_model() -> str:
    """Frees the resident image-generation pipeline (SDXL/FLUX) from this
    subprocess's memory/VRAM — NOT a tool for normal task use; it exists so
    the main flowAI process can reclaim resources when this pipeline sits
    idle (see model_lifecycle.py, called from the /settings 'unload models'
    button and automatically when voice_mode turns off). Rebuilds
    lazily on the next generate_image/edit_image call, same one-time cost
    as this subprocess's very first call."""
    global _pipe, _pipe_key
    if _pipe is None:
        return "Nothing to unload — no image pipeline was ever built in this session."
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
    return "Image-generation pipeline unloaded."


@mcp.tool()
async def generate_image(prompt: str, negative_prompt: str = "") -> str:
    """Generate an image from a text description locally via Stable
    Diffusion/FLUX. `prompt` MUST be in English only."""
    global _pipe, _pipe_key

    prompt = prompt.strip()
    if not prompt:
        return "Error: no prompt specified"

    model = settings.get("image_gen_model")
    device = settings.get("image_gen_device")
    safety = settings.get("imggen_safety")
    steps = settings.get("imggen_steps")
    guidance = settings.get("imggen_guidance")
    width = settings.get("imggen_width")
    height = settings.get("imggen_height")
    prompt_prefix = settings.get("imggen_prompt_prefix") or ""
    negative_prompt = negative_prompt or settings.get("imggen_negative_prompt") or ""
    key = (model, device, safety, steps)

    loop = asyncio.get_event_loop()

    if _pipe is None or _pipe_key != key:
        _pipe, err = await loop.run_in_executor(None, lambda: _build_pipeline(model, device, safety, steps))
        if err:
            return f"Error: {err}"
        _pipe_key = key

    OUTPUT_DIR.mkdir(exist_ok=True)
    final_prompt = f"{prompt_prefix}, {prompt}" if prompt_prefix else prompt
    use_negative = negative_prompt and not _is_flux(model)

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kwargs = dict(prompt=final_prompt, num_inference_steps=steps,
                          guidance_scale=guidance, width=width, height=height)
            if use_negative:
                kwargs["negative_prompt"] = negative_prompt
            result = _pipe(**kwargs)
        img = result.images[0]
        out = OUTPUT_DIR / f"{hash(final_prompt) & 0xFFFFFF:06x}.png"
        img.save(out)
        return str(out)

    path = await loop.run_in_executor(None, _run)
    return f"Image saved: {path}"


@mcp.tool()
async def edit_image(path: str, prompt: str, negative_prompt: str = "") -> str:
    """Edit/modify an EXISTING image via image-to-image diffusion — keeps the
    input's overall composition/layout and alters it per `prompt` (restyle,
    change colors/objects/setting). `path` must be a real, already-existing
    image file (e.g. one the user pasted, or a previous generate_image/
    edit_image result) — NOT for creating a picture from scratch, that's
    generate_image. `prompt` MUST be in English only. How much the input
    changes is controlled by the user's own image-edit strength setting (/settings),
    not by this call — there's no way to override it per-request."""
    global _pipe, _pipe_key
    from PIL import Image
    from diffusers import AutoPipelineForImage2Image

    prompt = prompt.strip()
    if not prompt:
        return "Error: no prompt specified"

    src = Path(path)
    if not src.is_file():
        return f"Error: file not found: {path}"

    model = settings.get("image_gen_model")
    device = settings.get("image_gen_device")
    safety = settings.get("imggen_safety")
    steps = settings.get("imggen_steps")
    guidance = settings.get("imggen_guidance")
    strength = settings.get("imggen_strength")
    width = settings.get("imggen_width")
    height = settings.get("imggen_height")
    prompt_prefix = settings.get("imggen_prompt_prefix") or ""
    negative_prompt = negative_prompt or settings.get("imggen_negative_prompt") or ""
    key = (model, device, safety, steps)

    loop = asyncio.get_event_loop()

    if _pipe is None or _pipe_key != key:
        _pipe, err = await loop.run_in_executor(None, lambda: _build_pipeline(model, device, safety, steps))
        if err:
            return f"Error: {err}"
        _pipe_key = key

    OUTPUT_DIR.mkdir(exist_ok=True)
    final_prompt = f"{prompt_prefix}, {prompt}" if prompt_prefix else prompt
    use_negative = negative_prompt and not _is_flux(model)

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # from_pipe() converts the already-loaded text2img pipeline (same
            # weights, no redownload/reload) into its img2img counterpart —
            # works generically across the SDXL/Lightning/Flux branches built
            # by _build_pipeline above, diffusers picks the matching Img2Img
            # class for whatever pipeline class _pipe already is.
            img2img = AutoPipelineForImage2Image.from_pipe(_pipe)
            init_image = Image.open(src).convert("RGB").resize((width, height))
            kwargs = dict(prompt=final_prompt, image=init_image, strength=strength,
                          num_inference_steps=steps, guidance_scale=guidance)
            if use_negative:
                kwargs["negative_prompt"] = negative_prompt
            result = img2img(**kwargs)
        img = result.images[0]
        out = OUTPUT_DIR / f"{hash(final_prompt + path) & 0xFFFFFF:06x}.png"
        img.save(out)
        return str(out)

    out_path = await loop.run_in_executor(None, _run)
    return f"Image saved: {out_path}"


if __name__ == "__main__":
    mcp.run()
