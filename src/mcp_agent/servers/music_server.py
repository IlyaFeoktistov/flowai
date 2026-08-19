"""
Кастомный MCP-сервер: генерация музыки/звуков локально через MusicGen
(facebook/musicgen-small, HF transformers — не audiocraft, у него более
тяжёлая/капризная цепочка зависимостей, а transformers уже стоит в основном
.venv ради diffusers/SDXL).

Устройство — settings.music_gen_device, дефолт CPU (тот же принцип, что уже
применён к STT/TTS: не соревноваться за те же 5.9 GB VRAM, что и Ollama/SDXL,
без явного запроса пользователя). GPU — опция по запросу, не тихий дефолт:
и на CPU, и на GPU генерация музыки объективно медленнее речи (больше токенов
на секунду аудио, выше sample rate), просто на GPU разница ощутимо меньше.

Запуск: python3 -m mcp_agent.servers.music_server
"""
import os
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("music")


def _output_dir() -> Path:
    """Path.cwd() — this server is always spawned with cwd=repo_path (see
    mcp_agent/config.py's build_mcp_connections), the project the user has
    open right now, not flowAI's own install directory: generated music
    belongs next to the project it was generated for, same convention as
    every other "generated/" dir in this app (gen3d.pipeline.
    generated_models_dir(), image_gen_server.py's OUTPUT_DIR)."""
    return Path.cwd() / "generated"

_MODEL_NAME = "facebook/musicgen-small"
_model = None
_processor = None
_model_device: str | None = None  # устройство, на котором реально лежит _model

# MusicGen генерирует ~50 токенов на секунду аудио на этой архитектуре —
# см. HF-доку модели. Ограничиваем сверху, чтобы случайный "duration=600" не
# посадил процесс на генерацию часами на CPU.
_MAX_DURATION_SECONDS = 30.0


def _load_model():
    """Перезагружает модель на нужное устройство, если settings.music_gen_device
    поменялся с прошлого вызова (тот же принцип переключения на лету, что уже
    есть у chat_model в agent_builder.py) — иначе смена настройки без
    перезапуска процесса молча игнорировалась бы."""
    global _model, _processor, _model_device
    import settings
    device = settings.get("music_gen_device")
    if _model is None or _model_device != device:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        # TRANSFORMERS_VERBOSITY/HF_HUB_DISABLE_PROGRESS_BARS (env, наверху
        # файла) не покрывают всё: обе библиотеки переустанавливают свой
        # уровень логирования лениво при первом обращении, через СОБСТВЕННУЮ
        # обёртку (transformers.utils.logging / huggingface_hub.utils.logging)
        # — внешний logging.getLogger(...).setLevel(...) эта обёртка тихо
        # перетирает. Настоящий способ приглушить и "Loading weights: N%"
        # (tqdm), и config-warning'и, и HTTP-warning'и с сервера HF Hub — их
        # же собственные функции, вызванные ПОСЛЕ импорта (лениво, здесь, а
        # не на старте всего процесса — transformers тяжёлый импорт, платить
        # за него при каждом запуске CLI ради тула, который не факт что
        # вызовут, не стоит).
        from transformers.utils import logging as _tf_logging
        _tf_logging.set_verbosity_error()
        _tf_logging.disable_progress_bar()
        from huggingface_hub.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
        _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
        _model = MusicgenForConditionalGeneration.from_pretrained(_MODEL_NAME).to(device)
        _model_device = device
    return _model, _processor


def unload_model() -> bool:
    """Frees the resident MusicGen model held by _load_model() above. This
    module is imported TWO different ways with TWO independent copies of
    these globals: directly, in the main flowAI process (cli.py's
    /music_gen, ui/music_stream.py's /music streaming), and as its own MCP
    subprocess (spawned for the agent's generate_music tool, see
    unload_music_gen_model below) — calling this only clears whichever
    copy the CALLING process actually holds. Rebuilds lazily on the next
    generate_clip_sync call. Returns True if something was actually
    unloaded. See model_lifecycle.py, the caller for the main-process copy
    (the /settings 'unload models' button and the automatic
    voice_mode-off sweep both go through it, not this function directly)."""
    global _model, _processor, _model_device
    if _model is None:
        return False
    device = _model_device
    _model = None
    _processor = None
    _model_device = None
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()
    return True


@mcp.tool()
async def unload_music_gen_model() -> str:
    """Frees the resident MusicGen model in THIS subprocess (the one
    backing the agent's own generate_music tool calls) — NOT a tool for
    normal task use; it exists so the main flowAI process can reclaim this
    subprocess's memory when it's been idle (see model_lifecycle.py,
    called from the /settings 'unload models' button and automatically
    when voice_mode turns off). Rebuilds lazily on the next generate_music
    call, same one-time cost as this subprocess's very first call."""
    if unload_model():
        return "MusicGen model unloaded."
    return "Nothing to unload — no MusicGen model was ever built in this session."


def generate_clip_sync(prompt: str, duration: float, continuation_audio=None, continuation_sr: int | None = None):
    """Синхронное ядро генерации, общее для generate_music (тул) и
    потокового /music (cli.py, см. ui/music_stream.py) — единственное
    отличие потокового режима: continuation_audio/continuation_sr (хвост
    ПРЕДЫДУЩЕГО куска, чтобы следующий кусок продолжал тот же стиль/тональность,
    а не начинал каждый раз с нуля). Модель принимает conditioning-аудио как
    НАЧАЛО последовательности и генерирует max_new_tokens ДАЛЬШЕ него — значит
    в выходе первые continuation_audio-секунды это тот же хвост, что мы дали
    на входе, а не новый материал; вызывающий сам решает, срезать их или нет
    (потоковый плеер срезает, generate_music — нет, там continuation не бывает).

    Возвращает (audio_np, sample_rate)."""
    model, processor = _load_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(text=[prompt], padding=True, return_tensors="pt")
        if continuation_audio is not None:
            kwargs["audio"] = continuation_audio
            kwargs["sampling_rate"] = continuation_sr
        inputs = processor(**kwargs).to(_model_device)
        max_new_tokens = int(duration * 50)
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sample_rate = model.config.audio_encoder.sampling_rate
    return audio_values[0, 0].cpu().numpy(), sample_rate


@mcp.tool()
async def generate_music(prompt: str, duration: float = 10.0) -> str:
    """Generate instrumental music/sound locally from a text description
    (e.g. "upbeat acoustic guitar loop", "calm ambient rain sounds"). No
    vocals/lyrics support. `prompt` MUST be in English only. `duration` in
    seconds (default 10, max 30) — generation is CPU-only and noticeably
    slower than speech synthesis or image generation, expect it to take
    longer than the requested duration."""
    import asyncio

    prompt = prompt.strip()
    if not prompt:
        return "Error: no prompt specified"
    duration = max(1.0, min(duration, _MAX_DURATION_SECONDS))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)
    output_dir = _output_dir()
    output_dir.mkdir(exist_ok=True)

    def _run():
        audio_np, sample_rate = generate_clip_sync(prompt, duration)
        out = output_dir / f"{hash(prompt) & 0xFFFFFF:06x}.wav"
        # scipy, not torchaudio — torchaudio isn't in this venv and must match
        # torch's exact version to install safely; scipy is already a
        # transitive dependency here and needs no version coordination.
        from scipy.io import wavfile
        wavfile.write(str(out), sample_rate, audio_np)
        return str(out)

    path = await loop.run_in_executor(None, _run)
    return f"Music saved: {path}"


if __name__ == "__main__":
    mcp.run()
