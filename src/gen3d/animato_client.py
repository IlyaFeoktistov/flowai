"""Drives Animato (github.com/otdnnc/Animato) for AI-generated animation,
using flowAI's own local Ollama instead of Animato's built-in /api/chat --
that endpoint is hardcoded to Gemini's REST contract (see
app/routers/chat.py in the Animato repo), despite the README's mention of
"bring your own ... local Ollama model" -- that claim only covers the MANUAL
flow (copy /api/prompt's text into any chat UI, paste the code back into
/api/run). This module reproduces that manual flow with an HTTP call to
Ollama instead of a human copy-paste.

Animato runs as a persistent `uv run fastapi run main.py` subprocess (its own
pip bpy==5.1.2 / Python 3.13 venv, managed by uv -- see setup.py),
started lazily on first use and left running, the same lifecycle as the
image-gen pipeline cached in tools/image_gen.py's module-level `_pipe`.
"""
import re
import subprocess
import time
from pathlib import Path

import httpx
import ollama

from gen3d.pipeline import ANIMATO_DIR, PipelineError, convert, generated_models_dir, _strip_glb_extras

PORT = 8791
BASE_URL = f"http://127.0.0.1:{PORT}"

_server_process: subprocess.Popen | None = None

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _server_alive() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/api/files", timeout=2)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_server_running() -> None:
    global _server_process
    if _server_alive():
        return
    if not ANIMATO_DIR.exists():
        raise PipelineError("animato isn't set up yet. Run: python3 setup.py --only animato")

    if _server_process is None or _server_process.poll() is not None:
        _server_process = subprocess.Popen(
            ["uv", "run", "fastapi", "run", "main.py", "--port", str(PORT)],
            cwd=ANIMATO_DIR,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    for _ in range(60):
        if _server_alive():
            return
        time.sleep(1)
    raise PipelineError("Animato server didn't come up within 60s")


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return m.group(1) if m else text


def upload_model(fbx_path: Path) -> str:
    with open(fbx_path, "rb") as f:
        r = httpx.post(f"{BASE_URL}/api/upload", files={"file": (fbx_path.name, f)}, timeout=30)
    r.raise_for_status()
    return r.json()["filename"]


def build_prompt(filename: str, message: str) -> str:
    r = httpx.post(f"{BASE_URL}/api/prompt", json={"filename": filename, "message": message}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt"]


def ask_ollama(prompt: str, model: str) -> str:
    resp = ollama.generate(model=model, prompt=prompt, stream=False, think=False)
    return _extract_code(resp.response)


def run_script(code: str) -> dict:
    r = httpx.post(f"{BASE_URL}/api/run", content=code, headers={"Content-Type": "text/plain"}, timeout=120)
    r.raise_for_status()
    return r.json()


_BAD_KEY_RE = re.compile(r'key "([^"]+)" not found')


def _correction_prompt(original_prompt: str, bad_code: str, error: str) -> str:
    extra = ""
    m = _BAD_KEY_RE.search(error)
    if m:
        extra = (
            f"\nSpecifically: '{m.group(1)}' DOES NOT EXIST in this rig -- that "
            "looks like a generic/Mixamo bone name, not one from the ACTUAL "
            "bone list above. Go back to the '## ARMATURES' section above and "
            "copy an exact name from there character-for-character."
        )
    return (
        f"{original_prompt}\n\n---\n"
        f"Your previous script failed. It was:\n```python\n{bad_code}\n```\n"
        f"It failed with: {error}\n{extra}\n\n"
        "Fix the mistake -- re-read the bone names and API cheat-sheet above "
        "carefully, do not invent names that aren't listed. Rewrite the "
        "COMPLETE corrected script now (same output format: one fenced python "
        "block, nothing else)."
    )


def animate(model_glb: Path, motion: str, chat_model: str, out_slug: str, max_retries: int = 2) -> Path:
    """Converts model_glb to .fbx, drives Animato+Ollama to animate it per
    `motion`, retries once (feeding the error back to the model) on failure,
    and returns the final animated .glb under generated/models/."""
    ensure_server_running()

    import tempfile
    with tempfile.TemporaryDirectory(prefix="gen3d_animate_") as tmp:
        tmp = Path(tmp)
        fbx_path = tmp / "model.fbx"
        convert(model_glb, fbx_path)

        filename = upload_model(fbx_path)
        prompt = build_prompt(filename, motion)
        code = ask_ollama(prompt, chat_model)
        result = run_script(code)

        attempt = 0
        while not result.get("ok") and attempt < max_retries:
            attempt += 1
            error = result.get("stderr") or result.get("error") or "unknown error"
            code = ask_ollama(_correction_prompt(prompt, code, error), chat_model)
            result = run_script(code)

        if not result.get("ok"):
            raise PipelineError(
                f"Animato script failed after {max_retries + 1} attempt(s): "
                f"{result.get('stderr') or result.get('error')}"
            )

        animated_fbx = ANIMATO_DIR / "public" / "upload" / filename
        out_glb_raw = tmp / "animated_raw.glb"
        convert(animated_fbx, out_glb_raw)

        models_dir = generated_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{out_slug}.glb"
        _strip_glb_extras(out_glb_raw, out_path)
        return out_path
