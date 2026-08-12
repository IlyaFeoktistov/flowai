#!/usr/bin/env python3
"""One-shot setup for the whole flowAI application -- everything the README's
own "Установка" section walks through by hand, in one script:

  main .venv       -- requirements.txt + .env (required, everything else optional)
  image-gen        -- diffusers/torch/transformers/accelerate into .venv (generate_image/edit_image, needs GPU)
  whisper          -- faster-whisper into .venv (voice input, CPU)
  tts              -- venv-tts, Chatterbox + Russian stress marks, its own Python 3.11 venv (voice output, CPU)
  hunyuan3d-2gp    -- image-to-3D mesh+texture (torch 2.5.1+cu124), own venv
  unirig           -- skeleton prediction (torch 2.3.1+cu121, spconv, flash_attn), own venv
  animato          -- AI-driven animation (its own pip bpy==5.1.2, Python 3.13), own venv
  supermat         -- AI roughness/metallic estimation (SD2.1-based, Python 3.12), own venv
  expert-streaming -- experimental dynamic MoE expert-offload llama-server (see expert_streaming.py), own venv-build-tools + CUDA build, clones into vendor/

The last five (gen3d's /gen_model pipeline, plus expert-streaming) clone into gitignored vendor/
subdirectories. Every venv here (.venv, venv-tts, vendor/*/venv) is independent
-- their torch/CUDA builds conflict with each other, so nothing imports across
venv boundaries; callers (gen3d/pipeline.py, ui/audio.py, ...) always talk to
these via subprocess.

All the version pins and patches below were worked out by hand against a
6 GB VRAM RTX 4050 laptop GPU -- see 3dtodo.md for the measurements and the
debugging trail (why each patch exists).

Usage:
    python3 setup.py                 # set up everything
    python3 setup.py --only unirig   # just one component (see SETUP_FUNCS below for the full list)
    python3 setup.py --dry-run       # print what would run, touch nothing
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FLOWAI_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = FLOWAI_ROOT / "vendor"
MAIN_VENV = FLOWAI_ROOT / ".venv"
TTS_VENV = FLOWAI_ROOT / "venv-tts"
TTS_UV_VENV = FLOWAI_ROOT / "venv-uv"
BUILD_TOOLS_VENV = FLOWAI_ROOT / "venv-build-tools"

# These four point at IlyaFeoktistov's own forks, not the upstream repos --
# each carries a "flowai-patches" branch (or "flowai-expert-streaming" for
# llama.cpp) with this project's local edits committed as real, permanent
# commits, instead of upstream + a patch reapplied by this script on every
# run. Two failure modes this avoids: (1) a PR ref (see
# EXPERT_STREAMING_PR_REF below) can vanish from GitHub with nothing this
# repo controls; (2) even a same-machine local edit, if only ever applied to
# the working tree and never committed anywhere durable, is invisible to
# `git status` in flowAI's OWN repo (vendor/ is gitignored) and silently
# gone the moment vendor/ gets recreated. All 4 forks are public --
# `git clone` over HTTPS needs no credentials to read them, same as the
# upstream repos they replace.
HUNYUAN_URL = "https://github.com/IlyaFeoktistov/Hunyuan3D-2GP.git"
UNIRIG_URL = "https://github.com/IlyaFeoktistov/UniRig.git"
ANIMATO_URL = "https://github.com/IlyaFeoktistov/Animato.git"  # fork, no local patches (kept for the same "commit can't vanish" reason as the other 4)
SUPERMAT_URL = "https://github.com/IlyaFeoktistov/SuperMat.git"
LLAMA_CPP_URL = "https://github.com/IlyaFeoktistov/llama.cpp.git"

# expert_streaming.py's docstring has the full story (what/why/tradeoffs) --
# in short, mainline llama.cpp has no dynamic per-token MoE expert
# offloading, only a static once-at-load CPU/GPU split (-ngl/-ncmoe/-ot).
# A real one exists as an unmerged community PR (#26824, closed not merged).
# Its head commit (originally refs/pull/26824/head) plus this project's own
# gpt-oss/Ollama GGUF compat fix now live as real commits on LLAMA_CPP_URL's
# "flowai-expert-streaming" branch -- see that URL's comment above for why.

# Only the multi-view checkpoint is needed -- gen3d/material_wrapper.py always
# runs inference_supermat_mv.py, never the single-image script (see
# gen3d/blender_scripts/render_multiview.py's docstring for why: SuperMat's
# single-image mode needs a LIT rendered view for shading cues, and its output
# isn't in the mesh's UV space anyway -- multi-view + our own back-projection
# is the only path that produces a UV texture covering the whole mesh).
# NOTE: inference_supermat_mv.py's own --checkpoint default says
# "checkpoints/supermat_mv.ckpt" but the file actually published on HF is
# "supermat_mv.pth" -- confirmed via the HF API's file listing, the .ckpt
# default in their own script is just stale/wrong. gen3d/material_wrapper.py
# must pass --checkpoint explicitly, not rely on that default.
SUPERMAT_CHECKPOINT_URL = "https://huggingface.co/oyiya/SuperMat/resolve/main/supermat_mv.pth"

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

DRY_RUN = False


def _run(cmd, **kw):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    if DRY_RUN:
        return
    subprocess.run(cmd, check=True, **kw)


def _clone_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  {dest} already exists, skipping clone (git pull it yourself to update)")
        return
    if DRY_RUN:
        print(f"  would clone {url} -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", url, str(dest)])


def _checkout_branch(dest: Path, branch: str) -> None:
    """Checks out `branch` in an already-cloned repo -- a no-op if it's
    already checked out. Used for the forked vendors above (HUNYUAN_URL,
    UNIRIG_URL, SUPERMAT_URL) whose flowAI-specific commits live on a branch
    other than the fork's default (mirrors upstream's own default branch,
    which this project doesn't otherwise touch)."""
    if DRY_RUN:
        print(f"  would checkout {branch} in {dest}")
        return
    _run(["git", "checkout", branch], cwd=dest)


def _check_prereqs() -> bool:
    """System-level prerequisites that need sudo/manual install -- never run
    sudo ourselves (matches how every sudo step in this project's own setup
    trail was always left to the user to run and approve)."""
    missing = []
    if shutil.which("python3.11") is None:
        missing.append(("python3.11", "sudo apt install python3.11 python3.11-venv"))
    if shutil.which("python3.12") is None:
        missing.append(("python3.12", "sudo apt install python3.12 python3.12-venv"))
    if shutil.which("nvcc") is None:
        missing.append(("nvcc (CUDA toolkit)", "sudo apt install nvidia-cuda-toolkit"))
    if shutil.which("blender") is None:
        missing.append((
            "blender",
            "sudo snap install blender --classic  "
            "# NOT apt's blender package -- its Cycles bake is broken for skinned "
            "meshes (see 3dtodo.md), the official snap build is required",
        ))
    if shutil.which("uv") is None:
        missing.append(("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"))
    if shutil.which("curl") is None:
        missing.append(("curl", "sudo apt install curl"))

    if missing:
        print("Missing system prerequisites -- install these yourself, then rerun:\n")
        for name, cmd in missing:
            print(f"  {name}:")
            print(f"    {cmd}\n")
        return False
    return True


def setup_main() -> None:
    print("\n=== flowAI main .venv ===")
    if not MAIN_VENV.exists():
        _run(["python3", "-m", "venv", str(MAIN_VENV)])
    pip = MAIN_VENV / "bin" / "pip"
    _run([str(pip), "install", "-r", "requirements.txt"], cwd=FLOWAI_ROOT)

    env_file = FLOWAI_ROOT / ".env"
    env_example = FLOWAI_ROOT / ".env.example"
    if env_file.exists():
        print("  .env already exists, leaving it alone")
    elif not env_example.is_file():
        print(f"  SKIPPED -- {env_example} not found")
    elif DRY_RUN:
        print(f"  would copy {env_example} -> {env_file}")
    else:
        env_file.write_text(env_example.read_text())
        print("  copied .env.example -> .env")
    print("  done. Pull chat/vision Ollama models yourself, e.g.:\n"
          "    ollama pull qwen3-coder:30b\n"
          "    ollama pull qwen2.5vl:7b")


def setup_image_gen() -> None:
    print("\n=== Image generation/editing (generate_image/edit_image, needs GPU) ===")
    pip = MAIN_VENV / "bin" / "pip"
    if not pip.is_file() and not DRY_RUN:
        print(f"  SKIPPED -- {MAIN_VENV} doesn't exist yet, run setup_main first (python3 setup.py --only main)")
        return
    _run([str(pip), "install", "diffusers", "torch", "transformers", "accelerate"])
    print("  done. Model (~4-7 GB) downloads itself on the first generate_image/edit_image call.")


def setup_whisper() -> None:
    print("\n=== Voice input (Alt+R, faster-whisper, CPU) ===")
    pip = MAIN_VENV / "bin" / "pip"
    if not pip.is_file() and not DRY_RUN:
        print(f"  SKIPPED -- {MAIN_VENV} doesn't exist yet, run setup_main first (python3 setup.py --only main)")
        return
    _run([str(pip), "install", "faster-whisper"])
    print("  done. Recognition model (default \"medium\", ~1.5 GB) downloads itself on first use "
          "(size configurable in /settings).")


def setup_tts() -> None:
    print("\n=== Voice output (Chatterbox TTS, own venv-tts, CPU) ===")
    print("  Chatterbox + Russian stress marking needs Python 3.11 and its own "
          "torch/transformers versions, incompatible with the main .venv (generate_image's) -- "
          "lives in its own venv-tts instead of requirements.txt.")

    # uv, installed into its own tiny venv-uv (not system-wide), just to fetch a
    # Python 3.11 build without needing sudo/a system package for it.
    uv_pip = TTS_UV_VENV / "bin" / "pip"
    uv_bin = TTS_UV_VENV / "bin" / "uv"
    if not TTS_UV_VENV.exists():
        _run(["python3", "-m", "venv", str(TTS_UV_VENV)])
    if not uv_bin.is_file() and not DRY_RUN:
        _run([str(uv_pip), "install", "uv"])
    if DRY_RUN:
        print(f"  would run: {uv_bin} python install 3.11")
        py311 = "<python3.11 path>"
    else:
        _run([str(uv_bin), "python", "install", "3.11"])
        py311 = subprocess.run([str(uv_bin), "python", "find", "3.11"],
                                capture_output=True, text=True, check=True).stdout.strip()

    if not TTS_VENV.exists():
        _run([py311, "-m", "venv", str(TTS_VENV)])
    tts_pip = TTS_VENV / "bin" / "pip"

    # numpy BEFORE everything else -- pkuseg (a chatterbox-tts dependency) fails
    # to build against a numpy version pulled in later in the same resolve.
    _run([str(tts_pip), "install", "numpy"])
    _run([str(tts_pip), "install", "chatterbox-tts"])

    # Without this, Chatterbox speaks Russian with no stress marks at all --
    # noticeably wrong prosody. --no-deps first (its own setup.py pulls in an
    # unpinned dependency set that conflicts with the explicit pins right below).
    _run([str(tts_pip), "install", "--no-deps",
          "russian-text-stresser @ git+https://github.com/Vuizur/add-stress-to-epub"])
    _run([str(tts_pip), "install", "spacy==3.6.*", "beautifulsoup4>=4.11.1", "lxml>=4.9.1",
          "pymorphy2>=0.9.1", "pymorphy2-dicts-ru>=2.4.417127.4579844",
          "stressed-cyrillic-tools>=0.1.10", "transliterate>=1.10.2"])

    # perth (Chatterbox's watermarking lib) loads its weights via pkg_resources,
    # which newer setuptools dropped entirely.
    _run([str(tts_pip), "install", "setuptools<81"])
    print("  done. Model (~2 GB, Coqui CPML license -- non-commercial) downloads itself on first voice reply.")


def setup_hunyuan3d() -> None:
    print("\n=== Hunyuan3D-2GP (mesh + texture generation) ===")
    dest = VENDOR_DIR / "hunyuan3d-2gp"
    _clone_if_missing(HUNYUAN_URL, dest)
    # "flowai-patches" branch carries this project's trust_remote_code=True
    # fix (hy3dgen/texgen/utils/multiview_utils.py, needed for diffusers>=0.39's
    # custom hunyuanpaint pipeline loading) as a real commit -- see HUNYUAN_URL's
    # comment above for why this replaced re-patching the file on every run.
    _checkout_branch(dest, "flowai-patches")

    venv = dest / "venv"
    if not venv.exists():
        _run(["python3.11", "-m", "venv", str(venv)])
    pip = venv / "bin" / "pip"

    _run([str(pip), "install", "torch==2.5.1", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu124"])
    _run([str(pip), "install", "-r", "requirements.txt"], cwd=dest)
    # torch must already be importable in this venv for diso's build step to
    # find it -- build isolation would create a fresh env without it.
    _run([str(pip), "install", "diso", "--no-build-isolation"])
    print("  done. VRAM peak on a 6 GB card was ~2.2 GB at profile 4 (see 3dtodo.md).")


def setup_unirig() -> None:
    print("\n=== UniRig (skeleton prediction) ===")
    dest = VENDOR_DIR / "unirig"
    _clone_if_missing(UNIRIG_URL, dest)
    # "flowai-patches" branch carries this project's 2 fixes (sdpa attention
    # instead of a hard flash_attention_2 dep, and dropping requirements.txt's
    # bare flash_attn line -- installed separately below with an exact
    # ABI-matched wheel) as real commits -- see UNIRIG_URL's comment above.
    _checkout_branch(dest, "flowai-patches")

    venv = dest / "venv"
    if not venv.exists():
        _run(["python3.11", "-m", "venv", str(venv)])
    pip = venv / "bin" / "pip"

    _run([str(pip), "install", "torch==2.3.1", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu121"])
    _run([str(pip), "install", "-r", "requirements.txt"], cwd=dest)
    _run([str(pip), "install", "spconv-cu121"])
    _run([str(pip), "install", "torch_scatter", "torch_cluster",
          "-f", "https://data.pyg.org/whl/torch-2.3.1+cu121.html", "--no-cache-dir"])
    _run([str(pip), "install", "numpy==1.26.4"])
    # The exact ABI match matters: the newest flash-attn release only ships a
    # torch2.4 wheel, which fails to import against torch 2.3.1 with
    # "undefined symbol: _ZNK3c105Error4whatEv" (C++ ABI mismatch). v2.6.3 is
    # the last release with a wheel built specifically for torch2.3+cu121.
    _run([str(pip), "install", FLASH_ATTN_WHEEL])
    print("  done. Skeleton step peaked at ~4.0 GB VRAM on a 6 GB card (see 3dtodo.md).")


def setup_animato() -> None:
    print("\n=== Animato (AI-driven animation) ===")
    dest = VENDOR_DIR / "animato"
    _clone_if_missing(ANIMATO_URL, dest)

    _run(["uv", "python", "install", "3.13"], cwd=dest)
    _run(["uv", "sync"], cwd=dest)
    print("  done. Frontend build skipped -- gen3d/animato_client.py drives the API directly.")


def setup_supermat() -> None:
    print("\n=== SuperMat (AI roughness/metallic estimation) ===")
    dest = VENDOR_DIR / "supermat"
    _clone_if_missing(SUPERMAT_URL, dest)
    # "flowai-patches" branch carries this project's fix (dropping
    # requirements.txt's bare torch/torchvision/xformers lines -- installed
    # separately below with exact pins) as a real commit -- see SUPERMAT_URL's
    # comment above.
    _checkout_branch(dest, "flowai-patches")

    venv = dest / "venv"
    if not venv.exists():
        _run(["python3.12", "-m", "venv", str(venv)])
    pip = venv / "bin" / "pip"

    _run([str(pip), "install", "torch==2.5.1", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu124"])

    # xformers is a HARD top-level import in src/models/custom_attention_processor.py
    # AND diffusers' own attention_processor.py unconditionally does
    # `import xformers.ops` too (confirmed by actually running
    # material_wrapper.py) -- needed regardless of whether it's ever called
    # at runtime (gen3d/material_wrapper.py always passes use_xformers=False).
    # Pulling it in unpinned via requirements.txt risks pip upgrading the
    # CUDA-specific torch build above to satisfy xformers' own pin (same class
    # of problem setup_unirig() avoids by stripping flash_attn) -- installed
    # separately below with an exact pin instead. Plain `pip install xformers
    # --no-deps` grabbed 0.0.35 (built for torch 2.10+cu128) here, which fails
    # at import time with an aten::_flash_attention_forward schema mismatch
    # against torch 2.5.1 -- 0.0.28.post3 is the release actually built
    # against torch==2.5.1 (confirmed importable: `import xformers.ops`
    # succeeds, even though its CUDA kernels are never exercised).
    _run([str(pip), "install", "-r", "requirements.txt"], cwd=dest)
    _run([str(pip), "install", "xformers==0.0.28.post3", "--no-deps"])
    # einops: used by src/models/custom_attention_processor.py but missing
    # from the repo's own requirements.txt entirely (confirmed by actually
    # running material_wrapper.py -- a plain ModuleNotFoundError, not a
    # version conflict, so no special pin needed here).
    _run([str(pip), "install", "einops"])

    checkpoints_dir = dest / "checkpoints"
    checkpoint_path = checkpoints_dir / "supermat_mv.pth"
    if checkpoint_path.exists():
        print(f"  checkpoint already downloaded: {checkpoint_path}")
    elif DRY_RUN:
        print(f"  would download {SUPERMAT_CHECKPOINT_URL} -> {checkpoint_path}")
    else:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        _run(["curl", "-L", "-o", str(checkpoint_path), SUPERMAT_CHECKPOINT_URL])
    print("  done. Only the multi-view checkpoint is fetched -- see SUPERMAT_CHECKPOINT_URL's comment above.")


def setup_expert_streaming() -> None:
    print("\n=== Experimental MoE expert-streaming llama-server (see expert_streaming.py) ===")
    dest = VENDOR_DIR / "llama-expert-streaming"
    _clone_if_missing(LLAMA_CPP_URL, dest)
    # "flowai-expert-streaming" branch on the fork already IS PR #26824's
    # head (refs/pull/26824/head, fetched once and pushed there) plus this
    # project's gpt-oss/Ollama GGUF compat fix, as real commits -- no more
    # fetching a PR ref that could vanish from GitHub independently of
    # anything this repo controls (see LLAMA_CPP_URL's comment above).
    _checkout_branch(dest, "flowai-expert-streaming")

    # cmake/ninja via their own tiny venv, not `pip install --user` (Debian's
    # python3 is PEP 668 externally-managed -- a plain --user install is
    # refused unless --break-system-packages, which we don't want to force
    # on anyone) and not apt (sudo -- see _check_prereqs' docstring on why
    # this project never runs sudo itself). Same "own tiny venv just to get
    # a build tool without touching the system" idea as TTS_UV_VENV above.
    if not BUILD_TOOLS_VENV.exists():
        _run(["python3", "-m", "venv", str(BUILD_TOOLS_VENV)])
    build_tools_bin = BUILD_TOOLS_VENV / "bin"
    _run([str(build_tools_bin / "pip"), "install", "--quiet", "cmake", "ninja"])

    # GGML_CUDA=ON -- this machine's GPU (see CLAUDE.md: RTX 4050 Laptop,
    # 5.9 GB VRAM) needs the CUDA backend for -ehs (VRAM-resident expert hot
    # store) to mean anything; LLAMA_CURL=OFF because this build never
    # downloads models itself (expert_streaming.py points -m at an existing
    # Ollama blob), so libcurl is dead weight, not a real prerequisite.
    env = {**os.environ, "PATH": f"{build_tools_bin}:{os.environ.get('PATH', '')}"}
    build_dir = dest / "build"
    if DRY_RUN:
        print(f"  would configure+build {build_dir} with GGML_CUDA=ON")
    else:
        subprocess.run(
            ["cmake", "-B", str(build_dir), "-G", "Ninja", "-DGGML_CUDA=ON",
             "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"],
            check=True, cwd=dest, env=env,
        )
        import multiprocessing
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release",
             "-j", str(multiprocessing.cpu_count())],
            check=True, cwd=dest, env=env,
        )
    print(
        "  done. Binary: vendor/llama-expert-streaming/build/bin/llama-server\n"
        "  This is an UNMERGED, experimental fork (see expert_streaming.py's "
        "docstring for exactly which PR, why it's not in mainline, and the "
        "known PP-for-TG tradeoff a real tester reported). Toggle "
        "'экспериментальный expert-streaming backend' in /settings to try it "
        "against the current model -- it only takes effect for the legacy "
        "agent path (pipeline_mode=ВЫКЛ)."
    )


# Order matters when running everything (no --only): main must exist before
# image-gen/whisper install into it, the rest (gen3d's own vendor/*/venv's)
# don't depend on any of the earlier ones.
SETUP_FUNCS = {
    "main": setup_main,
    "image-gen": setup_image_gen,
    "whisper": setup_whisper,
    "tts": setup_tts,
    "hunyuan3d-2gp": setup_hunyuan3d,
    "unirig": setup_unirig,
    "animato": setup_animato,
    "supermat": setup_supermat,
    "expert-streaming": setup_expert_streaming,
}


def main() -> int:
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(SETUP_FUNCS), help="set up just one component")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, touch nothing")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    if DRY_RUN:
        print("=== DRY RUN -- no commands will actually execute ===")

    if not _check_prereqs() and not DRY_RUN:
        return 1

    VENDOR_DIR.mkdir(exist_ok=True)
    targets = [args.only] if args.only else list(SETUP_FUNCS)
    for name in targets:
        SETUP_FUNCS[name]()

    print("\nAll done. Try it: ./flowai, then `/gen_model a cute penguin`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
