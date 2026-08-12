"""Parameterized image-to-3D generation via Hunyuan3D-2GP (mesh + baked albedo
texture). Adapted from that repo's own test_generate.py, which hardcodes
assets/demo.png as input -- this version takes a real image and output path so
gen3d/pipeline.py can drive it for arbitrary inputs.

Runs INSIDE the vendor/hunyuan3d-2gp venv (its own torch/mmgp/hy3dgen build,
incompatible with flowAI's main venv and with vendor/unirig's), invoked as
either:

    <vendor>/hunyuan3d-2gp/venv/bin/python gen3d/hunyuan_wrapper.py \\
        --vendor-dir <vendor>/hunyuan3d-2gp --image <path> --output <path.glb> \\
        [--profile 4]

    # multi-view: several views of ONE object fused into one shape (see below)
    <vendor>/hunyuan3d-2gp/venv/bin/python gen3d/hunyuan_wrapper.py \\
        --vendor-dir <vendor>/hunyuan3d-2gp --front <path> [--left <path>] \\
        [--back <path>] [--right <path>] --output <path.glb> [--profile 4]

profile is the mmgp offload profile (see 3dtodo.md for the profile 3 vs 4
measurements on 6 GB VRAM -- 4 is the default, more headroom for a small
speed cost).

Multi-view mode (any of --front/--left/--back/--right given instead of
--image) loads a DIFFERENT checkpoint, "Hunyuan3D-2mv", whose
hunyuan3d-dit-v2-mv config wires in hy3dgen's MVImageProcessorV2
(vendor/hunyuan3d-2gp/hy3dgen/shapegen/preprocessors.py) instead of the
default single-image ImageProcessorV2 -- confirmed against that vendor
repo's own examples/shape_gen_multiview.py: passing a
{"front": img, "left": img, ...} dict genuinely FUSES the views into one
shape-conditioning tensor (view2idx-ordered), unlike passing a bare list of
images to the single-view pipeline (which just batches them as independent,
unrelated generations -- see cli.py's own "@ref @ref @ref = batch" comment).
front/left/back/right are the model's own fixed training convention (0/90/
180/270 degrees around the object), not arbitrary angles -- see cli.py's
/help text and gen_model_server.py's docstring for the user-facing
explanation. Texture generation still only takes ONE image regardless of
mode (Hunyuan3D-2's paint pipeline isn't multi-view-aware) -- front if given,
else whichever view was provided.
"""
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--vendor-dir", required=True, help="path to the cloned Hunyuan3D-2GP repo")
parser.add_argument("--image", help="input reference image (single-view mode)")
parser.add_argument("--front", help="front view (multi-view mode)")
parser.add_argument("--left", help="left view, 90 degrees from front (multi-view mode)")
parser.add_argument("--back", help="back view, 180 degrees from front (multi-view mode)")
parser.add_argument("--right", help="right view, 270 degrees from front (multi-view mode)")
parser.add_argument("--output", required=True, help="output .glb path")
parser.add_argument("--profile", type=int, default=4, help="mmgp offload profile")
args = parser.parse_args()

mv_views = {k: v for k, v in (("front", args.front), ("left", args.left),
                               ("back", args.back), ("right", args.right)) if v}
if mv_views and args.image:
    parser.error("--image is single-view mode, --front/--left/--back/--right is multi-view -- pick one")
if not mv_views and not args.image:
    parser.error("need either --image or at least one of --front/--left/--back/--right")

sys.path.insert(0, args.vendor_dir)
os.chdir(args.vendor_dir)  # hy3dgen resolves some of its own asset paths relative to cwd

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import torch
from PIL import Image

from mmgp import offload
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def replace_property_getter(obj, name, getter):
    cls = obj.__class__
    new_cls = type(cls.__name__, (cls,), {name: property(getter)})
    obj.__class__ = new_cls


def vram_used():
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3


if mv_views:
    print(f"=== Loading shape model (Hunyuan3D-2mv, views: {list(mv_views)}) ===")
    t0 = time.time()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        variant="fp16",
    )
else:
    print("=== Loading shape model (Hunyuan3D-2mini) ===")
    t0 = time.time()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        use_safetensors=True,
        device="cuda",
    )
print(f"shape model loaded in {time.time()-t0:.1f}s")

print("=== Loading texture model (Hunyuan3D-2) ===")
t0 = time.time()
texgen_worker = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
print(f"texgen model loaded in {time.time()-t0:.1f}s")

replace_property_getter(i23d_worker, "_execution_device", lambda self: "cuda")
pipe = offload.extract_models("i23d_worker", i23d_worker)
pipe.update(offload.extract_models("texgen_worker", texgen_worker))
texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True

profile = args.profile
kwargs = {}
if profile < 5:
    kwargs["pinnedMemory"] = "i23d_worker/model"
if profile != 1 and profile != 3:
    kwargs["budgets"] = {"*": 2200}
print(f"=== Using offload profile {profile} ===")
offload.default_verboseLevel = 1
offload.profile(pipe, profile_no=profile, verboseLevel=1, **kwargs)

def _load_and_remove_bg(path: str):
    image = Image.open(path)
    if image.mode == "RGB":
        # Opaque photo/generated image, no alpha yet -- cut the subject out so
        # the shape model conditions on it alone, not whatever's behind it.
        # Must check BEFORE any RGBA conversion: forcing RGBA first makes this
        # check always false (mode is then always "RGBA"), silently skipping
        # background removal entirely -- which is what this file used to do.
        rembg = BackgroundRemover()
        image = rembg(image)  # rembg.remove() already returns RGBA
    else:
        image = image.convert("RGBA")
    return image


if mv_views:
    print(f"=== Preparing {len(mv_views)} input views: {mv_views} ===")
    images = {view: _load_and_remove_bg(path) for view, path in mv_views.items()}
    # Texture generation (below) only ever takes ONE image -- front if given
    # (matches vendor's own examples/textured_shape_gen_multiview.py), else
    # whichever view is actually available.
    texture_ref_image = images.get("front") or next(iter(images.values()))
else:
    print(f"=== Preparing input image: {args.image} ===")
    images = _load_and_remove_bg(args.image)
    texture_ref_image = images

torch.cuda.reset_peak_memory_stats()
print("=== Generating shape ===")
t0 = time.time()
mesh = i23d_worker(image=images)[0]
t_shape = time.time() - t0
alloc, peak = vram_used()
print(f"shape generated in {t_shape:.1f}s | vram now={alloc:.2f}GB peak={peak:.2f}GB | verts={len(mesh.vertices)} faces={len(mesh.faces)}")

print("=== Generating texture (baked to UV) ===")
t0 = time.time()
mesh = texgen_worker(mesh, image=texture_ref_image)
t_tex = time.time() - t0
alloc, peak = vram_used()
print(f"texture generated in {t_tex:.1f}s | vram now={alloc:.2f}GB peak={peak:.2f}GB")

mesh.export(args.output)
print(f"=== DONE. total={t_shape+t_tex:.1f}s. Saved to {args.output} ===")
