"""AI roughness/metallic estimation via SuperMat's multi-view UNet
(vendor/supermat). Adapted from that repo's own inference_supermat_mv.py --
this version always disables camera embeddings and xformers, and only ever
uses the multi-view checkpoint (see gen3d/setup.py's SUPERMAT_CHECKPOINT_URL
comment and gen3d/blender_scripts/render_multiview.py's docstring for why:
SuperMat's per-view estimates aren't in the mesh's UV space, and their
mv-mode is the only one that fits the "same object, several rendered angles"
input we already produce -- gen3d/blender_scripts/project_material_to_uv.py
back-projects the per-view outputs this script writes into the mesh's real UV).

Runs INSIDE the vendor/supermat venv (its own torch/diffusers build,
incompatible with flowAI's main venv and with the other vendor venvs), invoked
as:

    <vendor>/supermat/venv/bin/python gen3d/material_wrapper.py \\
        --vendor-dir <vendor>/supermat --input <dir with color_0000.png..> \\
        --output-dir <dir> [--checkpoint <path>] [--num-views 6] [--image-size 512]

--checkpoint defaults to <vendor-dir>/checkpoints/supermat_mv.pth -- NOT
inference_supermat_mv.py's own default (which says .ckpt, a stale filename
that doesn't match what's actually published on HuggingFace).

Writes to --output-dir, one file per input view index:
    albedo_XXXX.png, roughness_XXXX.png, metallic_XXXX.png
(each in that view's own image space, not the mesh's UV -- see
project_material_to_uv.py for the back-projection step).
"""
import argparse
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--vendor-dir", required=True, help="path to the cloned SuperMat repo")
parser.add_argument("--input", required=True, help="directory with color_0000.png.. rendered views")
parser.add_argument("--output-dir", required=True, help="directory to write albedo/roughness/metallic PNGs to")
parser.add_argument("--checkpoint", default=None, help="path to supermat_mv.pth (defaults to <vendor-dir>/checkpoints/supermat_mv.pth)")
parser.add_argument("--base-model", default="stabilityai/stable-diffusion-2-1")
parser.add_argument("--num-views", type=int, default=6)
parser.add_argument("--image-size", type=int, default=512)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

vendor_dir = Path(args.vendor_dir).resolve()
sys.path.insert(0, str(vendor_dir))
os.chdir(vendor_dir)  # src.* imports below are relative to the repo root

checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else vendor_dir / "checkpoints" / "supermat_mv.pth"
if not checkpoint_path.is_file():
    print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
    sys.exit(1)

import torch
from PIL import Image
from diffusers import DDIMScheduler

from src.adapters import SuperMatAdapterWrapper
from src.pipelines.pipeline_supermat_stable_diffusion import SuperMatStableDiffusionPipeline
from src.models.model_utils import set_supermat_mv_self_attention
from src.utils import collect_multi_view_images, load_unet_weights, load_rgba_image_as_rgb_tensor, to_uint8_rgb, orm_to_roughness_metallic, parse_image_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Loading SuperMat multi-view pipeline (SD2.1-based) ===")
t0 = time.time()
pipe = SuperMatStableDiffusionPipeline.from_pretrained(
    args.base_model,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# use_camera_embeddings=False -- inference_supermat_mv.py's --use-camera-embeds
# is opt-in and needs a meta.json matching SuperMat's own camera convention,
# which render_multiview.py deliberately doesn't produce (see its docstring).
pipe = SuperMatAdapterWrapper.convert(
    pipe,
    use_camera_embeddings=False,
    camera_embeddings_dim=16,
    replicate_num=2,
)

# use_xformers=False -- xformers is a hard top-level import in this repo's own
# code (needed regardless), but gen3d/setup.py installs it with --no-deps
# against a CUDA build it wasn't compiled for (see that function's own
# comment), so its memory-efficient attention kernels can't actually be used
# here -- only the plain import needs to succeed.
set_supermat_mv_self_attention(pipe.unet, num_views=args.num_views, use_xformers=False)

print("Loading UNet weights from checkpoint...")
unet_weights = load_unet_weights(checkpoint_path)
incompatible = pipe.unet.load_state_dict(unet_weights, strict=False)
pipe.unet.eval()
print(incompatible)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe = pipe.to(device)
print(f"pipeline ready in {time.time()-t0:.1f}s")

input_dir = Path(args.input)
image_paths = collect_multi_view_images(input_dir)
if len(image_paths) != args.num_views:
    print(f"ERROR: found {len(image_paths)} color_*.png in {input_dir}, expected --num-views={args.num_views}", file=sys.stderr)
    sys.exit(1)

source_images = torch.cat(
    [load_rgba_image_as_rgb_tensor(p, image_size=args.image_size, device=device) for p in image_paths],
    dim=0,
)

generator = None
if args.seed is not None:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed)

print("=== Estimating albedo/roughness/metallic ===")
t0 = time.time()
with torch.no_grad():
    outputs = pipe(
        prompt="",
        num_inference_steps=1,
        num_images_per_prompt=args.num_views,
        source_image=source_images,
        output_type="pt",
        camera_embeds=None,
        generator=generator,
    )
print(f"estimation done in {time.time()-t0:.1f}s")

albedo_branch, orm_branch = outputs[0], outputs[1]

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
for local_idx, image_path in enumerate(image_paths):
    image_index = parse_image_index(image_path)
    albedo = to_uint8_rgb(albedo_branch[local_idx])
    roughness, metallic = orm_to_roughness_metallic(orm_branch[local_idx])
    Image.fromarray(albedo).save(output_dir / f"albedo_{image_index}.png")
    Image.fromarray(roughness).save(output_dir / f"roughness_{image_index}.png")
    Image.fromarray(metallic).save(output_dir / f"metallic_{image_index}.png")

print(f"=== DONE. saved to {output_dir} ===")
