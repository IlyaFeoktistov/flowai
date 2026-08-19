"""Headless Blender multi-view render for SuperMat's material estimation
(vendor/supermat, run via gen3d/material_wrapper.py).

SuperMat needs a LIT rendered view of the object -- shading cues (highlights,
falloff) are what it reasons about to guess roughness/metallic. Feeding it our
own EMIT-baked albedo (flat, unlit -- see rebake_texture.py) would give it no
signal at all, so this renders the already-retextured mesh under a plain
studio light setup instead of using the baked albedo directly as the input.

Run as:
    blender --background --python render_multiview.py -- <input.glb> <output_dir> [num_views]

num_views defaults to 6, evenly spaced around a single ring (matching
SuperMat's own example convention, e.g. their bundled bag_rendered_6views/).

Writes to output_dir:
    color_0000.png .. color_{num_views-1:04d}.png -- RGBA renders, transparent
        background (SuperMat's own loader expects RGBA with alpha=0 outside
        the object, see src/utils.py's load_rgba_image_as_rgb_tensor).
    cameras.json -- each camera's matrix_world (row-major 4x4) and the object
        center it looks at, keyed by the same index used in the filenames --
        NOT the same schema as SuperMat's own meta.json (we don't use
        --use-camera-embeds, see gen3d/material_wrapper.py's docstring for
        why), this is purely for gen3d/blender_scripts/project_material_to_uv.py's
        own back-projection step to reconstruct each camera's exact transform.
"""
import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
in_path = argv[0]
out_dir = argv[1]
num_views = int(argv[2]) if len(argv) > 2 else 6

IMAGE_SIZE = 512  # matches SuperMat's own --image-size default

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=in_path)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError("no mesh objects found after import")
bpy.context.view_layer.objects.active = mesh_objs[0]
for o in mesh_objs:
    o.select_set(True)
if len(mesh_objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# Object center + radius from the world-space bounding box -- used to place
# the camera ring and the look-at target, works regardless of the mesh's own
# local origin/scale.
corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
min_c = Vector((min(c[i] for c in corners) for i in range(3)))
max_c = Vector((max(c[i] for c in corners) for i in range(3)))
center = (min_c + max_c) / 2
radius = max((max_c - min_c)[i] for i in range(3)) / 2

# Studio-style lighting: three-point (key/fill/rim) plus a bit of world
# ambient so unlit-facing areas aren't pure black -- SuperMat needs believable
# shading cues, not physically exact ones.
key = bpy.data.objects.new("key", bpy.data.lights.new("key", type="AREA"))
key.data.energy = 1000
key.data.size = radius * 2
key.location = center + Vector((radius * 3, -radius * 3, radius * 3))
bpy.context.collection.objects.link(key)

fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", type="AREA"))
fill.data.energy = 400
fill.data.size = radius * 3
fill.location = center + Vector((-radius * 3, -radius * 2, radius * 1.5))
bpy.context.collection.objects.link(fill)

rim = bpy.data.objects.new("rim", bpy.data.lights.new("rim", type="AREA"))
rim.data.energy = 600
rim.data.size = radius * 2
rim.location = center + Vector((0, radius * 3, radius * 2))
bpy.context.collection.objects.link(rim)

for light_obj in (key, fill, rim):
    direction = center - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("studio_world")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
bg.inputs["Strength"].default_value = 0.3

# Camera + single ring of `num_views` evenly spaced azimuths, level with the
# object's own center (matches SuperMat's example naming, e.g.
# bag_rendered_6views/ -- a single elevation ring, not a full dome).
cam_data = bpy.data.cameras.new("cam")
cam_obj = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Distance so the object fits in frame with a margin, from the camera's FOV.
distance = (radius / math.sin(cam_data.angle / 2)) * 1.4

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.render.resolution_x = IMAGE_SIZE
scene.render.resolution_y = IMAGE_SIZE
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

cameras_meta = {}
for i in range(num_views):
    angle = 2 * math.pi * i / num_views
    cam_obj.location = center + Vector((math.cos(angle), math.sin(angle), 0)) * distance
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    scene.render.filepath = f"{out_dir}/color_{i:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"=== rendered view {i}: {scene.render.filepath} ===")

    cameras_meta[f"{i:04d}"] = {
        "matrix_world": [list(row) for row in cam_obj.matrix_world],
        "angle": cam_data.angle,
        "center": list(center),
    }

with open(f"{out_dir}/cameras.json", "w") as f:
    json.dump(cameras_meta, f, indent=2)

print(f"=== done, {num_views} views + cameras.json saved to {out_dir} ===")
