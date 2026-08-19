"""Headless Blender back-projection: takes the per-view roughness/metallic
images gen3d/material_wrapper.py estimated (one pair per camera in
render_multiview.py's ring, each in that camera's own screen space) and bakes
them onto the mesh's REAL UV -- SuperMat's own inference script has no UV
output at all (see gen3d/material_wrapper.py's docstring), this is the step
that actually produces a texture covering the whole mesh.

Run as:
    blender --background --python project_material_to_uv.py -- \\
        <mesh.glb> <cameras.json> <material_dir> <output.glb> [texture_size]

texture_size defaults to 2048 (matches rebake_texture.py's own default).
material_dir must contain roughness_XXXX.png/metallic_XXXX.png per camera
index, cameras.json must be render_multiview.py's own output (matrix_world +
angle per index) -- NOT SuperMat's meta.json schema, see that script's
docstring for why they differ.

Per camera: build a temporary "camera-projected" UV layer via
bpy_extras.object_utils.world_to_camera_view (plain per-vertex screen-space
math, not Blender's interactive "Project From View" -- that needs a live
viewport, unusable headless), then self-bake (EMIT, not DIFFUSE -- see
rebake_texture.py's own docstring for why) that camera's roughness/metallic
image into the mesh's REAL uv, restricted to whichever faces this camera is
the "best" view for (face normal vs. direction-to-camera dot product, highest
among all cameras wins -- no occlusion raycast in this version, see
3dtodo.md's plan notes for the known limitation on self-occluding shapes).

Roughness and metallic end up combined into ONE image (G=roughness,
B=metallic) rather than two separate images -- Blender's glTF exporter only
auto-detects metallicRoughnessTexture when both are driven by the same image
through a Separate Color node (confirmed against the exporter's own behavior
while building rebake_texture.py's AO/occlusionTexture wiring); two separate
images would silently fail to export as a real metallicRoughnessTexture.
"""
import json
import math
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:]
mesh_path = argv[0]
cameras_json_path = argv[1]
material_dir = argv[2]
out_path = argv[3]
texture_size = int(argv[4]) if len(argv) > 4 else 2048

with open(cameras_json_path) as f:
    cameras_meta = json.load(f)
indices = sorted(cameras_meta.keys())

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=mesh_path)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError("no mesh objects found after import")
bpy.context.view_layer.objects.active = mesh_objs[0]
for o in mesh_objs:
    o.select_set(True)
if len(mesh_objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
mesh = obj.data

real_uv = mesh.uv_layers.active
if real_uv is None:
    raise RuntimeError("mesh has no active UV layer -- run retopologize()/rebake_texture() first")
real_uv_name = real_uv.name

print(f"=== {len(mesh.polygons)} faces, real UV = '{real_uv_name}' ===")

# Recreate each camera exactly as render_multiview.py placed it.
cam_objs = {}
for idx in indices:
    meta = cameras_meta[idx]
    cam_data = bpy.data.cameras.new(f"cam_{idx}")
    cam_data.angle = meta["angle"]
    cam_obj = bpy.data.objects.new(f"cam_{idx}", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    # Assigning a plain nested list to matrix_world silently drops the
    # translation column (confirmed empirically) -- must be an actual
    # mathutils.Matrix, and the object must already be linked to the scene
    # for the assignment to stick.
    cam_obj.matrix_world = Matrix(meta["matrix_world"])
    cam_objs[idx] = cam_obj
bpy.context.view_layer.update()

# Per-face best-camera assignment: highest dot(face normal, direction to
# camera) wins -- no visibility/occlusion raycast, see module docstring.
bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()
bm.normal_update()
face_best_cam = {}
for face in bm.faces:
    world_normal = (obj.matrix_world.to_3x3() @ face.normal).normalized()
    world_center = obj.matrix_world @ face.calc_center_median()
    best_idx, best_dot = None, -2.0
    for idx in indices:
        to_cam = (cam_objs[idx].matrix_world.translation - world_center).normalized()
        dot = world_normal.dot(to_cam)
        if dot > best_dot:
            best_dot, best_idx = dot, idx
    face_best_cam[face.index] = best_idx
bm.free()

for idx in indices:
    count = sum(1 for v in face_best_cam.values() if v == idx)
    print(f"=== camera {idx}: best view for {count} faces ===")

# Per-vertex world position, computed once -- world_to_camera_view needs the
# scene + camera + a world-space coordinate.
world_coords = [obj.matrix_world @ v.co for v in mesh.vertices]
scene = bpy.context.scene

roughness_combined = bpy.data.images.new("roughness_combined", width=texture_size, height=texture_size)
roughness_combined.colorspace_settings.name = 'Non-Color'
metallic_combined = bpy.data.images.new("metallic_combined", width=texture_size, height=texture_size)
metallic_combined.colorspace_settings.name = 'Non-Color'

# One material for the whole self-bake: a target texture node (swapped between
# the roughness/metallic combined images across the two passes below) and a
# source texture node re-pointed at each camera's estimated image + its own
# camera-projected UV layer in turn.
mat = bpy.data.materials.new("proj_mat")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out_node = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
src_uv_node = nt.nodes.new("ShaderNodeUVMap")
src_tex_node = nt.nodes.new("ShaderNodeTexImage")
nt.links.new(src_uv_node.outputs["UV"], src_tex_node.inputs["Vector"])
nt.links.new(src_tex_node.outputs["Color"], emit.inputs["Color"])
target_tex_node = nt.nodes.new("ShaderNodeTexImage")

# mesh_path already has albedo+normal+AO (rebake_texture.py's own output) --
# swapped out for this bake-only material for the duration of the self-bake
# below, then restored (with roughness/metallic added into it) right before
# export, so none of that is lost.
orig_mat = mesh.materials[0] if mesh.materials else None
mesh.materials.clear()
mesh.materials.append(mat)

scene.render.engine = "CYCLES"
scene.render.bake.use_selected_to_active = False
scene.render.bake.margin = 4

for pass_name, combined_img in (("roughness", roughness_combined), ("metallic", metallic_combined)):
    target_tex_node.image = combined_img
    nt.nodes.active = target_tex_node
    # Bake writes into whichever UV layer is active for rendering -- keep it
    # pinned to the mesh's real UV for every camera in this pass.
    mesh.uv_layers.active = real_uv

    for idx in indices:
        img_path = f"{material_dir}/{pass_name}_{idx}.png"
        src_img = bpy.data.images.load(img_path)
        src_img.colorspace_settings.name = 'Non-Color'
        src_tex_node.image = src_img

        # Build (or rebuild) this camera's projected UV layer -- per-loop, so
        # a vertex shared by faces with different best-cameras still gets a
        # correct per-face-corner value even though this loop only fires for
        # `idx`'s own layer.
        proj_name = f"proj_{idx}"
        if proj_name in mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[proj_name])
        proj_uv = mesh.uv_layers.new(name=proj_name)
        cam_obj = cam_objs[idx]
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                vert_index = mesh.loops[loop_index].vertex_index
                co = world_to_camera_view(scene, cam_obj, world_coords[vert_index])
                proj_uv.data[loop_index].uv = (co.x, co.y)
        src_uv_node.uv_map = proj_name

        for poly in mesh.polygons:
            poly.select = face_best_cam[poly.index] == idx
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        print(f"=== baking {pass_name} from camera {idx}, "
              f"active={bpy.context.view_layer.objects.active.name!r} "
              f"({bpy.context.view_layer.objects.active.type}), "
              f"selected={obj.select_get()} ===")
        result = bpy.ops.object.bake(type='EMIT')
        print(f"=== {pass_name}/{idx} bake result: {result} ===")

        bpy.data.images.remove(src_img)
        mesh.uv_layers.remove(proj_uv)

# Combine into one RGB image (R unused, G=roughness, B=metallic) -- the
# pattern Blender's glTF exporter auto-detects as metallicRoughnessTexture
# (Separate Color node fed by one shared image, G->Roughness, B->Metallic).
orm = bpy.data.images.new("roughness_metallic", width=texture_size, height=texture_size)
orm.colorspace_settings.name = 'Non-Color'
r_px = list(roughness_combined.pixels)
m_px = list(metallic_combined.pixels)
orm_px = list(orm.pixels)
for i in range(0, len(orm_px), 4):
    orm_px[i] = 0.0
    orm_px[i + 1] = r_px[i]      # roughness -> G
    orm_px[i + 2] = m_px[i]      # metallic -> B
    orm_px[i + 3] = 1.0
orm.pixels = orm_px

# Restore the original material (albedo+normal+AO from rebake_texture.py)
# and wire the new roughness/metallic texture into IT -- rather than
# replacing it with a bare new material -- so nothing from the earlier
# pipeline stage is lost.
if orig_mat is None:
    raise RuntimeError(f"{mesh_path} has no material to add roughness/metallic to -- run rebake_texture() first")
mesh.materials.clear()
mesh.materials.append(orig_mat)
orig_nt = orig_mat.node_tree
bsdf = next(n for n in orig_nt.nodes if n.type == 'BSDF_PRINCIPLED')
orm_tex_node = orig_nt.nodes.new("ShaderNodeTexImage")
orm_tex_node.image = orm
separate = orig_nt.nodes.new("ShaderNodeSeparateColor")
orig_nt.links.new(orm_tex_node.outputs["Color"], separate.inputs["Color"])
orig_nt.links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
orig_nt.links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])

bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
print(f"=== saved to {out_path} ===")
