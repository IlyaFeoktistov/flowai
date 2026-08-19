"""Headless Blender: parent a mesh to an armature with Automatic Weights (heat
diffusion skinning), as an alternative to UniRig's own skin-prediction model --
no extra VRAM/model needed, just Blender's built-in algorithm.

Run as:
    blender --background --python auto_weight.py -- <skeleton.fbx> <output.glb>

<skeleton.fbx> is expected to contain one ARMATURE object and one MESH object
(e.g. UniRig's generate_skeleton.sh output), both unparented.

NOTE: Blender's glTF exporter reproducibly (but non-deterministically) leaves a
stray helper mesh node (seen as "Icosphere") in the output when exporting a
skinned mesh headlessly. Re-exporting after deleting it in-session doesn't help
-- it comes back. Run strip_glb_extras.py on this script's output as a mandatory
follow-up pass (see gen3d/pipeline.py) instead of fighting the exporter here.
"""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_path = argv[0]
out_path = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=in_path)

armature = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH')

print(f"=== armature: {armature.name} ({len(armature.data.bones)} bones), mesh: {mesh.name} ({len(mesh.data.vertices)} verts) ===")

bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

print(f"=== vertex groups after auto weights: {len(mesh.vertex_groups)} ===")
print(f"=== vg names: {[g.name for g in mesh.vertex_groups]} ===")
print(f"=== mesh modifiers: {[(m.type, m.name) for m in mesh.modifiers]} ===")

# sanity check: how many verts have zero total weight (unassigned to any bone)?
zero_weight = 0
for v in mesh.data.vertices:
    total = 0.0
    for g in v.groups:
        total += g.weight
    if total < 1e-6:
        zero_weight += 1
print(f"=== verts with zero total weight: {zero_weight}/{len(mesh.data.vertices)} ===")

bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
print(f"=== saved to {out_path} ===")
