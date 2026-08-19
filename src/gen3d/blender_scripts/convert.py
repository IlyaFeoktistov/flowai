"""Headless Blender format conversion (.glb/.gltf/.fbx in any direction) --
e.g. Animato only accepts .fbx/.gltf/.obj (not .glb, see gen3d/animato_client.py).

Run as: blender --background --python convert.py -- <in> <out>
"""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
in_path, out_path = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)

if in_path.lower().endswith((".glb", ".gltf")):
    bpy.ops.import_scene.gltf(filepath=in_path)
elif in_path.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=in_path)
else:
    raise ValueError(f"unsupported input extension: {in_path}")

if out_path.lower().endswith(".glb"):
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
elif out_path.lower().endswith(".gltf"):
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLTF_SEPARATE")
elif out_path.lower().endswith(".fbx"):
    bpy.ops.export_scene.fbx(filepath=out_path, add_leaf_bones=False)
else:
    raise ValueError(f"unsupported output extension: {out_path}")

print(f"=== converted {in_path} -> {out_path} ===")
