"""Headless Blender retopology for raw generator meshes. Run as:

    blender --background --python retopo.py -- <input.glb> <output.glb> [target_faces] [voxel_size]

target_faces defaults to 15000, voxel_size defaults to 0.01.

Raw Hunyuan3D output is a marching-cubes extraction full of disconnected shell
fragments/noise (tens of thousands of tiny islands) -- Decimate alone gets stuck
respecting that broken topology and plateaus well above the target. Voxel Remesh
first rebuilds a single clean watertight surface, then Decimate reduces that to
the target polycount.
"""
import sys
import bpy
import bmesh

argv = sys.argv[sys.argv.index("--") + 1:]
in_path = argv[0]
out_path = argv[1]
target_faces = int(argv[2]) if len(argv) > 2 else 15000
voxel_size = float(argv[3]) if len(argv) > 3 else 0.01


def _clean_mesh(obj, merge_threshold, min_island_faces):
    """Merge near-duplicate vertices (Blender's own "Merge by Distance") and
    drop whatever's still disconnected below min_island_faces. Both the raw
    generator mesh and Voxel Remesh's output are marching-cubes reconstructions
    full of noise islands (see module docstring) -- left alone, Smart UV Project
    and the Cycles bake in rebake_texture.py treat each fragment as its own
    island, which shows up as torn/black patches in the final texture."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
    bpy.ops.object.mode_set(mode='OBJECT')

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    visited = set()
    to_delete = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        stack = [seed]
        island = []
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            island.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        if len(island) < min_island_faces:
            to_delete.extend(island)
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _keep_largest_island(obj) -> int:
    """Hard guarantee: delete every connected component except the single
    largest one. Voxel Remesh's whole point is a single watertight surface --
    _clean_mesh's distance-based merge closes near-touching gaps, but Hunyuan3D
    shape generation is a diffusion process (non-deterministic per run) and a
    worse roll can leave genuinely-separated fragments (confirmed on a real
    generation: some runs remesh to 1 clean island, others still had 613
    components afterwards with no threshold that safely bridges them without
    risking eating into real geometry). Anything not connected to the main
    surface at this point in the pipeline can't be salvaged -- it's noise or a
    snapped-off wisp either way, and either is a torn/black patch waiting to
    happen once Smart UV Project and the Cycles bake treat it as its own
    island. Returns the number of faces removed."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    visited = set()
    islands = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        stack = [seed]
        island = []
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            island.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        islands.append(island)
    islands.sort(key=len, reverse=True)
    to_delete = [f for island in islands[1:] for f in island]
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(to_delete)


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
src_faces = len(obj.data.polygons)
print(f"=== source: {len(obj.data.vertices)} verts / {src_faces} faces ===")

# Clean the raw mesh BEFORE remeshing -- per 3dtodo.md's own island survey on
# this exact generator, ~100 faces is where real geometry (body/limbs/props)
# separates from marching-cubes noise (tens of thousands of sub-100-face
# shell fragments). Merge threshold is relative to the mesh's own size since
# raw output scale varies per generation.
avg_dim = sum(obj.dimensions) / 3
print("=== cleaning raw mesh: merge near-duplicate verts, drop noise islands ===")
_clean_mesh(obj, merge_threshold=avg_dim * 0.0005, min_island_faces=100)
cleaned_faces = len(obj.data.polygons)
print(f"=== after cleanup: {len(obj.data.vertices)} verts / {cleaned_faces} faces "
      f"(dropped {src_faces - cleaned_faces} noise faces) ===")

print(f"=== voxel remesh: voxel_size={voxel_size} ===")
remesh = obj.modifiers.new(name="Remesh", type="REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = voxel_size
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=remesh.name)

remeshed_faces = len(obj.data.polygons)
print(f"=== after remesh: {len(obj.data.vertices)} verts / {remeshed_faces} faces ===")

# Voxel Remesh is supposed to produce a single watertight surface, but thin
# protrusions (hair curls, beard strands, ...) can still come out as
# near-touching-but-disconnected fragments -- confirmed on a real generation:
# 643 connected components post-remesh, 480 of them under 10 verts. Those
# fragments are exactly where the bake later fails (rays miss the source mesh
# for anything not welded into the main surface), so clean up again here,
# before Decimate/UV-unwrap/bake ever see the mesh.
print("=== post-remesh cleanup: merge split verts, drop residual noise islands ===")
_clean_mesh(obj, merge_threshold=voxel_size * 0.5, min_island_faces=max(8, int(remeshed_faces * 0.001)))
print(f"=== after post-remesh cleanup: {len(obj.data.vertices)} verts / {len(obj.data.polygons)} faces ===")

dropped = _keep_largest_island(obj)
print(f"=== keep-largest-island: dropped {dropped} faces from smaller islands, "
      f"{len(obj.data.vertices)} verts / {len(obj.data.polygons)} faces left ===")

# Decimate's ratio param is approximate and overshoots badly on huge single-shot
# reductions (e.g. 679k -> 15k target landed at 36.7k faces in one pass). Iterate,
# recomputing the ratio against the actual current count each time, until within
# tolerance of the target or further passes stop making progress.
for i in range(6):
    cur_faces = len(obj.data.polygons)
    if cur_faces <= target_faces * 1.1:
        break
    ratio = target_faces / cur_faces
    print(f"=== pass {i+1}: {cur_faces} faces, ratio {ratio:.4f} ===")
    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    new_faces = len(obj.data.polygons)
    if new_faces >= cur_faces:
        print(f"=== pass {i+1}: no progress ({new_faces} >= {cur_faces}), stopping ===")
        break

dst_faces = len(obj.data.polygons)
print(f"=== result: {len(obj.data.vertices)} verts / {dst_faces} faces ===")

# Decimate/Remesh leave the mesh flat-shaded (one normal per face) -- exporting
# that to glTF forces a vertex split at every face edge (per-vertex normals must
# match), so adjacent triangles end up sharing a position but not an actual vertex.
# Weld coincident verts back together, then shade-smooth so the exporter keeps
# them merged.
#
# threshold is relative to mesh size (matches _clean_mesh's own convention),
# NOT the fixed 1e-5 this used to be -- confirmed on a real multi-view
# generation (Hunyuan3D-2mv): Decimate's collapse can reposition a vertex to a
# point a hair's width away from where a neighboring collapse left its own
# vertex, instead of exactly on top of it -- topologically two separate
# points, so the mesh silently fragments (one real generation came out at
# 251 disconnected islands post-decimate, despite _keep_largest_island()
# having forced it down to exactly 1 island right before the Decimate loop
# started -- confirmed by re-running retopo with per-stage island counts).
# That's small enough that flat 1e-5 never bridges it, but a threshold this
# small relative to mesh scale does, and for free -- tested against that
# exact 251-island mesh: even avg_dim*0.0001 reconnects it back to 1 island
# with ZERO faces lost; avg_dim*0.0005 keeps the same safety margin the
# pre-remesh cleanup already uses elsewhere in this file.
weld_threshold = (sum(obj.dimensions) / 3) * 0.0005
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=weld_threshold)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.shade_smooth()

welded_verts = len(obj.data.vertices)
print(f"=== after weld+smooth: {welded_verts} verts / {len(obj.data.polygons)} faces ===")

bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
print(f"=== saved to {out_path} ===")
