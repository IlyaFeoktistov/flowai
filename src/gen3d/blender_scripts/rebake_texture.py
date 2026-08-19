"""Headless Blender texture transfer: bake the source mesh's baseColorTexture AND
its surface detail (as a tangent-space normal map) onto a retopologized target
mesh's new UVs, via Cycles "Selected to Active" bake. No AI/diffusion involved --
pure geometric projection of the existing albedo, plus a geometry-only normal bake
that recovers the bumps/wrinkles Decimate/Remesh threw away -- the standard
high->low-poly game-asset workflow.

Run as:
    blender --background --python rebake_texture.py -- <source.glb> <target.glb> <output.glb> [texture_size]

texture_size defaults to 2048.

Albedo uses bake type=EMIT (source's texture rerouted through an Emission shader) rather
than type=DIFFUSE with pass_filter={'COLOR'} -- the latter reliably baked to all-zero
RGB in testing (only the alpha channel came through), on both apt and official Blender
builds, self-bake and selected-to-active alike. EMIT sidesteps whatever that filter
combination breaks and reproducibly picks up the real texture colors.

Normal map uses bake type=NORMAL, tangent space (glTF's own convention, so the
exporter's default swizzle needs no adjustment) -- pure geometry, doesn't touch
source's material/texture at all: it reads the surface offset between target and
source within cage_extrusion and encodes it as a tangent-space normal, wired into
the target material's Normal input so the glTF exporter picks it up as normalTexture.

AO map uses bake type=AO, same selected-to-active setup as the normal bake -- also
pure geometry, and also recovers crevice occlusion from detail Decimate/Remesh threw
away (a hard shell that ends up smooth still gets the shading cues of its original
bumps). Wired into a "glTF Settings" node group's Occlusion input, the documented
way to get Blender's glTF exporter to emit occlusionTexture -- Principled BSDF has no
Occlusion socket, so there's no more direct route for the exporter to pick this up.

Roughness/metallic have no per-pixel source yet (Hunyuan3D-2GP's texgen only
produces albedo) -- set as flat factors (ROUGHNESS_DEFAULT/METALLIC_DEFAULT below)
via the Principled BSDF's own inputs, which the exporter reads as roughnessFactor/
metallicFactor. No texture needed for a constant -- swap to a baked/estimated
metallicRoughnessTexture later without touching the AO/normal wiring above.
"""
import sys
import bpy

ROUGHNESS_DEFAULT = 0.8
METALLIC_DEFAULT = 0.0

argv = sys.argv[sys.argv.index("--") + 1:]
source_path = argv[0]
target_path = argv[1]
out_path = argv[2]
texture_size = int(argv[3]) if len(argv) > 3 else 2048

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.import_scene.gltf(filepath=source_path)
source_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.context.view_layer.objects.active = source_objs[0]
for o in source_objs:
    o.select_set(True)
if len(source_objs) > 1:
    bpy.ops.object.join()
source = bpy.context.view_layer.objects.active
source.name = "source"

for mat in source.data.materials:
    if mat is None:
        continue
    mat.use_backface_culling = False
    nt = mat.node_tree
    src_tex = next((n for n in nt.nodes if n.type == 'TEX_IMAGE'), None)
    out_node = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if src_tex is not None and out_node is not None:
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(src_tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

bpy.ops.import_scene.gltf(filepath=target_path)
target_objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "source"]
bpy.context.view_layer.objects.active = target_objs[0]
for o in target_objs:
    o.select_set(True)
if len(target_objs) > 1:
    bpy.ops.object.join()
target = bpy.context.view_layer.objects.active
target.name = "target"

print(f"=== source: {len(source.data.polygons)} faces, target: {len(target.data.polygons)} faces ===")

# target has no UVs after Remesh -- give it one to bake into
bpy.ops.object.select_all(action='DESELECT')
target.select_set(True)
bpy.context.view_layer.objects.active = target
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
bpy.ops.object.mode_set(mode='OBJECT')

# bake target image + material wiring (kept unconnected until after the bake --
# linking it into Base Color beforehand triggers a "circular dependency" warning
# and an all-black result)
img = bpy.data.images.new("baked_albedo", width=texture_size, height=texture_size)
mat = bpy.data.materials.new("baked_mat")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
tex_node = nt.nodes.new("ShaderNodeTexImage")
tex_node.image = img
out_node = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
nt.nodes.active = tex_node

target.data.materials.clear()
target.data.materials.append(mat)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.bake.use_selected_to_active = True
bpy.context.scene.render.bake.cage_extrusion = 0.05
bpy.context.scene.render.bake.margin = 4

bpy.ops.object.select_all(action='DESELECT')
source.select_set(True)
target.select_set(True)
bpy.context.view_layer.objects.active = target

print("=== baking albedo (selected-to-active, EMIT) ===")
result = bpy.ops.object.bake(type='EMIT')
print(f"=== albedo bake result: {result} ===")

nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

# Normal map: captures the surface detail Decimate/Remesh discarded, as a
# tangent-space normal map -- pure geometry (offset between source's and
# target's surfaces within cage_extrusion), doesn't need source's material at
# all. Same selected-to-active setup as the albedo bake above (source+target
# still selected, target still active) -- only the active image/bake type change.
normal_img = bpy.data.images.new("baked_normal", width=texture_size, height=texture_size)
normal_img.colorspace_settings.name = 'Non-Color'  # normal maps aren't sRGB data
normal_tex_node = nt.nodes.new("ShaderNodeTexImage")
normal_tex_node.image = normal_img
nt.nodes.active = normal_tex_node
bpy.context.scene.render.bake.normal_space = 'TANGENT'

print("=== baking normal map (selected-to-active, NORMAL/tangent) ===")
result = bpy.ops.object.bake(type='NORMAL')
print(f"=== normal bake result: {result} ===")

normal_map_node = nt.nodes.new("ShaderNodeNormalMap")
nt.links.new(normal_tex_node.outputs["Color"], normal_map_node.inputs["Color"])
nt.links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

# AO map: same selected-to-active geometry bake as the normal map above, recovers
# crevice occlusion from detail Decimate/Remesh discarded. bake.normal_space above
# doesn't apply to AO -- no extra bake-settings needed beyond swapping the active image.
ao_img = bpy.data.images.new("baked_ao", width=texture_size, height=texture_size)
ao_img.colorspace_settings.name = 'Non-Color'  # AO is a scalar multiplier, not sRGB color
ao_tex_node = nt.nodes.new("ShaderNodeTexImage")
ao_tex_node.image = ao_img
nt.nodes.active = ao_tex_node

print("=== baking AO (selected-to-active) ===")
result = bpy.ops.object.bake(type='AO')
print(f"=== AO bake result: {result} ===")

# Principled BSDF has no Occlusion input -- "glTF Settings" is glTF-Blender-IO's own
# documented side-channel node group for exporting occlusionTexture: the exporter scans
# the material tree for a group node backed by a node-group literally named "glTF
# Settings" with an "Occlusion" input socket, independent of the main BSDF chain.
gltf_settings_tree = bpy.data.node_groups.new("glTF Settings", "ShaderNodeTree")
gltf_settings_tree.interface.new_socket(name="Occlusion", in_out='INPUT', socket_type='NodeSocketFloat')
gltf_settings_tree.nodes.new("NodeGroupOutput")
gltf_settings_group = nt.nodes.new("ShaderNodeGroup")
gltf_settings_group.node_tree = gltf_settings_tree
nt.links.new(ao_tex_node.outputs["Color"], gltf_settings_group.inputs["Occlusion"])

# Roughness/metallic: no per-pixel source yet (Hunyuan3D-2GP's texgen is albedo-only),
# so flat factors via the BSDF's own inputs -- exporter reads these as roughnessFactor/
# metallicFactor directly, no texture required for a constant.
bsdf.inputs["Roughness"].default_value = ROUGHNESS_DEFAULT
bsdf.inputs["Metallic"].default_value = METALLIC_DEFAULT

bpy.ops.object.select_all(action='DESELECT')
source.select_set(True)
bpy.ops.object.delete()

bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB")
print(f"=== saved to {out_path} ===")
