"""Strip any glTF node that isn't the armature, its bones, or the (single)
skinned character mesh -- a whitelist-based cleanup, since Blender's automatic
weights / exporters have reproducibly (but non-deterministically) left a stray
extra mesh node (seen as "Icosphere") in the output across several runs of the
same pipeline. Edits the glTF JSON chunk directly, no Blender involved.

Usage: python3 strip_glb_extras.py <in.glb> <out.glb>
"""
import json
import struct
import sys

in_path, out_path = sys.argv[1], sys.argv[2]

with open(in_path, "rb") as f:
    magic, version, length = struct.unpack("<4sII", f.read(12))
    assert magic == b"glTF"
    json_len, json_type = struct.unpack("<II", f.read(8))
    json_bytes = f.read(json_len)
    bin_chunk = None
    rest = f.read(8)
    if rest:
        bin_len, bin_type = struct.unpack("<II", rest)
        bin_chunk = f.read(bin_len)

gltf = json.loads(json_bytes)
nodes = gltf["nodes"]


def keep(n):
    name = n.get("name", "")
    return name == "Armature" or name.startswith("bone_") or "mesh" in n


keep_idx = [i for i, n in enumerate(nodes) if keep(n)]
# Among kept mesh nodes, keep only the one with the most vertices (the real
# character) -- any other "mesh" node is the stray object.
mesh_nodes = [i for i in keep_idx if "mesh" in nodes[i]]
if len(mesh_nodes) > 1:
    def vcount(i):
        mesh_idx = nodes[i]["mesh"]
        prim = gltf["meshes"][mesh_idx]["primitives"][0]
        pos_accessor = gltf["accessors"][prim["attributes"]["POSITION"]]
        return pos_accessor["count"]
    best = max(mesh_nodes, key=vcount)
    keep_idx = [i for i in keep_idx if i not in mesh_nodes or i == best]

drop_idx = sorted(set(range(len(nodes))) - set(keep_idx))
if not drop_idx:
    print("nothing to strip, copying through unchanged")
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(fin.read())
    sys.exit(0)

print(f"dropping node(s): {[(i, nodes[i].get('name')) for i in drop_idx]}")

remap = {}
new_nodes = []
for i, n in enumerate(nodes):
    if i in drop_idx:
        continue
    remap[i] = len(new_nodes)
    new_nodes.append(n)

for n in new_nodes:
    if "children" in n:
        n["children"] = [remap[c] for c in n["children"] if c in remap]
for scene in gltf.get("scenes", []):
    if "nodes" in scene:
        scene["nodes"] = [remap[i] for i in scene["nodes"] if i in remap]
for skin in gltf.get("skins", []):
    if "joints" in skin:
        skin["joints"] = [remap[i] for i in skin["joints"] if i in remap]
    if "skeleton" in skin and skin["skeleton"] in remap:
        skin["skeleton"] = remap[skin["skeleton"]]

gltf["nodes"] = new_nodes

new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
pad = (-len(new_json)) % 4
new_json += b" " * pad

with open(out_path, "wb") as f:
    total_len = 12 + 8 + len(new_json)
    if bin_chunk is not None:
        total_len += 8 + len(bin_chunk)
    f.write(struct.pack("<4sII", b"glTF", 2, total_len))
    f.write(struct.pack("<II", len(new_json), 0x4E4F534A))
    f.write(new_json)
    if bin_chunk is not None:
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)

print(f"saved {out_path}")
