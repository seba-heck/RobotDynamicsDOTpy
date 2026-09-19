import trimesh
from pathlib import Path

mesh_dir = Path("model/IRB120/meshes")

for path in sorted(mesh_dir.glob("*.stl")):
    mesh = trimesh.load(path, force="mesh")

    print(f"\n{path.name}")
    print("  vertices:", len(mesh.vertices))
    print("  faces:", len(mesh.faces))
    print("  bounds:" mesh.bounds)
    print("  center:", mesh.centroid)
    print("  extents:", mesh.extents)


# Converting the STL meshes to center at origin

# origins = {
#     "base":  (0,   0,   0),
#     "link1": (0,   0, 145),
#     "link2": (0,   0, 290),
#     "link3": (0,   0, 560),
#     "link4": (134, 0, 630),
#     "link5": (302, 0, 630),
#     "link6": (374, 0, 630),
# }

# for name, origin in origins.items():

#     mesh = trimesh.load_mesh(mesh_dir / f"{name}.stl")

#     print(f"\n{name}")
#     print("  before:")
#     print(mesh.bounds)

#     mesh.apply_translation(-np.array(origin))

#     print("  after:")
#     print(mesh.bounds)

#     mesh.export(out_dir / f"{name}.stl")

