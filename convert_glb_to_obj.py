import trimesh
import os

# Load the GLB file
mesh = trimesh.load("/home/shri/GitHub/megapose6d/local_data/examples/nic/meshes/nic.glb")

# Create the directory if it doesn't exist
output_dir = "/home/shri/GitHub/megapose6d/local_data/examples/nic/meshes"
os.makedirs(output_dir, exist_ok=True)

# Export as OBJ (trimesh handles texture extraction automatically)
mesh.export(os.path.join(output_dir, "nic.obj"))
