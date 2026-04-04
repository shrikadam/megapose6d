import os
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# MegaPose Environment setup
try:
    import megapose_server

    variables_file = "./megapose_server/megapose_variables_final.json"
    with open(variables_file, "r") as f:
        json_vars = json.load(f)
        os.environ["MEGAPOSE_DIR"] = json_vars["megapose_dir"]
        os.environ["MEGAPOSE_DATA_DIR"] = json_vars["megapose_data_dir"]
except:
    pass

if "HOME" not in os.environ:
    os.environ["HOME"] = "."

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d


def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    for object_dir in meshes_dir.iterdir():
        if not object_dir.is_dir():
            continue
        label = object_dir.name
        mesh_path = next(
            (fn for fn in object_dir.glob("*") if fn.suffix in {".obj", ".ply", ".glb"}), None
        )
        assert mesh_path, f"Could not find the mesh for {label}"
        # Set to mm based on our previous debugging
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units="mm"))
    return RigidObjectDataset(rigid_objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaPose Template Debugger")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--meshes-directory", type=str, required=True)
    parser.add_argument("--output", type=str, default="debug_templates.png")
    args = parser.parse_args()

    print("[DEBUG] Initializing Renderer...", flush=True)
    dataset = make_object_dataset(Path(args.meshes_directory).absolute())
    renderer = Panda3dSceneRenderer(dataset)

    # Setup a standard virtual camera
    camera_data = CameraData()
    camera_data.K = np.array([[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]])
    camera_data.resolution = (480, 640)
    camera_data.TWC = Transform(np.eye(4))

    # MegaPose usually uses a flat ambient light for template matching
    light_datas = [Panda3dLightData(light_type="ambient", color=((1.0, 1.0, 1.0, 1)))]

    print("[DEBUG] Generating 16 viewpoints...", flush=True)
    rendered_images = []

    # We will orbit the camera around the object (Pitch and Yaw)
    distance = 0.3  # 30cm away

    for pitch in [-45, 0, 45, 90]:
        row_images = []
        for yaw in [0, 90, 180, 270]:
            # Calculate rotation and translation
            rot = R.from_euler("YX", [yaw, pitch], degrees=True).as_matrix()
            tvec = np.array([0.0, 0.0, distance])

            # Create a 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = tvec

            object_datas = [ObjectData(label=args.label, TWO=Transform(pose))]
            cam_data, obj_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

            # Render the template
            renderings = renderer.render_scene(
                obj_datas,
                [cam_data],
                light_datas,
                render_depth=False,
                render_binary_mask=False,
                render_normals=False,
                copy_arrays=True,
            )[0]

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(renderings.rgb, cv2.COLOR_RGB2BGR)

            # Add text so you know what angle you are looking at
            cv2.putText(
                img_bgr,
                f"P:{pitch} Y:{yaw}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            row_images.append(img_bgr)

        # Stitch the row together horizontally
        rendered_images.append(cv2.hconcat(row_images))

    # Stitch all rows together vertically
    final_grid = cv2.vconcat(rendered_images)

    cv2.imwrite(args.output, final_grid)
    print(f"[SUCCESS] Saved debug grid to {args.output}")
