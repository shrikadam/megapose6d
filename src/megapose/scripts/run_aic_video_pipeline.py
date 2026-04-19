import numpy as np
import argparse
import json
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from ultralytics import YOLO

# MegaPose Core
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model

# MegaPose Rendering (For Overlay)
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d

# --- MAPPINGS ---
TARGET_MAP = {"nic": "NIC_Target", "sc": "SC_Target"}


def make_object_dataset(meshes_dir: Path, target_label: str) -> RigidObjectDataset:
    print("[DEBUG] Initializing Object Dataset from meshes...", flush=True)
    rigid_objects = []

    # Only load the specific target mesh to save memory
    target_dir = meshes_dir / target_label
    if not target_dir.is_dir():
        raise ValueError(f"Could not find directory for target '{target_label}' in {meshes_dir}")

    mesh_path = next(
        (fn for fn in target_dir.glob("*") if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}),
        None,
    )
    assert mesh_path, f"Could not find the mesh for {target_label}"

    rigid_objects.append(RigidObject(label=target_label, mesh_path=mesh_path, mesh_units="mm"))
    return RigidObjectDataset(rigid_objects)


def smooth_pose(
    prev_pose: Optional[List[List[float]]], curr_pose: List[List[float]], alpha: float = 0.4
) -> List[List[float]]:
    if prev_pose is None:
        return curr_pose

    prev_np = np.array(prev_pose)
    curr_np = np.array(curr_pose)

    t_prev = prev_np[:3, 3]
    t_curr = curr_np[:3, 3]
    t_smoothed = (1.0 - alpha) * t_prev + alpha * t_curr

    r_prev = R.from_matrix(prev_np[:3, :3])
    r_curr = R.from_matrix(curr_np[:3, :3])

    slerp = Slerp([0, 1], R.concatenate([r_prev, r_curr]))
    r_smoothed = slerp([alpha])[0].as_matrix()

    smoothed_pose = np.eye(4)
    smoothed_pose[:3, :3] = r_smoothed
    smoothed_pose[:3, 3] = t_smoothed

    return smoothed_pose.tolist()


class VideoPoseEstimator:
    def __init__(self, model_name: str, mesh_dir: Path, target_label: str, num_workers: int = 1):
        print(f"\n[DEBUG] --- Initializing MegaPose Estimator ---", flush=True)
        self.object_dataset = make_object_dataset(mesh_dir, target_label)
        print(f"[DEBUG] Loading MegaPose model '{model_name}' into GPU...", flush=True)

        self.model_info, self.model = (
            NAMED_MODELS[model_name],
            load_named_model(model_name, self.object_dataset, n_workers=num_workers).cuda(),
        )
        self.model.eval()

        # VRAM OPTIMIZATION: Slash batch size and grid
        self.model.bsz_images = 16
        self.model.load_SO3_grid(512)  # Bumped slightly for better recovery on fast movements

        torch.backends.cudnn.benchmark = True
        print(f"[DEBUG] --- MegaPose Initialization Complete ---\n", flush=True)

    def estimate_pose(
        self,
        img_np: np.ndarray,
        label: str,
        bbox: List[float],
        K: np.ndarray,
        previous_pose: Optional[List[List[float]]] = None,
    ):
        h, w = img_np.shape[:2]
        c = CameraData()
        c.K, c.resolution, c.z_near, c.z_far = K, (h, w), 0.001, 100000

        observation = ObservationTensor.from_numpy(img_np, None, c.K).cuda()

        obj_data = ObjectData(label)
        obj_data.bbox_modal = np.array(bbox, dtype=np.float32)
        detections = make_detections_from_object_data([obj_data]).cuda()

        coarse_estimates = None
        if previous_pose is not None:
            cTos_np = np.array([previous_pose]).reshape(-1, 4, 4)
            tensor = torch.from_numpy(cTos_np).float().cuda()
            infos = pd.DataFrame.from_dict(
                {"label": [label], "batch_im_id": [0], "instance_id": [0]}
            )
            coarse_estimates = PoseEstimatesType(infos, poses=tensor)

        inference_params = self.model_info["inference_parameters"].copy()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, _ = self.model.run_inference_pipeline(
                observation,
                detections=detections,
                coarse_estimates=coarse_estimates,
                **inference_params,
            )

        pose = output.poses[0].cpu().numpy().reshape(4, 4).tolist()
        score = float(output.infos["pose_score"].iloc[0])

        return {"cTo": pose, "score": score}


# --- VISUALIZATION HELPERS ---
def render_cad_overlay(
    frame_rgb: np.ndarray,
    label: str,
    pose: List[List[float]],
    K: np.ndarray,
    renderer: Panda3dSceneRenderer,
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    camera_data = CameraData()
    camera_data.K, camera_data.resolution, camera_data.TWC = K, (h, w), Transform(np.eye(4))

    object_datas = [ObjectData(label=label, TWO=Transform(np.array(pose)))]
    cam_data, obj_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [Panda3dLightData(light_type="ambient", color=((0.8, 0.8, 0.8, 1)))]

    renderings = renderer.render_scene(
        obj_datas,
        [cam_data],
        light_datas,
        render_depth=True,
        render_binary_mask=True,
        render_normals=False,
        copy_arrays=True,
    )[0]

    cad_rgb = renderings.rgb
    mask = renderings.binary_mask > 0
    blended_img = frame_rgb.copy()
    alpha = 0.75
    blended_img[mask] = (frame_rgb[mask] * (1 - alpha) + cad_rgb[mask] * alpha).astype(np.uint8)
    return blended_img


def draw_pose_text(frame_bgr: np.ndarray, pose: List[List[float]], score: float) -> np.ndarray:
    pose_np = np.array(pose)
    x, y, z = pose_np[0, 3], pose_np[1, 3], pose_np[2, 3]
    euler = R.from_matrix(pose_np[:3, :3]).as_euler("xyz", degrees=True)

    text_lines = [
        f"Score: {score:.2f}",
        f"X: {x*1000:.1f} mm",
        f"Y: {y*1000:.1f} mm",
        f"Z: {z*1000:.1f} mm",
        f"R: {euler[0]:.1f} deg",
        f"P: {euler[1]:.1f} deg",
        f"Y: {euler[2]:.1f} deg",
    ]

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (200, 240), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)

    for i, line in enumerate(text_lines):
        cv2.putText(
            frame_bgr, line, (20, 40 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return frame_bgr


def draw_3d_axes(frame_bgr: np.ndarray, pose: List[List[float]], K: np.ndarray) -> np.ndarray:
    pose_np = np.array(pose)
    rvec, _ = cv2.Rodrigues(pose_np[:3, :3])
    tvec = pose_np[:3, 3]
    cv2.drawFrameAxes(frame_bgr, K, np.zeros((4, 1)), rvec, tvec, 0.03, 3)
    return frame_bgr


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="MegaPose + YOLO Autonomous Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--target", type=str, choices=["nic", "sc"], required=True, help="Object target to track"
    )
    parser.add_argument(
        "--yolo-weights", type=str, required=True, help="Path to your trained YOLO11n .pt file"
    )
    parser.add_argument("--camera-json", type=str, required=True, help="Path to camera_data.json")
    parser.add_argument("--meshes-directory", type=str, default="./data/models")
    parser.add_argument("--output-video", type=str, default="tracked_output.mp4")
    args = parser.parse_args()

    target_yolo_class = TARGET_MAP[args.target]

    # Load Camera Intrinsic Matrix
    print(f"[DEBUG] Loading camera data from: {args.camera_json}", flush=True)
    with open(args.camera_json, "r") as f:
        cam_data = json.load(f)
        K = np.array(cam_data["K"], dtype=np.float32)

    # Initialize YOLO Model
    print(f"[DEBUG] Loading YOLOv11 model from: {args.yolo_weights}", flush=True)
    yolo_model = YOLO(args.yolo_weights)

    # Initialize MegaPose Model
    estimator = VideoPoseEstimator(
        "megapose-1.0-RGB-multi-hypothesis",
        Path(args.meshes_directory).absolute(),
        target_label=args.target,
        num_workers=1,
    )

    # Open Video
    print(f"[DEBUG] Opening video file: {args.video}", flush=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Panda3D Renderer
    print("[DEBUG] Initializing Panda3D CAD Renderer...", flush=True)
    renderer = Panda3dSceneRenderer(estimator.object_dataset)

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    current_bbox = None
    last_known_pose = None
    smoothed_pose = None
    missed_frames = 0

    print(f"\n[DEBUG] Starting Autonomous Video Loop ({total_frames} frames)...", flush=True)
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- 1. YOLO DETECTION ---
        results = yolo_model(frame_bgr, conf=0.8, verbose=False)[0]

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = yolo_model.names[cls_id]

            if cls_name == target_yolo_class and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy().tolist()

        # --- 2. TRACKING LOGIC ---
        if best_box is not None:
            current_bbox = best_box
            missed_frames = 0
        else:
            missed_frames += 1
            if missed_frames > 5:
                # If target is lost for too long, reset the pose so MegaPose doesn't hallucinate
                last_known_pose = None
                smoothed_pose = None
                current_bbox = None

        # --- 3. MEGAPOSE INFERENCE & VISUALIZATION ---
        if current_bbox is not None:
            result = estimator.estimate_pose(
                frame_rgb, args.target, current_bbox, K, previous_pose=last_known_pose
            )
            last_known_pose = result["cTo"]

            if smoothed_pose is None:
                smoothed_pose = last_known_pose
            else:
                smoothed_pose = smooth_pose(smoothed_pose, last_known_pose, alpha=0.4)

            # Render Overlays
            overlay_rgb = render_cad_overlay(frame_rgb, args.target, smoothed_pose, K, renderer)
            final_frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            final_frame_bgr = draw_3d_axes(final_frame_bgr, smoothed_pose, K)

            xmin, ymin, xmax, ymax = current_bbox
            cv2.rectangle(
                final_frame_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2
            )
            cv2.putText(
                final_frame_bgr,
                f"YOLO: {target_yolo_class} ({best_conf:.2f})",
                (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            final_frame_bgr = draw_pose_text(final_frame_bgr, smoothed_pose, result["score"])
        else:
            # Target is lost; write raw frame with warning text
            final_frame_bgr = frame_bgr.copy()
            cv2.putText(
                final_frame_bgr,
                f"Searching for {target_yolo_class}...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # Save & Iterate
        out_vid.write(final_frame_bgr)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_vid.release()
    print(f"\n[SUCCESS] Saved fully autonomous tracked video to {args.output_video}")
