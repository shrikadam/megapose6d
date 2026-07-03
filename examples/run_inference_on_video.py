import sys
import time
import numpy as np
import argparse
import json
import cv2
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

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


def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    print("[DEBUG] Initializing Object Dataset from meshes...", flush=True)
    rigid_objects = []
    object_dirs = meshes_dir.iterdir()
    for object_dir in object_dirs:
        if not object_dir.is_dir():
            continue
        label = object_dir.name
        mesh_path = next(
            (fn for fn in object_dir.glob("*") if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}),
            None,
        )
        assert mesh_path, f"Could not find the mesh for {label}"

        # NOTE: Using "mm" to ensure correct scale as discovered previously
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units="mm"))
    return RigidObjectDataset(rigid_objects)


def smooth_pose(
    prev_pose: Optional[List[List[float]]], curr_pose: List[List[float]], alpha: float = 0.4
) -> List[List[float]]:
    """Applies EMA to translation and Slerp to rotation to safely smooth 3D poses"""
    if prev_pose is None:
        return curr_pose

    prev_np = np.array(prev_pose)
    curr_np = np.array(curr_pose)

    # 1. EMA for Translation (XYZ)
    t_prev = prev_np[:3, 3]
    t_curr = curr_np[:3, 3]
    t_smoothed = (1.0 - alpha) * t_prev + alpha * t_curr

    # 2. Slerp for Rotation (Quaternions)
    r_prev = R.from_matrix(prev_np[:3, :3])
    r_curr = R.from_matrix(curr_np[:3, :3])

    # Create a Slerp interpolator between time=0 (prev) and time=1 (curr)
    slerp = Slerp([0, 1], R.concatenate([r_prev, r_curr]))
    r_smoothed = slerp([alpha])[0].as_matrix()

    # 3. Reconstruct the 4x4 Transformation Matrix
    smoothed_pose = np.eye(4)
    smoothed_pose[:3, :3] = r_smoothed
    smoothed_pose[:3, 3] = t_smoothed

    return smoothed_pose.tolist()


class VideoPoseEstimator:
    def __init__(self, model_name: str, mesh_dir: Path, num_workers: int = 1):
        print(f"\n[DEBUG] --- Initializing MegaPose Estimator ---", flush=True)
        self.object_dataset = make_object_dataset(mesh_dir)
        print(f"[DEBUG] Loading named model '{model_name}' into GPU...", flush=True)

        # VRAM OPTIMIZATION 1: Restrict workers
        self.model_info, self.model = (
            NAMED_MODELS[model_name],
            load_named_model(model_name, self.object_dataset, n_workers=num_workers).cuda(),
        )
        self.model.eval()
        self.model = self.model.half()

        self.model.bsz_images = 32
        self.model.bsz_objects = 4
        self.model.load_SO3_grid(
            72
        )

        torch.backends.cudnn.benchmark = True
        print(f"[DEBUG] --- Estimator Initialization Complete ---\n", flush=True)

    def estimate_pose(
        self,
        img_np: np.ndarray,
        label: str,
        bbox: List[float],
        K: np.ndarray,
        previous_pose: Optional[List[List[float]]] = None,
        n_hypotheses: int = 2,  
        n_iterations: int = 5   
    ):
        # 1. Prepare Camera & Observation Tensor natively
        h, w = img_np.shape[:2]
        c = CameraData()
        c.K, c.resolution, c.z_near, c.z_far = K, (h, w), 0.001, 100000
        
        observation = ObservationTensor.from_numpy(img_np, None, c.K).cuda()

        # 2. Prepare Bounding Box Detections
        obj_data = ObjectData(label)
        obj_data.bbox_modal = np.array(bbox, dtype=np.float32)
        detections = make_detections_from_object_data([obj_data]).cuda()

        # 3. Pull default params, inject dynamic gearbox params
        inference_params = self.model_info["inference_parameters"].copy()
        inference_params["n_pose_hypotheses"] = n_hypotheses
        inference_params["n_refiner_iterations"] = n_iterations

        # 4. FAST TRACKING LOGIC: Bypass Coarse network if we have a previous pose
        coarse_estimates = None
        if previous_pose is not None:
            cTos_np = np.array([previous_pose]).reshape(-1, 4, 4)
            tensor = torch.from_numpy(cTos_np).float().cuda()
            infos = pd.DataFrame.from_dict(
                {"label": [label], "batch_im_id": [0], "instance_id": [0]}
            )
            coarse_estimates = PoseEstimatesType(infos, poses=tensor)

        # 5. Run the Refiner
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output, _ = self.model.run_inference_pipeline(
                observation,
                detections=detections,
                coarse_estimates=coarse_estimates,
                **inference_params,
            )

        # 6. Extract results
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
    return blended_img, mask


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

    parser = argparse.ArgumentParser(description="MegaPose Video Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Object label (must match folder in meshes directory)",
    )
    parser.add_argument("--camera-json", type=str, required=True, help="Path to camera_data.json")
    parser.add_argument("--meshes-directory", type=str, default="./data/models")
    parser.add_argument("--output-video", type=str, default="tracked_output.mp4")
    args = parser.parse_args()

    # Load Camera Intrinsic Matrix from JSON
    print(f"[DEBUG] Loading camera data from: {args.camera_json}", flush=True)
    with open(args.camera_json, "r") as f:
        cam_data = json.load(f)
        K = np.array(cam_data["K"], dtype=np.float32)

    # Initialize Deep Learning Model
    estimator = VideoPoseEstimator(
        "megapose-1.0-RGB-multi-hypothesis", Path(args.meshes_directory).absolute(), num_workers=1
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

    # Read First Frame for GUI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame.")

    # 1. BBox Selection GUI
    print("\n[GUI] Select the object, then press SPACE or ENTER.", flush=True)
    bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = bbox
    if w == 0 or h == 0:
        raise ValueError("Bounding box selection cancelled.")

    init_tracker_bbox = (int(x), int(y), int(w), int(h))
    current_bbox = [x, y, x + w, y + h]

    # Reset video capture to start from Frame 0 again
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2. Initialize Panda3D Renderer (MUST be after GUI is destroyed to prevent X11 deadlock)
    print("[DEBUG] Initializing Panda3D CAD Renderer safely...", flush=True)
    renderer = Panda3dSceneRenderer(estimator.object_dataset)

    # 3. Setup Video Writer & 2D Tracker
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    frame_idx = 0
    last_known_pose = None
    smoothed_pose = None
    score_history = deque(maxlen=5)

    print(f"\n[DEBUG] Starting Main Video Loop ({total_frames} frames)...", flush=True)
    pbar = tqdm(total=total_frames)

    prev_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        n_hypotheses = 2
        n_iterations = 5
        
        # GEARBOX TRIGGER: If we are actively tracking, we already solved 
        # global orientation symmetry, so we drop hypotheses to 1 immediately.
        if last_known_pose is not None:
            n_hypotheses = 1
            
            # Check if our confidence history buffer is full and perfectly stable
            if len(score_history) == score_history.maxlen:
                score_variance = np.var(score_history)
                score_mean = np.mean(score_history)
                
                # If the tracking is highly confident and rock-solid steady, 
                # drop refiner iterations to 2 to maximize frame rate.
                if score_mean >= 0.50 and score_variance <= 0.02:
                    n_iterations = 2

        # Run inference with our dynamically assigned values
        result = estimator.estimate_pose(
            frame_rgb, 
            args.label, 
            current_bbox, 
            K, 
            previous_pose=last_known_pose,
            n_hypotheses=n_hypotheses,
            n_iterations=n_iterations
        )
        
        raw_pose = result["cTo"]
        current_score = result["score"]
        
        # --- CONFIDENCE FILTERING & RESET LOGIC ---
        score_history.append(current_score)
        
        if len(score_history) == score_history.maxlen:
            score_variance = np.var(score_history)
            score_mean = np.mean(score_history)
            
            # TRIGGER: If tracking fails or oscillates wildly, drop everything
            if score_mean < 0.40 or score_variance > 0.05:
                print(f"\n[WARNING] Instability detected (Mean: {score_mean:.2f}, Var: {score_variance:.3f}). Forcing Coarse Reset!")
                last_known_pose = None 
                smoothed_pose = None
                score_history.clear() 
            else:
                last_known_pose = raw_pose
        else:
            last_known_pose = raw_pose

        # Apply geometric smoothing (only if we didn't just reset)
        if last_known_pose is not None:
            if smoothed_pose is None:
                smoothed_pose = last_known_pose
            else:
                smoothed_pose = smooth_pose(smoothed_pose, last_known_pose, alpha=0.4)
        else:
            smoothed_pose = raw_pose

        # --- VISUALIZATION PIPELINE (Using Smoothed Pose) ---
        # Catch the mask returned by the renderer
        overlay_rgb, mask = render_cad_overlay(frame_rgb, args.label, smoothed_pose, K, renderer)
        final_frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        final_frame_bgr = draw_3d_axes(final_frame_bgr, smoothed_pose, K)

        # --- CLOSED-LOOP BOUNDING BOX ---
        # Find all the X and Y pixel coordinates where the Panda3D mask exists
        y_indices, x_indices = np.where(mask)
        
        if len(x_indices) > 0:
            xmin, xmax = np.min(x_indices), np.max(x_indices)
            ymin, ymax = np.min(y_indices), np.max(y_indices)
            
            # Add a small 10-pixel padding so MegaPose has context for the next frame
            pad = 10
            current_bbox = [
                max(0, xmin - pad), 
                max(0, ymin - pad), 
                min(width, xmax + pad), 
                min(height, ymax + pad)
            ]
            
            # Draw the mathematically perfect bounding box
            cv2.rectangle(
                final_frame_bgr, (int(current_bbox[0]), int(current_bbox[1])), 
                (int(current_bbox[2]), int(current_bbox[3])), (0, 0, 255), 2
            )
        else:
            # If the mask is empty, tracking failed completely. 
            last_known_pose = None
        
        # --- XYZWPR POSE ---
        final_frame_bgr = draw_pose_text(final_frame_bgr, smoothed_pose, result["score"])

        # --- FPS CALCULATION & HUD ---
        curr_time = time.time()
        fps_val = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Draw in the top right corner (assuming 'width' is defined from your video cap)
        cv2.putText(
            final_frame_bgr, 
            f"FPS: {fps_val:.1f}", 
            (width - 160, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), # Yellow text
            2
        )

        # Save & Iterate
        out_vid.write(final_frame_bgr)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_vid.release()

    print(f"\n[SUCCESS] Saved tracked video to {args.output_video}")

    # --- CLEAN SHUTDOWN ---
    print("\n[DEBUG] Shutting down background rendering processes...", flush=True)

    # 1. Deleting these objects triggers their internal __del__ methods,
    # which safely terminates the PyTorch multiprocessing queues.
    del estimator
    del renderer

    # 2. Force the main process to exit, bypassing any lingering zombie threads
    sys.exit(0)
