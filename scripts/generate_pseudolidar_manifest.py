from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import depth_pro
import numpy as np
import open3d as o3d
from PIL import Image
from pyquaternion import Quaternion
import torch
from nuscenes.nuscenes import NuScenes


CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

ML_DEPTH_PRO_ROOT = Path("/home/clara/ml-depth-pro")
DEPTH_PRO_CHECKPOINT = ML_DEPTH_PRO_ROOT / "checkpoints" / "depth_pro.pt"

SCALE_BY_CHANNEL = {
    "CAM_FRONT": 1.00,
    "CAM_FRONT_LEFT": 0.99,
    "CAM_FRONT_RIGHT": 0.99,
    "CAM_BACK": 0.97,
    "CAM_BACK_LEFT": 0.98,
    "CAM_BACK_RIGHT": 0.98,
}


def transform_from_translation_rotation(translation: List[float], rotation: List[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def depth_to_colored_points_pinhole(
    depth: np.ndarray,
    image_np: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float,
    max_depth: float,
    crop_top_ratio: float,
    crop_bottom_ratio: float,
    crop_side_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    z = depth.copy()
    z[(z < min_depth) | (z > max_depth)] = np.nan
    z[v < h * crop_top_ratio] = np.nan
    z[v > h * (1.0 - crop_bottom_ratio)] = np.nan
    z[u < w * crop_side_ratio] = np.nan
    z[u > w * (1.0 - crop_side_ratio)] = np.nan

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid = np.isfinite(z)
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = image_np.reshape(-1, 3)[valid.reshape(-1)].astype(np.float32) / 255.0
    return points, colors


def pseudo_lidar_from_dense_cloud(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    n_scan: int = 64,
    horizon_scan: int = 2048,
    fov_up_deg: float = 12.0,
    fov_down_deg: float = -30.0,
    range_min: float = 2.0,
    range_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    ranges = np.linalg.norm(points_xyz, axis=1)
    mask = (ranges >= range_min) & (ranges <= range_max)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points_xyz = points_xyz[mask]
    points_rgb = points_rgb[mask]
    ranges = ranges[mask]
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / np.maximum(ranges, 1e-6), -1.0, 1.0))

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_x = np.floor(proj_x * horizon_scan).astype(np.int32)
    proj_x = np.clip(proj_x, 0, horizon_scan - 1)

    proj_y = 1.0 - (pitch + abs(fov_down)) / fov
    proj_y = np.floor(proj_y * n_scan).astype(np.int32)
    proj_y = np.clip(proj_y, 0, n_scan - 1)

    order = np.argsort(ranges)[::-1]
    proj_x = proj_x[order]
    proj_y = proj_y[order]
    ranges = ranges[order]
    points_xyz = points_xyz[order]
    points_rgb = points_rgb[order]

    range_image = np.full((n_scan, horizon_scan), np.inf, dtype=np.float32)
    xyz_image = np.full((n_scan, horizon_scan, 3), np.nan, dtype=np.float32)
    rgb_image = np.full((n_scan, horizon_scan, 3), np.nan, dtype=np.float32)

    for px, py, r, xyz, rgb in zip(proj_x, proj_y, ranges, points_xyz, points_rgb):
        if r < range_image[py, px]:
            range_image[py, px] = r
            xyz_image[py, px] = xyz
            rgb_image[py, px] = rgb

    valid = np.isfinite(range_image)
    return xyz_image[valid], rgb_image[valid]


def per_channel_filter_params(channel: str) -> Dict[str, float]:
    if channel == "CAM_BACK":
        return dict(min_depth=2.5, max_depth=30.0, crop_top_ratio=0.15, crop_bottom_ratio=0.10, crop_side_ratio=0.04)
    if channel in ["CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
        return dict(min_depth=2.5, max_depth=35.0, crop_top_ratio=0.15, crop_bottom_ratio=0.10, crop_side_ratio=0.05)
    return dict(min_depth=2.5, max_depth=40.0, crop_top_ratio=0.18, crop_bottom_ratio=0.12, crop_side_ratio=0.06)


def run_inference_for_sample(
    nusc: NuScenes,
    sample_token: str,
    model,
    depth_transform,
    device: torch.device,
    max_width: int = 960,
) -> Dict[str, dict]:
    sample = nusc.get("sample", sample_token)
    results: Dict[str, dict] = {}

    for channel in CAMERA_CHANNELS:
        sample_data = nusc.get("sample_data", sample["data"][channel])
        calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        image_path = Path(nusc.dataroot) / sample_data["filename"]
        image_pil = Image.open(image_path).convert("RGB")

        if image_pil.width > max_width:
            new_height = int(image_pil.height * max_width / image_pil.width)
            image_pil = image_pil.resize((max_width, new_height))

        image_np = np.asarray(image_pil)
        intrinsic = np.asarray(calib["camera_intrinsic"], dtype=np.float32)
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        scale_x = image_pil.width / float(sample_data["width"])
        scale_y = image_pil.height / float(sample_data["height"])
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        f_px = float((fx + fy) * 0.5)

        image_tensor = depth_transform(image_pil).to(device)
        f_px_tensor = torch.tensor(f_px, device=device, dtype=torch.float32)
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px_tensor)
        depth = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
        del image_tensor, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

        results[channel] = {
            "depth": depth,
            "image": image_np,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "calibrated_sensor": calib,
        }
    return results


def build_outputs_for_sample(results: Dict[str, dict]) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, dict]:
    ring_points = []
    ring_colors = []

    for channel in CAMERA_CHANNELS:
        record = results[channel]
        scaled_depth = record["depth"] * SCALE_BY_CHANNEL[channel]
        params = per_channel_filter_params(channel)

        points_cam, colors_cam = depth_to_colored_points_pinhole(
            scaled_depth,
            record["image"],
            record["fx"],
            record["fy"],
            record["cx"],
            record["cy"],
            **params,
        )

        t_ego_from_cam = transform_from_translation_rotation(
            record["calibrated_sensor"]["translation"],
            record["calibrated_sensor"]["rotation"],
        )
        points_ego = transform_points(points_cam, t_ego_from_cam)
        ring_points.append(points_ego)
        ring_colors.append(colors_cam)

    ring_points_dense = np.concatenate(ring_points, axis=0)
    ring_colors_dense = np.concatenate(ring_colors, axis=0)

    ring_pcd = make_point_cloud(ring_points_dense, ring_colors_dense)
    ring_pcd = ring_pcd.voxel_down_sample(voxel_size=0.12)
    ring_pcd, _ = ring_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    ring_points_clean = np.asarray(ring_pcd.points, dtype=np.float32)
    ring_colors_clean = np.asarray(ring_pcd.colors, dtype=np.float32)
    pseudo_points, pseudo_colors = pseudo_lidar_from_dense_cloud(
        ring_points_clean,
        ring_colors_clean,
        n_scan=64,
        horizon_scan=2048,
        fov_up_deg=12.0,
        fov_down_deg=-30.0,
        range_min=2.0,
        range_max=50.0,
    )
    pseudo_pcd = make_point_cloud(pseudo_points, pseudo_colors)
    pseudo_pcd = pseudo_pcd.voxel_down_sample(voxel_size=0.01)

    summary = {
        "ring_num_points": int(np.asarray(ring_pcd.points).shape[0]),
        "pseudo_num_points": int(np.asarray(pseudo_pcd.points).shape[0]),
    }
    return ring_pcd, pseudo_pcd, summary


def load_lidar_top_pcd(nusc: NuScenes, sample_token: str) -> o3d.geometry.PointCloud:
    sample = nusc.get("sample", sample_token)
    lidar_info = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_path = Path(nusc.dataroot) / lidar_info["filename"]
    raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    lidar_points_sensor = raw[:, :3]

    calib = nusc.get("calibrated_sensor", lidar_info["calibrated_sensor_token"])
    t_ego_from_lidar = transform_from_translation_rotation(calib["translation"], calib["rotation"]).astype(np.float32)
    lidar_points_ego = transform_points(lidar_points_sensor.astype(np.float32), t_ego_from_lidar)
    lidar_colors = np.tile(np.array([[0.95, 0.15, 0.15]], dtype=np.float32), (lidar_points_ego.shape[0], 1))
    return make_point_cloud(lidar_points_ego, lidar_colors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/home/clara/ml-depth-pro/slam_readiness_nuscenes/outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    nusc = NuScenes(version=manifest["version"], dataroot=manifest["dataroot"], verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(str(ML_DEPTH_PRO_ROOT))
    model, depth_transform = depth_pro.create_model_and_transforms()
    model.eval()
    model = model.to(device)

    scene_dir = args.output_root / manifest["scene_name"]
    scene_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    selected_samples = manifest["samples"][: args.max_samples] if args.max_samples else manifest["samples"]

    for sample_meta in selected_samples:
        sample_token = sample_meta["sample_token"]
        sample_dir = scene_dir / sample_token
        sample_dir.mkdir(parents=True, exist_ok=True)

        results = run_inference_for_sample(nusc, sample_token, model, depth_transform, device)
        ring_pcd, pseudo_pcd, summary = build_outputs_for_sample(results)
        lidar_pcd = load_lidar_top_pcd(nusc, sample_token)

        ring_path = sample_dir / "pcd_ring_6cams_ego.ply"
        pseudo_path = sample_dir / "pcd_pseudolidar_ego.ply"
        lidar_path = sample_dir / "pcd_lidar_top_ego.ply"
        o3d.io.write_point_cloud(str(ring_path), ring_pcd)
        o3d.io.write_point_cloud(str(pseudo_path), pseudo_pcd)
        o3d.io.write_point_cloud(str(lidar_path), lidar_pcd)

        info = {
            "sample_token": sample_token,
            "index": sample_meta["index"],
            "timestamp_s": sample_meta["timestamp_s"],
            "ring_path": str(ring_path),
            "pseudo_path": str(pseudo_path),
            "lidar_path": str(lidar_path),
            **summary,
        }
        (sample_dir / "summary.json").write_text(json.dumps(info, indent=2))
        processed.append(info)
        print(json.dumps(info, indent=2))

    run_summary = {
        "scene_name": manifest["scene_name"],
        "manifest_path": str(args.manifest),
        "num_processed": len(processed),
        "samples": processed,
    }
    (scene_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print("saved:", scene_dir / "run_summary.json")


if __name__ == "__main__":
    main()
