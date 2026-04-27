from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


def transform_from_translation_rotation(translation: List[float], rotation: List[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def pose_global_from_sample(nusc: NuScenes, sample_token: str) -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    return transform_from_translation_rotation(ego_pose["translation"], ego_pose["rotation"])


def relative_transform(source_global: np.ndarray, target_global: np.ndarray) -> np.ndarray:
    return np.linalg.inv(target_global) @ source_global


def rotation_angle_deg(transform: np.ndarray) -> float:
    r = transform[:3, :3]
    trace = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(transform[:3, 3]))


def transform_error(est: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    delta = np.linalg.inv(gt) @ est
    return {
        "translation_error_m": translation_norm(delta),
        "rotation_error_deg": rotation_angle_deg(delta),
    }


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float64)


def select_subset(points: np.ndarray, subset: str) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)
    base = (r >= 2.0) & (r <= 35.0) & (z >= -2.0) & (z <= 2.5)

    if subset == "all":
        mask = base
    elif subset == "front":
        mask = base & (x > 0.0) & (np.abs(y) < 15.0)
    elif subset == "front_left":
        mask = base & (
            ((x > 0.0) & (np.abs(y) < 15.0)) |
            (y > 0.0)
        )
    elif subset == "front_right":
        mask = base & (
            ((x > 0.0) & (np.abs(y) < 15.0)) |
            (y < 0.0)
        )
    else:
        raise ValueError(subset)
    return points[mask]


def prepare_pcd(points: np.ndarray, voxel_size: float = 0.5) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) > 50:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
        )
    return pcd


def register_subset(source_points: np.ndarray, target_points: np.ndarray):
    if source_points.shape[0] < 100 or target_points.shape[0] < 100:
        return None
    source = prepare_pcd(source_points, voxel_size=0.5)
    target = prepare_pcd(target_points, voxel_size=0.5)
    if len(source.points) < 50 or len(target.points) < 50:
        return None
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        1.5,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def metrics_from_result(result, gt_rel):
    if result is None:
        return None
    err = transform_error(result.transformation, gt_rel)
    return {
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "est_translation_m": translation_norm(result.transformation),
        "est_rotation_deg": rotation_angle_deg(result.transformation),
        "translation_error_m": err["translation_error_m"],
        "rotation_error_deg": err["rotation_error_deg"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, default=Path("/home/clara/datasets/nuscenes"))
    parser.add_argument("--version", type=str, default="v1.0-mini")
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text())
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    samples = sorted(run_summary["samples"], key=lambda x: x["index"])
    subsets = ["all", "front", "front_left", "front_right"]

    pair_results = []
    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]
        src_points = load_points(Path(src_meta["pseudo_path"]))
        tgt_points = load_points(Path(tgt_meta["pseudo_path"]))

        src_pose = pose_global_from_sample(nusc, src_meta["sample_token"])
        tgt_pose = pose_global_from_sample(nusc, tgt_meta["sample_token"])
        gt_rel = relative_transform(src_pose, tgt_pose)

        pair_info = {
            "pair_index": i,
            "source_token": src_meta["sample_token"],
            "target_token": tgt_meta["sample_token"],
            "dt_s": float(tgt_meta["timestamp_s"] - src_meta["timestamp_s"]),
            "subsets": {},
        }

        for subset in subsets:
            src_subset = select_subset(src_points, subset)
            tgt_subset = select_subset(tgt_points, subset)
            result = register_subset(src_subset, tgt_subset)
            pair_info["subsets"][subset] = {
                "num_source_points": int(src_subset.shape[0]),
                "num_target_points": int(tgt_subset.shape[0]),
                "metrics": metrics_from_result(result, gt_rel),
            }

        pair_results.append(pair_info)

    aggregate = {}
    for subset in subsets:
        valid = [p["subsets"][subset]["metrics"] for p in pair_results if p["subsets"][subset]["metrics"] is not None]
        aggregate[subset] = None if not valid else {
            "translation_error_mean_m": float(np.mean([m["translation_error_m"] for m in valid])),
            "rotation_error_mean_deg": float(np.mean([m["rotation_error_deg"] for m in valid])),
            "fitness_mean": float(np.mean([m["fitness"] for m in valid])),
            "num_valid_pairs": int(len(valid)),
        }

    output = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "pairs": pair_results,
        "aggregate": aggregate,
    }
    out_path = args.run_summary.parent / "subset_registration_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path)
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
