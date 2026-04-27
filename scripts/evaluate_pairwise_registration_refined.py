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


def make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def filter_points_for_slam(
    points: np.ndarray,
    r_min: float = 3.0,
    r_max: float = 30.0,
    z_min: float = -1.5,
    z_max: float = 2.0,
    x_abs_min: float = 1.5,
) -> np.ndarray:
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = (ranges >= r_min) & (ranges <= r_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    mask &= np.abs(points[:, 0]) >= x_abs_min
    return points[mask]


def prepare_pcd(points: np.ndarray, voxel_size: float, remove_outlier: bool) -> o3d.geometry.PointCloud:
    pcd = make_pcd(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if remove_outlier and len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
    )
    return pcd


def register_naive(source_points: np.ndarray, target_points: np.ndarray) -> o3d.pipelines.registration.RegistrationResult:
    source = prepare_pcd(source_points, voxel_size=0.5, remove_outlier=False)
    target = prepare_pcd(target_points, voxel_size=0.5, remove_outlier=False)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        1.5,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def register_refined(source_points: np.ndarray, target_points: np.ndarray) -> o3d.pipelines.registration.RegistrationResult:
    source_filtered = filter_points_for_slam(source_points)
    target_filtered = filter_points_for_slam(target_points)

    current = np.eye(4)
    final_result = None
    for voxel_size, max_corr in [(1.0, 3.0), (0.5, 1.5), (0.25, 0.75)]:
        source = prepare_pcd(source_filtered, voxel_size=voxel_size, remove_outlier=True)
        target = prepare_pcd(target_filtered, voxel_size=voxel_size, remove_outlier=True)
        final_result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_corr,
            current,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        current = final_result.transformation
    return final_result


def result_to_metrics(prefix: str, result: o3d.pipelines.registration.RegistrationResult, gt_rel: np.ndarray) -> Dict[str, float]:
    err = transform_error(result.transformation, gt_rel)
    return {
        f"{prefix}_fitness": float(result.fitness),
        f"{prefix}_rmse": float(result.inlier_rmse),
        f"{prefix}_est_translation_m": translation_norm(result.transformation),
        f"{prefix}_est_rotation_deg": rotation_angle_deg(result.transformation),
        f"{prefix}_translation_error_m": err["translation_error_m"],
        f"{prefix}_rotation_error_deg": err["rotation_error_deg"],
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
    pair_results = []

    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]

        src_pseudo_points = load_points(Path(src_meta["pseudo_path"]))
        tgt_pseudo_points = load_points(Path(tgt_meta["pseudo_path"]))
        src_gt_points = load_points(Path(src_meta["lidar_path"]))
        tgt_gt_points = load_points(Path(tgt_meta["lidar_path"]))

        src_pose = pose_global_from_sample(nusc, src_meta["sample_token"])
        tgt_pose = pose_global_from_sample(nusc, tgt_meta["sample_token"])
        gt_rel = relative_transform(src_pose, tgt_pose)

        pseudo_naive = register_naive(src_pseudo_points, tgt_pseudo_points)
        pseudo_refined = register_refined(src_pseudo_points, tgt_pseudo_points)
        gt_naive = register_naive(src_gt_points, tgt_gt_points)
        gt_refined = register_refined(src_gt_points, tgt_gt_points)

        pair_results.append(
            {
                "pair_index": i,
                "source_token": src_meta["sample_token"],
                "target_token": tgt_meta["sample_token"],
                "dt_s": float(tgt_meta["timestamp_s"] - src_meta["timestamp_s"]),
                "gt_relative_translation_m": translation_norm(gt_rel),
                "gt_relative_rotation_deg": rotation_angle_deg(gt_rel),
                **result_to_metrics("pseudo_naive", pseudo_naive, gt_rel),
                **result_to_metrics("pseudo_refined", pseudo_refined, gt_rel),
                **result_to_metrics("gt_naive", gt_naive, gt_rel),
                **result_to_metrics("gt_refined", gt_refined, gt_rel),
            }
        )

    metrics = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "pairs": pair_results,
    }
    if pair_results:
        metrics["aggregate"] = {
            "pseudo_naive_translation_error_mean_m": float(np.mean([p["pseudo_naive_translation_error_m"] for p in pair_results])),
            "pseudo_naive_rotation_error_mean_deg": float(np.mean([p["pseudo_naive_rotation_error_deg"] for p in pair_results])),
            "pseudo_naive_fitness_mean": float(np.mean([p["pseudo_naive_fitness"] for p in pair_results])),
            "pseudo_refined_translation_error_mean_m": float(np.mean([p["pseudo_refined_translation_error_m"] for p in pair_results])),
            "pseudo_refined_rotation_error_mean_deg": float(np.mean([p["pseudo_refined_rotation_error_deg"] for p in pair_results])),
            "pseudo_refined_fitness_mean": float(np.mean([p["pseudo_refined_fitness"] for p in pair_results])),
            "gt_naive_translation_error_mean_m": float(np.mean([p["gt_naive_translation_error_m"] for p in pair_results])),
            "gt_refined_translation_error_mean_m": float(np.mean([p["gt_refined_translation_error_m"] for p in pair_results])),
        }

    out_path = args.run_summary.parent / "pairwise_registration_refined_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(out_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
