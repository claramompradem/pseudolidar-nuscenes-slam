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
    # ICP estima la transformacion que mapea puntos del frame source al frame target.
    # Si T_global_from_ego representa la pose del vehiculo en global, entonces:
    # p_target = inv(T_global_from_target) @ T_global_from_source @ p_source
    return np.linalg.inv(target_global) @ source_global


def load_and_prepare_pcd(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
    )
    return pcd


def rotation_angle_deg(transform: np.ndarray) -> float:
    r = transform[:3, :3]
    trace = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(transform[:3, 3]))


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_corr_dist: float,
    init_transform: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr_dist,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def transform_error(est: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    delta = np.linalg.inv(gt) @ est
    return {
        "translation_error_m": translation_norm(delta),
        "rotation_error_deg": rotation_angle_deg(delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, default=Path("/home/clara/datasets/nuscenes"))
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--max-corr-dist", type=float, default=1.5)
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text())
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)

    samples = sorted(run_summary["samples"], key=lambda x: x["index"])
    pair_results = []

    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]

        src_token = src_meta["sample_token"]
        tgt_token = tgt_meta["sample_token"]

        src_pseudo = load_and_prepare_pcd(Path(src_meta["pseudo_path"]), voxel_size=args.voxel_size)
        tgt_pseudo = load_and_prepare_pcd(Path(tgt_meta["pseudo_path"]), voxel_size=args.voxel_size)
        src_gt = load_and_prepare_pcd(
            Path(src_meta["pseudo_path"]).parent / "pcd_lidar_top_ego.ply", voxel_size=args.voxel_size
        )
        tgt_gt = load_and_prepare_pcd(
            Path(tgt_meta["pseudo_path"]).parent / "pcd_lidar_top_ego.ply", voxel_size=args.voxel_size
        )

        src_pose = pose_global_from_sample(nusc, src_token)
        tgt_pose = pose_global_from_sample(nusc, tgt_token)
        gt_rel = relative_transform(src_pose, tgt_pose)

        pseudo_icp = register_pair(src_pseudo, tgt_pseudo, args.max_corr_dist, np.eye(4))
        gt_icp = register_pair(src_gt, tgt_gt, args.max_corr_dist, np.eye(4))

        pseudo_err = transform_error(pseudo_icp.transformation, gt_rel)
        gt_err = transform_error(gt_icp.transformation, gt_rel)

        pair_results.append(
            {
                "pair_index": i,
                "source_token": src_token,
                "target_token": tgt_token,
                "dt_s": float(tgt_meta["timestamp_s"] - src_meta["timestamp_s"]),
                "gt_relative_translation_m": translation_norm(gt_rel),
                "gt_relative_rotation_deg": rotation_angle_deg(gt_rel),
                "pseudo_icp_fitness": float(pseudo_icp.fitness),
                "pseudo_icp_rmse": float(pseudo_icp.inlier_rmse),
                "pseudo_est_translation_m": translation_norm(pseudo_icp.transformation),
                "pseudo_est_rotation_deg": rotation_angle_deg(pseudo_icp.transformation),
                **{f"pseudo_{k}": v for k, v in pseudo_err.items()},
                "gt_icp_fitness": float(gt_icp.fitness),
                "gt_icp_rmse": float(gt_icp.inlier_rmse),
                "gt_est_translation_m": translation_norm(gt_icp.transformation),
                "gt_est_rotation_deg": rotation_angle_deg(gt_icp.transformation),
                **{f"gt_{k}": v for k, v in gt_err.items()},
            }
        )

    metrics = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "pairs": pair_results,
    }
    if pair_results:
        metrics["aggregate"] = {
            "pseudo_translation_error_mean_m": float(np.mean([p["pseudo_translation_error_m"] for p in pair_results])),
            "pseudo_rotation_error_mean_deg": float(np.mean([p["pseudo_rotation_error_deg"] for p in pair_results])),
            "pseudo_fitness_mean": float(np.mean([p["pseudo_icp_fitness"] for p in pair_results])),
            "gt_translation_error_mean_m": float(np.mean([p["gt_translation_error_m"] for p in pair_results])),
            "gt_rotation_error_mean_deg": float(np.mean([p["gt_rotation_error_deg"] for p in pair_results])),
            "gt_fitness_mean": float(np.mean([p["gt_icp_fitness"] for p in pair_results])),
        }

    out_path = args.run_summary.parent / "pairwise_registration_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(out_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
