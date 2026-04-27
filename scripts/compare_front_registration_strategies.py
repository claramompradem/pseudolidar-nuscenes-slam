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


def select_front(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)
    mask = (r >= 2.0) & (r <= 35.0) & (z >= -2.0) & (z <= 2.5)
    mask &= (x > 0.0) & (np.abs(y) < 15.0)
    return points[mask]


def prepare_pcd(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) > 50:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
        )
    return pcd


def register_front_icp_only(source_points: np.ndarray, target_points: np.ndarray):
    source = prepare_pcd(source_points, voxel_size=0.5)
    target = prepare_pcd(target_points, voxel_size=0.5)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        1.5,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def preprocess_fpfh(points: np.ndarray, voxel_size: float):
    pcd = prepare_pcd(points, voxel_size=voxel_size)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd, fpfh


def register_front_global_then_icp(source_points: np.ndarray, target_points: np.ndarray):
    coarse_voxel = 1.0
    source_down, source_fpfh = preprocess_fpfh(source_points, coarse_voxel)
    target_down, target_fpfh = preprocess_fpfh(target_points, coarse_voxel)

    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.0),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500),
    )

    source_fine = prepare_pcd(source_points, voxel_size=0.5)
    target_fine = prepare_pcd(target_points, voxel_size=0.5)
    icp_result = o3d.pipelines.registration.registration_icp(
        source_fine,
        target_fine,
        1.5,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return global_result, icp_result


def result_metrics(result, gt_rel):
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

    pair_results = []
    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]

        src_points = select_front(load_points(Path(src_meta["pseudo_path"])))
        tgt_points = select_front(load_points(Path(tgt_meta["pseudo_path"])))

        src_pose = pose_global_from_sample(nusc, src_meta["sample_token"])
        tgt_pose = pose_global_from_sample(nusc, tgt_meta["sample_token"])
        gt_rel = relative_transform(src_pose, tgt_pose)

        direct = register_front_icp_only(src_points, tgt_points)
        global_result, refined = register_front_global_then_icp(src_points, tgt_points)

        pair_results.append(
            {
                "pair_index": i,
                "source_token": src_meta["sample_token"],
                "target_token": tgt_meta["sample_token"],
                "dt_s": float(tgt_meta["timestamp_s"] - src_meta["timestamp_s"]),
                "num_source_points": int(src_points.shape[0]),
                "num_target_points": int(tgt_points.shape[0]),
                "gt_relative_translation_m": translation_norm(gt_rel),
                "gt_relative_rotation_deg": rotation_angle_deg(gt_rel),
                "front_icp_only": result_metrics(direct, gt_rel),
                "front_global_then_icp": {
                    "global_fitness": float(global_result.fitness),
                    "global_rmse": float(global_result.inlier_rmse),
                    **result_metrics(refined, gt_rel),
                },
            }
        )

    output = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "pairs": pair_results,
        "aggregate": {
            "front_icp_only_translation_error_mean_m": float(np.mean([p["front_icp_only"]["translation_error_m"] for p in pair_results])),
            "front_icp_only_rotation_error_mean_deg": float(np.mean([p["front_icp_only"]["rotation_error_deg"] for p in pair_results])),
            "front_icp_only_fitness_mean": float(np.mean([p["front_icp_only"]["fitness"] for p in pair_results])),
            "front_global_then_icp_translation_error_mean_m": float(np.mean([p["front_global_then_icp"]["translation_error_m"] for p in pair_results])),
            "front_global_then_icp_rotation_error_mean_deg": float(np.mean([p["front_global_then_icp"]["rotation_error_deg"] for p in pair_results])),
            "front_global_then_icp_fitness_mean": float(np.mean([p["front_global_then_icp"]["fitness"] for p in pair_results])),
        },
    }

    out_path = args.run_summary.parent / "front_registration_strategies.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path)
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
