from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


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


def preprocess_fpfh(points: np.ndarray, voxel_size: float):
    pcd = prepare_pcd(points, voxel_size=voxel_size)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd, fpfh


def register_with_config(source_points: np.ndarray, target_points: np.ndarray, config: Dict[str, float]):
    coarse_voxel = float(config["coarse_voxel"])
    fine_voxel = float(config["fine_voxel"])
    global_corr = float(config["global_corr"])
    icp_corr = float(config["icp_corr"])

    source_down, source_fpfh = preprocess_fpfh(source_points, coarse_voxel)
    target_down, target_fpfh = preprocess_fpfh(target_points, coarse_voxel)
    if len(source_down.points) < 50 or len(target_down.points) < 50:
        return None, None

    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=global_corr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(global_corr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500),
    )

    source_fine = prepare_pcd(source_points, fine_voxel)
    target_fine = prepare_pcd(target_points, fine_voxel)
    if len(source_fine.points) < 50 or len(target_fine.points) < 50:
        return global_result, None

    icp_result = o3d.pipelines.registration.registration_icp(
        source_fine,
        target_fine,
        icp_corr,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return global_result, icp_result


def result_metrics(result, gt_rel):
    if result is None:
        return None
    err = transform_error(result.transformation, gt_rel)
    return {
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "translation_error_m": err["translation_error_m"],
        "rotation_error_deg": err["rotation_error_deg"],
    }


def aggregate_metrics(valid: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "translation_error_mean_m": float(np.mean([m["translation_error_m"] for m in valid])),
        "rotation_error_mean_deg": float(np.mean([m["rotation_error_deg"] for m in valid])),
        "fitness_mean": float(np.mean([m["fitness"] for m in valid])),
        "rmse_mean": float(np.mean([m["rmse"] for m in valid])),
        "num_valid_pairs": int(len(valid)),
    }


def get_param_grid() -> List[Dict[str, float]]:
    coarse_voxels = [0.8, 1.0, 1.2]
    global_corrs = [1.6, 2.0]
    fine_voxels = [0.4, 0.5]
    icp_corrs = [1.0, 1.5]
    configs = []
    for idx, (coarse_voxel, global_corr, fine_voxel, icp_corr) in enumerate(
        itertools.product(coarse_voxels, global_corrs, fine_voxels, icp_corrs)
    ):
        configs.append(
            {
                "name": f"cfg_{idx:02d}",
                "coarse_voxel": coarse_voxel,
                "global_corr": global_corr,
                "fine_voxel": fine_voxel,
                "icp_corr": icp_corr,
            }
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, default=Path("/home/clara/datasets/nuscenes"))
    parser.add_argument("--version", type=str, default="v1.0-mini")
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text())
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    samples = sorted(run_summary["samples"], key=lambda x: x["index"])
    configs = get_param_grid()

    pair_data = []
    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]
        src_points = select_front(load_points(Path(src_meta["pseudo_path"])))
        tgt_points = select_front(load_points(Path(tgt_meta["pseudo_path"])))
        src_pose = pose_global_from_sample(nusc, src_meta["sample_token"])
        tgt_pose = pose_global_from_sample(nusc, tgt_meta["sample_token"])
        gt_rel = relative_transform(src_pose, tgt_pose)
        pair_data.append(
            {
                "pair_index": i,
                "source_token": src_meta["sample_token"],
                "target_token": tgt_meta["sample_token"],
                "source_points": src_points,
                "target_points": tgt_points,
                "gt_rel": gt_rel,
            }
        )

    config_results = []
    for config in configs:
        pair_results = []
        for pair in pair_data:
            global_result, icp_result = register_with_config(pair["source_points"], pair["target_points"], config)
            pair_results.append(
                {
                    "pair_index": pair["pair_index"],
                    "global_fitness": None if global_result is None else float(global_result.fitness),
                    "global_rmse": None if global_result is None else float(global_result.inlier_rmse),
                    "metrics": result_metrics(icp_result, pair["gt_rel"]),
                }
            )

        valid = [p["metrics"] for p in pair_results if p["metrics"] is not None]
        config_results.append(
            {
                **config,
                "aggregate": None if not valid else aggregate_metrics(valid),
                "pairs": pair_results,
            }
        )

    valid_configs = [cfg for cfg in config_results if cfg["aggregate"] is not None]
    valid_configs.sort(
        key=lambda cfg: (
            cfg["aggregate"]["translation_error_mean_m"],
            cfg["aggregate"]["rotation_error_mean_deg"],
        )
    )

    output = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_data),
        "configs": config_results,
        "best_by_translation_then_rotation": valid_configs[:5],
    }

    out_path = args.run_summary.parent / "front_global_icp_param_sweep.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path)
    print(json.dumps(output["best_by_translation_then_rotation"], indent=2))


if __name__ == "__main__":
    main()
