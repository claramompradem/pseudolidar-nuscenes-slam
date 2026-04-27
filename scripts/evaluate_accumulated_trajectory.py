from __future__ import annotations

import argparse
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


def select_front_baseline(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)
    mask = (r >= 2.0) & (r <= 35.0) & (z >= -2.0) & (z <= 2.5)
    mask &= (x > 0.0) & (np.abs(y) < 15.0)
    return points[mask]


def select_front_narrow(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points[:, :3], axis=1)
    mask = (r >= 2.0) & (r <= 35.0) & (z >= -2.0) & (z <= 2.5)
    mask &= (x > 0.0) & (np.abs(y) < 10.0)
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


def register_front_icp_only(source_points: np.ndarray, target_points: np.ndarray):
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


def register_front_cfg05(source_points: np.ndarray, target_points: np.ndarray):
    coarse_voxel = 0.8
    global_corr = 2.0
    fine_voxel = 0.4
    icp_corr = 1.5

    source_down, source_fpfh = preprocess_fpfh(source_points, coarse_voxel)
    target_down, target_fpfh = preprocess_fpfh(target_points, coarse_voxel)
    if len(source_down.points) < 50 or len(target_down.points) < 50:
        return None

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
        return None

    return o3d.pipelines.registration.registration_icp(
        source_fine,
        target_fine,
        icp_corr,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def build_trajectory_summary(name: str, estimated_poses_to_ref: List[np.ndarray], gt_poses_to_ref: List[np.ndarray]):
    frame_errors = []
    xy_points = []
    gt_xy_points = []
    for idx, (est_pose, gt_pose) in enumerate(zip(estimated_poses_to_ref, gt_poses_to_ref)):
        err = transform_error(est_pose, gt_pose)
        frame_errors.append(
            {
                "frame_index": idx,
                "translation_error_m": err["translation_error_m"],
                "rotation_error_deg": err["rotation_error_deg"],
            }
        )
        xy_points.append([float(est_pose[0, 3]), float(est_pose[1, 3])])
        gt_xy_points.append([float(gt_pose[0, 3]), float(gt_pose[1, 3])])

    valid_errors = frame_errors[1:] if len(frame_errors) > 1 else frame_errors
    aggregate = {
        "mean_translation_error_m": float(np.mean([e["translation_error_m"] for e in valid_errors])),
        "mean_rotation_error_deg": float(np.mean([e["rotation_error_deg"] for e in valid_errors])),
        "final_translation_error_m": float(valid_errors[-1]["translation_error_m"]),
        "final_rotation_error_deg": float(valid_errors[-1]["rotation_error_deg"]),
        "num_frames": int(len(frame_errors)),
    }
    return {
        "name": name,
        "aggregate": aggregate,
        "frame_errors": frame_errors,
        "trajectory_xy": xy_points,
        "gt_trajectory_xy": gt_xy_points,
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

    sample_poses_global = [pose_global_from_sample(nusc, s["sample_token"]) for s in samples]
    ref_pose_global = sample_poses_global[0]
    gt_poses_to_ref = [np.linalg.inv(ref_pose_global) @ pose for pose in sample_poses_global]

    strategies = {
        "front_icp_only": (select_front_baseline, register_front_icp_only),
        "front_baseline_cfg05": (select_front_baseline, register_front_cfg05),
        "front_narrow_cfg05": (select_front_narrow, register_front_cfg05),
    }

    pairwise_results = {name: [] for name in strategies}
    estimated_poses_to_ref = {name: [np.eye(4, dtype=np.float64)] for name in strategies}

    for i in range(len(samples) - 1):
        src_meta = samples[i]
        tgt_meta = samples[i + 1]
        src_all = load_points(Path(src_meta["pseudo_path"]))
        tgt_all = load_points(Path(tgt_meta["pseudo_path"]))
        gt_rel = relative_transform(sample_poses_global[i], sample_poses_global[i + 1])

        for name, (selector, registrar) in strategies.items():
            src_points = selector(src_all)
            tgt_points = selector(tgt_all)
            result = registrar(src_points, tgt_points)
            if result is None:
                metrics = None
                transform = None
                next_pose = estimated_poses_to_ref[name][-1].copy()
            else:
                metrics = {
                    "fitness": float(result.fitness),
                    "rmse": float(result.inlier_rmse),
                    "translation_error_m": transform_error(result.transformation, gt_rel)["translation_error_m"],
                    "rotation_error_deg": transform_error(result.transformation, gt_rel)["rotation_error_deg"],
                }
                transform = result.transformation
                next_pose = estimated_poses_to_ref[name][-1] @ np.linalg.inv(transform)

            estimated_poses_to_ref[name].append(next_pose)
            pairwise_results[name].append(
                {
                    "pair_index": i,
                    "source_token": src_meta["sample_token"],
                    "target_token": tgt_meta["sample_token"],
                    "num_source_points": int(src_points.shape[0]),
                    "num_target_points": int(tgt_points.shape[0]),
                    "estimated_transform": None if transform is None else transform.tolist(),
                    "metrics": metrics,
                }
            )

    trajectories = {
        name: build_trajectory_summary(name, estimated_poses_to_ref[name], gt_poses_to_ref)
        for name in strategies
    }

    output = {
        "scene_name": run_summary["scene_name"],
        "num_samples": len(samples),
        "sample_tokens": [s["sample_token"] for s in samples],
        "gt_trajectory_xy": [[float(p[0, 3]), float(p[1, 3])] for p in gt_poses_to_ref],
        "pairwise_results": pairwise_results,
        "trajectories": trajectories,
        "best_recommendation": "front_baseline_cfg05",
    }

    out_path = args.run_summary.parent / "accumulated_trajectory_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(out_path)
    print(
        json.dumps(
            {name: output["trajectories"][name]["aggregate"] for name in output["trajectories"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
