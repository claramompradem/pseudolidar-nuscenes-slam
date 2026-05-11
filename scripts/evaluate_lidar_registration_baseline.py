from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


REGIONS = ["all", "front", "near", "mid", "far"]


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
    # Maps points from the source ego frame into the target ego frame.
    return np.linalg.inv(target_global) @ source_global


def rotation_angle_deg(transform: np.ndarray) -> float:
    trace = np.clip((np.trace(transform[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(transform[:3, 3]))


def transform_error(estimated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    delta = np.linalg.inv(ground_truth) @ estimated
    return {
        "translation_error_m": translation_norm(delta),
        "rotation_error_deg": rotation_angle_deg(delta),
    }


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


def filter_region(points: np.ndarray, region: str) -> np.ndarray:
    if region == "all":
        return points

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ranges = np.linalg.norm(points[:, :3], axis=1)
    base = (ranges >= 2.0) & (ranges <= 35.0) & (z >= -2.0) & (z <= 2.5)

    if region == "front":
        mask = base & (x > 0.0) & (np.abs(y) < 15.0)
    elif region == "near":
        mask = base & (ranges < 10.0)
    elif region == "mid":
        mask = base & (ranges >= 10.0) & (ranges < 20.0)
    elif region == "far":
        mask = base & (ranges >= 20.0) & (ranges <= 35.0)
    else:
        raise ValueError(f"Unknown region: {region}")
    return points[mask]


def prepare_pcd(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) > 50:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=30)
        )
    return pcd


def register_icp_direct(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_size: float,
    max_corr_dist: float,
) -> o3d.pipelines.registration.RegistrationResult | None:
    if source_points.shape[0] < 100 or target_points.shape[0] < 100:
        return None
    source = prepare_pcd(source_points, voxel_size)
    target = prepare_pcd(target_points, voxel_size)
    if len(source.points) < 50 or len(target.points) < 50:
        return None
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def preprocess_fpfh(points: np.ndarray, voxel_size: float):
    pcd = prepare_pcd(points, voxel_size)
    if len(pcd.points) < 50:
        return None, None
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd, fpfh


def register_global_then_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    coarse_voxel: float,
    global_corr: float,
    fine_voxel: float,
    icp_corr: float,
) -> tuple[o3d.pipelines.registration.RegistrationResult, o3d.pipelines.registration.RegistrationResult] | None:
    if source_points.shape[0] < 100 or target_points.shape[0] < 100:
        return None

    source_down, source_fpfh = preprocess_fpfh(source_points, coarse_voxel)
    target_down, target_fpfh = preprocess_fpfh(target_points, coarse_voxel)
    if source_down is None or target_down is None:
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

    icp_result = o3d.pipelines.registration.registration_icp(
        source_fine,
        target_fine,
        icp_corr,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return global_result, icp_result


def metrics_from_result(
    result: o3d.pipelines.registration.RegistrationResult | None,
    gt_rel: np.ndarray,
    method: str,
    region: str,
    num_source_points: int,
    num_target_points: int,
    global_result: o3d.pipelines.registration.RegistrationResult | None = None,
) -> dict | None:
    if result is None:
        return None
    err = transform_error(result.transformation, gt_rel)
    metrics = {
        "method": method,
        "region": region,
        "num_source_points": int(num_source_points),
        "num_target_points": int(num_target_points),
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "estimated_translation_m": translation_norm(result.transformation),
        "estimated_rotation_deg": rotation_angle_deg(result.transformation),
        **err,
    }
    if global_result is not None:
        metrics["global_fitness"] = float(global_result.fitness)
        metrics["global_rmse"] = float(global_result.inlier_rmse)
    return metrics


def aggregate_rows(rows: Iterable[dict]) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['method']}::{row['region']}", []).append(row)

    aggregate = {}
    for key, values in grouped.items():
        method, region = key.split("::", 1)
        aggregate.setdefault(method, {})[region] = {
            "translation_error_mean_m": float(np.mean([item["translation_error_m"] for item in values])),
            "rotation_error_mean_deg": float(np.mean([item["rotation_error_deg"] for item in values])),
            "fitness_mean": float(np.mean([item["fitness"] for item in values])),
            "rmse_mean": float(np.mean([item["rmse"] for item in values])),
            "num_source_points_mean": float(np.mean([item["num_source_points"] for item in values])),
            "num_target_points_mean": float(np.mean([item["num_target_points"] for item in values])),
            "num_valid_pairs": int(len(values)),
        }
    return aggregate


def write_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "pair_index",
        "source_token",
        "target_token",
        "dt_s",
        "method",
        "region",
        "num_source_points",
        "num_target_points",
        "fitness",
        "rmse",
        "estimated_translation_m",
        "estimated_rotation_deg",
        "translation_error_m",
        "rotation_error_deg",
        "global_fitness",
        "global_rmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_pseudolidar_aggregates(output_dir: Path) -> dict:
    aggregates = {}

    pairwise_path = output_dir / "pairwise_registration_metrics.json"
    if pairwise_path.exists():
        pairwise = json.loads(pairwise_path.read_text(encoding="utf-8-sig"))
        agg = pairwise.get("aggregate", {})
        if agg:
            aggregates["pseudo_all_icp_direct"] = {
                "translation_error_mean_m": agg.get("pseudo_translation_error_mean_m"),
                "rotation_error_mean_deg": agg.get("pseudo_rotation_error_mean_deg"),
                "fitness_mean": agg.get("pseudo_fitness_mean"),
            }

    region_path = output_dir / "pairwise_region_analysis.json"
    if region_path.exists():
        region = json.loads(region_path.read_text(encoding="utf-8-sig"))
        front = region.get("aggregate", {}).get("front")
        if front:
            aggregates["pseudo_front_icp_direct"] = {
                "translation_error_mean_m": front.get("translation_error_mean_m"),
                "rotation_error_mean_deg": front.get("rotation_error_mean_deg"),
                "fitness_mean": front.get("fitness_mean"),
            }

    best_path = output_dir / "front_best_combination.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8-sig"))
        cfg = best.get("aggregate", {}).get("front_baseline_cfg05")
        if cfg:
            aggregates["pseudo_front_cfg05"] = {
                "translation_error_mean_m": cfg.get("translation_error_mean_m"),
                "rotation_error_mean_deg": cfg.get("rotation_error_mean_deg"),
                "fitness_mean": cfg.get("fitness_mean"),
                "rmse_mean": cfg.get("rmse_mean"),
            }
    return aggregates


def write_comparison_figure(output: dict, figure_path: Path) -> None:
    output_dir = figure_path.parent
    pseudo = load_existing_pseudolidar_aggregates(output_dir)
    lidar = output["aggregate"]

    rows = []
    if "icp_direct" in lidar and "all" in lidar["icp_direct"]:
        rows.append(("LiDAR all ICP", lidar["icp_direct"]["all"]))
    if "icp_direct" in lidar and "front" in lidar["icp_direct"]:
        rows.append(("LiDAR front ICP", lidar["icp_direct"]["front"]))
    if "pseudo_all_icp_direct" in pseudo:
        rows.append(("Pseudo all ICP", pseudo["pseudo_all_icp_direct"]))
    if "pseudo_front_icp_direct" in pseudo:
        rows.append(("Pseudo front ICP", pseudo["pseudo_front_icp_direct"]))
    if "pseudo_front_cfg05" in pseudo:
        rows.append(("Pseudo front cfg05", pseudo["pseudo_front_cfg05"]))

    if not rows:
        return

    labels = [row[0] for row in rows]
    trans = [row[1].get("translation_error_mean_m", np.nan) for row in rows]
    rot = [row[1].get("rotation_error_mean_deg", np.nan) for row in rows]
    fitness = [row[1].get("fitness_mean", np.nan) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].bar(labels, trans, color="#4C78A8")
    axes[0].set_title("Translation error mean [m]")
    axes[0].set_ylabel("m")
    axes[1].bar(labels, rot, color="#F58518")
    axes[1].set_title("Rotation error mean [deg]")
    axes[1].set_ylabel("deg")
    axes[2].bar(labels, fitness, color="#54A24B")
    axes[2].set_title("Fitness mean")
    axes[2].set_ylabel("ratio")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--max-corr-dist", type=float, default=1.5)
    parser.add_argument("--coarse-voxel", type=float, default=1.0)
    parser.add_argument("--global-corr", type=float, default=2.0)
    parser.add_argument("--fine-voxel", type=float, default=0.5)
    parser.add_argument("--icp-corr", type=float, default=1.5)
    parser.add_argument("--skip-global", action="store_true")
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8-sig"))
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    samples = sorted(run_summary["samples"], key=lambda item: item["index"])
    if len(samples) < 2:
        raise ValueError("At least two samples are required to evaluate registration.")

    pair_results = []
    flat_rows = []

    for pair_index in range(len(samples) - 1):
        source_meta = samples[pair_index]
        target_meta = samples[pair_index + 1]

        source_points_all = load_points(Path(source_meta["lidar_path"]))
        target_points_all = load_points(Path(target_meta["lidar_path"]))

        source_pose = pose_global_from_sample(nusc, source_meta["sample_token"])
        target_pose = pose_global_from_sample(nusc, target_meta["sample_token"])
        gt_rel = relative_transform(source_pose, target_pose)

        pair_info = {
            "pair_index": pair_index,
            "source_token": source_meta["sample_token"],
            "target_token": target_meta["sample_token"],
            "dt_s": float(target_meta["timestamp_s"] - source_meta["timestamp_s"]),
            "gt_relative_translation_m": translation_norm(gt_rel),
            "gt_relative_rotation_deg": rotation_angle_deg(gt_rel),
            "regions": {},
        }

        for region in REGIONS:
            source_points = filter_region(source_points_all, region)
            target_points = filter_region(target_points_all, region)
            pair_info["regions"][region] = {}

            direct = register_icp_direct(source_points, target_points, args.voxel_size, args.max_corr_dist)
            direct_metrics = metrics_from_result(
                direct,
                gt_rel,
                "icp_direct",
                region,
                source_points.shape[0],
                target_points.shape[0],
            )
            if direct_metrics is not None:
                pair_info["regions"][region]["icp_direct"] = direct_metrics
                flat_rows.append(
                    {
                        "pair_index": pair_index,
                        "source_token": source_meta["sample_token"],
                        "target_token": target_meta["sample_token"],
                        "dt_s": float(target_meta["timestamp_s"] - source_meta["timestamp_s"]),
                        **direct_metrics,
                    }
                )

            if not args.skip_global:
                global_pair = register_global_then_icp(
                    source_points,
                    target_points,
                    args.coarse_voxel,
                    args.global_corr,
                    args.fine_voxel,
                    args.icp_corr,
                )
                if global_pair is not None:
                    global_result, refined = global_pair
                    global_metrics = metrics_from_result(
                        refined,
                        gt_rel,
                        "global_then_icp",
                        region,
                        source_points.shape[0],
                        target_points.shape[0],
                        global_result=global_result,
                    )
                    if global_metrics is not None:
                        pair_info["regions"][region]["global_then_icp"] = global_metrics
                        flat_rows.append(
                            {
                                "pair_index": pair_index,
                                "source_token": source_meta["sample_token"],
                                "target_token": target_meta["sample_token"],
                                "dt_s": float(target_meta["timestamp_s"] - source_meta["timestamp_s"]),
                                **global_metrics,
                            }
                        )

        pair_results.append(pair_info)

    output = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "method_note": (
            "LIDAR_TOP point clouds are already saved in ego frame by the generation pipeline. "
            "This script registers consecutive LiDAR scans using the same Open3D point-to-plane ICP "
            "backend used for pseudo-LiDAR. Global+ICP uses FPFH RANSAC initialization when enabled."
        ),
        "regions": REGIONS,
        "pairs": pair_results,
        "aggregate": aggregate_rows(flat_rows),
    }

    output_path = args.output or args.run_summary.parent / "lidar_registration_baseline.json"
    csv_path = args.csv_output or output_path.with_suffix(".csv")
    figure_path = args.figure_output or output_path.with_name("lidar_vs_pseudolidar_registration_baseline.png")

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(flat_rows, csv_path)
    write_comparison_figure(output, figure_path)

    print(output_path)
    print(csv_path)
    print(figure_path)
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
