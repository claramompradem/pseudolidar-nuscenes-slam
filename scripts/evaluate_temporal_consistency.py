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


REGIONS = ["all", "front", "near", "mid", "far", "left", "right"]


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


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return points


def filter_region(points: np.ndarray, region: str) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ranges = np.linalg.norm(points[:, :3], axis=1)
    base = (ranges >= 2.0) & (ranges <= 35.0) & (z >= -2.0) & (z <= 2.5)

    if region == "all":
        mask = base
    elif region == "front":
        mask = base & (x > 0.0) & (np.abs(y) < 15.0)
    elif region == "near":
        mask = base & (ranges < 10.0)
    elif region == "mid":
        mask = base & (ranges >= 10.0) & (ranges < 20.0)
    elif region == "far":
        mask = base & (ranges >= 20.0) & (ranges <= 35.0)
    elif region == "left":
        mask = base & (y > 0.0)
    elif region == "right":
        mask = base & (y < 0.0)
    else:
        raise ValueError(f"Unknown region: {region}")
    return points[mask]


def nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(target_pcd)

    distances = np.empty(source.shape[0], dtype=np.float32)
    for idx, point in enumerate(source.astype(np.float64)):
        _, _, dist2 = tree.search_knn_vector_3d(point, 1)
        distances[idx] = math.sqrt(dist2[0])
    return distances


def evaluate_region(source_aligned: np.ndarray, target: np.ndarray, region: str) -> Dict[str, float] | None:
    source_region = filter_region(source_aligned, region)
    target_region = filter_region(target, region)
    if source_region.shape[0] == 0 or target_region.shape[0] == 0:
        return None

    distances = nearest_neighbor_distances(source_region, target_region)
    return {
        "num_source_points": int(source_region.shape[0]),
        "num_target_points": int(target_region.shape[0]),
        "mean_nn_m": float(np.mean(distances)),
        "median_nn_m": float(np.median(distances)),
        "p90_nn_m": float(np.percentile(distances, 90)),
        "overlap_ratio_0p5m": float(np.mean(distances <= 0.5)),
        "overlap_ratio_1p0m": float(np.mean(distances <= 1.0)),
    }


def aggregate_region(pair_results: List[dict], region: str) -> Dict[str, float] | None:
    valid = [pair["regions"][region] for pair in pair_results if pair["regions"].get(region) is not None]
    if not valid:
        return None

    return {
        "mean_nn_m": float(np.mean([item["mean_nn_m"] for item in valid])),
        "median_nn_m": float(np.mean([item["median_nn_m"] for item in valid])),
        "p90_nn_m": float(np.mean([item["p90_nn_m"] for item in valid])),
        "overlap_ratio_0p5m": float(np.mean([item["overlap_ratio_0p5m"] for item in valid])),
        "overlap_ratio_1p0m": float(np.mean([item["overlap_ratio_1p0m"] for item in valid])),
        "num_source_points_mean": float(np.mean([item["num_source_points"] for item in valid])),
        "num_target_points_mean": float(np.mean([item["num_target_points"] for item in valid])),
        "num_valid_pairs": int(len(valid)),
    }


def flatten_rows(pair_results: Iterable[dict]) -> List[dict]:
    rows = []
    for pair in pair_results:
        for region, metrics in pair["regions"].items():
            if metrics is None:
                rows.append(
                    {
                        "pair_index": pair["pair_index"],
                        "source_token": pair["source_token"],
                        "target_token": pair["target_token"],
                        "dt_s": pair["dt_s"],
                        "region": region,
                    }
                )
                continue
            rows.append(
                {
                    "pair_index": pair["pair_index"],
                    "source_token": pair["source_token"],
                    "target_token": pair["target_token"],
                    "dt_s": pair["dt_s"],
                    "region": region,
                    **metrics,
                }
            )
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "pair_index",
        "source_token",
        "target_token",
        "dt_s",
        "region",
        "num_source_points",
        "num_target_points",
        "mean_nn_m",
        "median_nn_m",
        "p90_nn_m",
        "overlap_ratio_0p5m",
        "overlap_ratio_1p0m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_region_figure(aggregate: dict, path: Path) -> None:
    valid_regions = [region for region in REGIONS if aggregate.get(region) is not None]
    mean_nn = [aggregate[region]["mean_nn_m"] for region in valid_regions]
    overlap = [aggregate[region]["overlap_ratio_1p0m"] for region in valid_regions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(valid_regions, mean_nn, color="#4C78A8")
    axes[0].set_title("Temporal consistency: mean NN distance")
    axes[0].set_ylabel("m")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(valid_regions, overlap, color="#54A24B")
    axes[1].set_title("Temporal consistency: overlap within 1.0 m")
    axes[1].set_ylabel("ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--figure-output", type=Path, default=None)
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8-sig"))
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    samples = sorted(run_summary["samples"], key=lambda item: item["index"])
    if len(samples) < 2:
        raise ValueError("At least two samples are required to evaluate temporal consistency.")

    pair_results = []
    for pair_index in range(len(samples) - 1):
        source_meta = samples[pair_index]
        target_meta = samples[pair_index + 1]

        source_points = load_points(Path(source_meta["pseudo_path"]))
        target_points = load_points(Path(target_meta["pseudo_path"]))

        source_pose = pose_global_from_sample(nusc, source_meta["sample_token"])
        target_pose = pose_global_from_sample(nusc, target_meta["sample_token"])
        gt_source_to_target = relative_transform(source_pose, target_pose)
        source_aligned = transform_points(source_points, gt_source_to_target)

        pair_info = {
            "pair_index": pair_index,
            "source_token": source_meta["sample_token"],
            "target_token": target_meta["sample_token"],
            "dt_s": float(target_meta["timestamp_s"] - source_meta["timestamp_s"]),
            "regions": {},
        }

        for region in REGIONS:
            pair_info["regions"][region] = evaluate_region(source_aligned, target_points, region)

        pair_results.append(pair_info)

    aggregate = {region: aggregate_region(pair_results, region) for region in REGIONS}
    output = {
        "scene_name": run_summary["scene_name"],
        "num_pairs": len(pair_results),
        "method": (
            "Source pseudo-LiDAR is transformed into the next ego frame using the nuScenes "
            "ground-truth relative pose, then compared to the target pseudo-LiDAR with nearest neighbors."
        ),
        "regions": REGIONS,
        "pairs": pair_results,
        "aggregate": aggregate,
    }

    output_path = args.output or args.run_summary.parent / "temporal_consistency_metrics.json"
    csv_path = args.csv_output or output_path.with_name("temporal_consistency_by_region.csv")
    figure_path = args.figure_output or output_path.with_name("temporal_consistency_by_region.png")

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(flatten_rows(pair_results), csv_path)
    write_region_figure(aggregate, figure_path)

    print(output_path)
    print(csv_path)
    print(figure_path)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
