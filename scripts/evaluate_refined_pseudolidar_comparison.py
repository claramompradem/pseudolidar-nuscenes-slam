from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"No points found in point cloud: {path}")
    return points


def filter_common_support(
    points: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = (
        (ranges >= r_min)
        & (ranges <= r_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(target_pcd)

    distances = np.empty(source.shape[0], dtype=np.float32)
    for idx, point in enumerate(source.astype(np.float64)):
        _, _, dist2 = tree.search_knn_vector_3d(point, 1)
        distances[idx] = np.sqrt(dist2[0])
    return distances


def bev_cells(points: np.ndarray, resolution: float) -> set[tuple[int, int]]:
    cells = np.floor(points[:, :2] / resolution).astype(np.int32)
    return set(map(tuple, cells))


def bev_iou(points_a: np.ndarray, points_b: np.ndarray, resolution: float) -> float:
    cells_a = bev_cells(points_a, resolution)
    cells_b = bev_cells(points_b, resolution)
    if not cells_a and not cells_b:
        return 0.0
    return float(len(cells_a & cells_b) / len(cells_a | cells_b))


def evaluate_cloud(pseudo_points: np.ndarray, lidar_points: np.ndarray, bev_resolution: float) -> Dict[str, float]:
    pseudo_to_gt = nearest_neighbor_distances(pseudo_points, lidar_points)
    gt_to_pseudo = nearest_neighbor_distances(lidar_points, pseudo_points)
    return {
        "pseudo_points": int(pseudo_points.shape[0]),
        "lidar_points": int(lidar_points.shape[0]),
        "pseudo_to_gt_mean_nn": float(np.mean(pseudo_to_gt)),
        "pseudo_to_gt_median_nn": float(np.median(pseudo_to_gt)),
        "pseudo_to_gt_p90_nn": float(np.percentile(pseudo_to_gt, 90)),
        "pseudo_to_gt_p95_nn": float(np.percentile(pseudo_to_gt, 95)),
        "gt_to_pseudo_mean_nn": float(np.mean(gt_to_pseudo)),
        "gt_to_pseudo_median_nn": float(np.median(gt_to_pseudo)),
        "gt_to_pseudo_p90_nn": float(np.percentile(gt_to_pseudo, 90)),
        "gt_to_pseudo_p95_nn": float(np.percentile(gt_to_pseudo, 95)),
        "bev_iou": bev_iou(pseudo_points, lidar_points, bev_resolution),
    }


def aggregate_rows(rows: List[dict]) -> Dict[str, float]:
    numeric_keys = [
        "pseudo_points",
        "lidar_points",
        "pseudo_to_gt_mean_nn",
        "pseudo_to_gt_median_nn",
        "pseudo_to_gt_p90_nn",
        "pseudo_to_gt_p95_nn",
        "gt_to_pseudo_mean_nn",
        "gt_to_pseudo_median_nn",
        "gt_to_pseudo_p90_nn",
        "gt_to_pseudo_p95_nn",
        "bev_iou",
    ]
    return {f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in numeric_keys}


def sample_map(run_summary: dict) -> dict[int, dict]:
    return {int(item["index"]): item for item in run_summary["samples"]}


def validate_matching_sample(sample_index: int, depth_pro_item: dict, refined_item: dict) -> None:
    depth_pro_token = depth_pro_item.get("sample_token")
    refined_token = refined_item.get("sample_token")
    if depth_pro_token != refined_token:
        raise ValueError(
            "Mismatched sample tokens for index "
            f"{sample_index}: depth_pro={depth_pro_token}, refined={refined_token}"
        )


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path, aggregate: dict) -> None:
    labels = ["Depth Pro", "Refinada"]
    mean_nn = [
        aggregate["depth_pro"]["pseudo_to_gt_mean_nn_mean"],
        aggregate["refined"]["pseudo_to_gt_mean_nn_mean"],
    ]
    median_nn = [
        aggregate["depth_pro"]["pseudo_to_gt_median_nn_mean"],
        aggregate["refined"]["pseudo_to_gt_median_nn_mean"],
    ]
    iou = [aggregate["depth_pro"]["bev_iou_mean"], aggregate["refined"]["bev_iou_mean"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(labels, mean_nn, color=["tab:blue", "tab:orange"])
    axes[0].set_title("Mean NN pseudo -> LiDAR")
    axes[0].set_ylabel("m")
    axes[1].bar(labels, median_nn, color=["tab:blue", "tab:orange"])
    axes[1].set_title("Median NN pseudo -> LiDAR")
    axes[1].set_ylabel("m")
    axes[2].bar(labels, iou, color=["tab:blue", "tab:orange"])
    axes[2].set_title("BEV IoU")
    axes[2].set_ylabel("IoU")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native Depth Pro pseudo-LiDAR and refined pseudo-LiDAR against LIDAR_TOP."
    )
    parser.add_argument("--depth-pro-run-summary", type=Path, required=True)
    parser.add_argument("--refined-run-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=35.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument("--bev-resolution", type=float, default=0.5)
    args = parser.parse_args()

    depth_pro_summary = json.loads(args.depth_pro_run_summary.read_text(encoding="utf-8-sig"))
    refined_summary = json.loads(args.refined_run_summary.read_text(encoding="utf-8-sig"))

    depth_pro_samples = sample_map(depth_pro_summary)
    refined_samples = sample_map(refined_summary)
    common_indices = sorted(set(depth_pro_samples) & set(refined_samples))
    if not common_indices:
        raise ValueError("No common sample indices found between both run summaries.")

    support = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "bev_resolution": args.bev_resolution,
    }

    rows = []
    for sample_index in common_indices:
        items = {
            "depth_pro": depth_pro_samples[sample_index],
            "refined": refined_samples[sample_index],
        }
        validate_matching_sample(sample_index, items["depth_pro"], items["refined"])
        lidar_points = filter_common_support(
            load_points(Path(items["depth_pro"]["lidar_path"])),
            args.r_min,
            args.r_max,
            args.z_min,
            args.z_max,
        )
        for name, item in items.items():
            pseudo_points = filter_common_support(
                load_points(Path(item["pseudo_path"])),
                args.r_min,
                args.r_max,
                args.z_min,
                args.z_max,
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_token": item["sample_token"],
                    "depth_source": name,
                    **evaluate_cloud(pseudo_points, lidar_points, args.bev_resolution),
                }
            )

    aggregate = {}
    for name in ["depth_pro", "refined"]:
        aggregate[name] = aggregate_rows([row for row in rows if row["depth_source"] == name])

    result = {
        "scene_name": depth_pro_summary.get("scene_name"),
        "num_samples": len(common_indices),
        "support": support,
        "rows": rows,
        "aggregate": aggregate,
        "interpretation": (
            "This controlled comparison uses the same stored Depth Pro maps, same camera resolution, "
            "same nuScenes calibrations and same pseudo-LiDAR conversion. Differences therefore mainly "
            "come from the learned depth refinement step."
        ),
    }

    output_path = args.output or args.refined_run_summary.parent.parent / "refined_pseudolidar_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), rows)
    save_figure(output_path.with_suffix(".png"), aggregate)

    print(output_path)
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
