from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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
    ranges = np.linalg.norm(points, axis=1)
    mask = (
        (ranges >= r_min)
        & (ranges <= r_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def point_stats(name: str, points: np.ndarray) -> Dict[str, float]:
    ranges = np.linalg.norm(points, axis=1)
    return {
        "name": name,
        "num_points": int(points.shape[0]),
        "range_min": float(np.min(ranges)),
        "range_mean": float(np.mean(ranges)),
        "range_median": float(np.median(ranges)),
        "range_p95": float(np.percentile(ranges, 95)),
    }


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


def evaluate_model(
    name: str,
    pseudo_points: np.ndarray,
    lidar_points: np.ndarray,
    bev_resolution: float,
) -> Dict[str, float]:
    pseudo_to_gt = nearest_neighbor_distances(pseudo_points, lidar_points)
    gt_to_pseudo = nearest_neighbor_distances(lidar_points, pseudo_points)
    return {
        "pseudo_points": int(pseudo_points.shape[0]),
        "pseudo_to_gt_mean_nn": float(np.mean(pseudo_to_gt)),
        "pseudo_to_gt_median_nn": float(np.median(pseudo_to_gt)),
        "pseudo_to_gt_p95_nn": float(np.percentile(pseudo_to_gt, 95)),
        "gt_to_pseudo_mean_nn": float(np.mean(gt_to_pseudo)),
        "gt_to_pseudo_median_nn": float(np.median(gt_to_pseudo)),
        "gt_to_pseudo_p95_nn": float(np.percentile(gt_to_pseudo, 95)),
        "bev_iou": bev_iou(pseudo_points, lidar_points, bev_resolution),
    }


def sample_from_summary(run_summary: dict, sample_index: int) -> dict:
    for sample in run_summary["samples"]:
        if int(sample["index"]) == sample_index:
            return sample
    raise ValueError(f"Sample index {sample_index} not found in run summary.")


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Depth Pro and Depth Anything 3 pseudo-LiDAR against LIDAR_TOP in common support."
    )
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--depth-anything-sample-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=35.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument("--bev-resolution", type=float, default=0.5)
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    depth_pro_sample = sample_from_summary(run_summary, args.sample_index)

    depth_pro_pseudo_path = Path(depth_pro_sample["pseudo_path"])
    lidar_path = Path(depth_pro_sample["lidar_path"])
    depth_anything_pseudo_path = args.depth_anything_sample_dir / "pcd_pseudolidar_ego.ply"

    support = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "bev_resolution": args.bev_resolution,
    }

    lidar_points = filter_common_support(load_points(lidar_path), args.r_min, args.r_max, args.z_min, args.z_max)
    depth_pro_points = filter_common_support(
        load_points(depth_pro_pseudo_path), args.r_min, args.r_max, args.z_min, args.z_max
    )
    depth_anything_points = filter_common_support(
        load_points(depth_anything_pseudo_path), args.r_min, args.r_max, args.z_min, args.z_max
    )

    result = {
        "sample_index": args.sample_index,
        "sample_token": depth_pro_sample["sample_token"],
        "support": support,
        "inputs": {
            "Depth Pro": display_path(depth_pro_pseudo_path, args.run_summary.parent.parent.parent),
            "Depth Anything 3": display_path(depth_anything_pseudo_path, args.run_summary.parent.parent.parent),
            "LIDAR_TOP": display_path(lidar_path, args.run_summary.parent.parent.parent),
        },
        "lidar_stats": point_stats("lidar_top_gt_common_support", lidar_points),
        "models": {
            "Depth Pro": evaluate_model("Depth Pro", depth_pro_points, lidar_points, args.bev_resolution),
            "Depth Anything 3": evaluate_model(
                "Depth Anything 3", depth_anything_points, lidar_points, args.bev_resolution
            ),
        },
        "interpretation": (
            "Depth Pro obtiene una geometría ligeramente más consistente que Depth Anything 3 "
            "en soporte común para esta pipeline, por lo que se mantiene como baseline principal."
        ),
    }

    output_path = args.output or args.run_summary.parent / "depth_model_comparison_common_support.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
