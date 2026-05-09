from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from pyquaternion import Quaternion


CFG_05 = {
    "coarse_voxel": 0.8,
    "global_corr": 2.0,
    "fine_voxel": 0.4,
    "icp_corr": 1.5,
}


def load_pcd(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64)
    if colors.size == 0:
        colors = None
    return points, colors


def sample_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    max_points: int = 20000,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    sampled_colors = None if colors is None else colors[idx]
    return points[idx], sampled_colors


def rgb_strings(colors: np.ndarray | None, fallback: str) -> List[str] | str:
    if colors is None:
        return fallback
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in colors_uint8]


def point_trace(
    name: str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    color: str = "rgb(31,119,180)",
    size: float = 1.6,
    opacity: float = 0.85,
    max_points: int = 20000,
) -> go.Scatter3d:
    points, colors = sample_points(points, colors, max_points=max_points)
    marker_color = rgb_strings(colors, color)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker={
            "size": size,
            "color": marker_color,
            "opacity": opacity,
        },
    )


def set_scene_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        width=1100,
        height=780,
        legend={"itemsizing": "constant"},
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": -1.7, "y": -1.9, "z": 1.15},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )


def write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def select_front(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = (ranges >= 2.0) & (ranges <= 35.0) & (z >= -2.0) & (z <= 2.5)
    mask &= (x > 0.0) & (np.abs(y) < 15.0)
    return points[mask]


def register_icp_only(source_points: np.ndarray, target_points: np.ndarray):
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


def register_cfg05(source_points: np.ndarray, target_points: np.ndarray):
    source_down, source_fpfh = preprocess_fpfh(source_points, CFG_05["coarse_voxel"])
    target_down, target_fpfh = preprocess_fpfh(target_points, CFG_05["coarse_voxel"])
    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=CFG_05["global_corr"],
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(CFG_05["global_corr"]),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500),
    )

    source_fine = prepare_pcd(source_points, CFG_05["fine_voxel"])
    target_fine = prepare_pcd(target_points, CFG_05["fine_voxel"])
    return o3d.pipelines.registration.registration_icp(
        source_fine,
        target_fine,
        CFG_05["icp_corr"],
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def rotation_angle_deg(transform: np.ndarray) -> float:
    trace = np.clip((np.trace(transform[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(transform[:3, 3]))


def create_phase_figures(run_summary: Dict, output_dir: Path) -> List[Path]:
    samples = sorted(run_summary["samples"], key=lambda item: item["index"])
    if len(samples) < 2:
        raise ValueError("At least two processed samples are required for registration visualizations.")

    first = samples[0]
    second = samples[1]

    ring_points, ring_colors = load_pcd(Path(first["ring_path"]))
    pseudo_points, pseudo_colors = load_pcd(Path(first["pseudo_path"]))
    lidar_points, _ = load_pcd(Path(first["lidar_path"]))

    written: List[Path] = []

    fig = go.Figure()
    fig.add_trace(point_trace("fused dense ring from 6 cameras", ring_points, ring_colors, max_points=35000, size=1.4))
    set_scene_layout(fig, "Phase 1 - Dense 3D fusion from the six cameras")
    path = output_dir / "phase_01_dense_ring_6cams.html"
    write_figure(fig, path)
    written.append(path)

    fig = go.Figure()
    fig.add_trace(point_trace("pseudo-LiDAR", pseudo_points, pseudo_colors, color="rgb(31,119,180)", max_points=25000))
    fig.add_trace(point_trace("LIDAR_TOP ground truth", lidar_points, None, color="rgb(220,40,40)", max_points=25000, opacity=0.55))
    set_scene_layout(fig, "Phase 2 - Pseudo-LiDAR compared with LIDAR_TOP")
    path = output_dir / "phase_02_pseudolidar_vs_lidar_gt.html"
    write_figure(fig, path)
    written.append(path)

    fig = go.Figure()
    fig.add_trace(point_trace("dense ring", ring_points, ring_colors, max_points=25000, opacity=0.55, size=1.2))
    fig.add_trace(point_trace("pseudo-LiDAR", pseudo_points, None, color="rgb(31,119,180)", max_points=22000, opacity=0.9, size=1.8))
    fig.add_trace(point_trace("LIDAR_TOP", lidar_points, None, color="rgb(220,40,40)", max_points=22000, opacity=0.45, size=1.5))
    set_scene_layout(fig, "Phase 3 - Dense cloud, pseudo-LiDAR and real LiDAR")
    path = output_dir / "phase_03_ring_pseudo_lidar_overlay.html"
    write_figure(fig, path)
    written.append(path)

    src_points, _ = load_pcd(Path(first["pseudo_path"]))
    tgt_points, _ = load_pcd(Path(second["pseudo_path"]))
    src_front = select_front(src_points)
    tgt_front = select_front(tgt_points)

    fig = go.Figure()
    fig.add_trace(point_trace("full pseudo-LiDAR", src_points, None, color="rgb(150,150,150)", max_points=26000, opacity=0.45, size=1.4))
    fig.add_trace(point_trace("selected front region", src_front, None, color="rgb(31,119,180)", max_points=14000, opacity=0.95, size=2.0))
    set_scene_layout(fig, "Phase 3B - Full pseudo-LiDAR ring and selected front region")
    path = output_dir / "phase_03b_full_ring_vs_front_subset.html"
    write_figure(fig, path)
    written.append(path)

    all_result = register_icp_only(src_points, tgt_points)
    src_all_aligned = transform_points(src_points, all_result.transformation)

    fig = go.Figure()
    fig.add_trace(point_trace("source full ring after ICP", src_all_aligned, None, color="rgb(150,150,150)", max_points=22000, opacity=0.55, size=1.4))
    fig.add_trace(point_trace("target full ring", tgt_points, None, color="rgb(255,127,14)", max_points=22000, opacity=0.55, size=1.4))
    set_scene_layout(
        fig,
        f"Phase 3C - Full-ring registration after ICP "
        f"(fitness={all_result.fitness:.3f}, rmse={all_result.inlier_rmse:.3f})",
    )
    path = output_dir / "phase_03c_full_ring_after_icp.html"
    write_figure(fig, path)
    written.append(path)

    result = register_cfg05(src_front, tgt_front)
    src_aligned = transform_points(src_front, result.transformation)

    fig = go.Figure()
    fig.add_trace(point_trace("source front before registration", src_front, None, color="rgb(31,119,180)", max_points=18000, opacity=0.75))
    fig.add_trace(point_trace("target front", tgt_front, None, color="rgb(255,127,14)", max_points=18000, opacity=0.75))
    set_scene_layout(fig, "Phase 4A - Consecutive pseudo-LiDAR fronts before registration")
    path = output_dir / "phase_04a_front_pair_before_registration.html"
    write_figure(fig, path)
    written.append(path)

    fig = go.Figure()
    fig.add_trace(point_trace("source front after cfg_05", src_aligned, None, color="rgb(31,119,180)", max_points=18000, opacity=0.75))
    fig.add_trace(point_trace("target front", tgt_front, None, color="rgb(255,127,14)", max_points=18000, opacity=0.75))
    set_scene_layout(
        fig,
        f"Phase 4B - Consecutive pseudo-LiDAR fronts after cfg_05 "
        f"(fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.3f})",
    )
    path = output_dir / "phase_04b_front_pair_after_cfg05.html"
    write_figure(fig, path)
    written.append(path)

    metadata = {
        "sample_token": first["sample_token"],
        "target_sample_token": second["sample_token"],
        "registration": {
            "strategy": "front baseline + cfg_05",
            "fitness": float(result.fitness),
            "rmse": float(result.inlier_rmse),
            "estimated_translation_m": translation_norm(result.transformation),
            "estimated_rotation_deg": rotation_angle_deg(result.transformation),
        },
        "figures": [str(path) for path in written],
    }
    (output_dir / "phase_visualizations_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return written


def write_index(paths: Iterable[Path], output_dir: Path) -> Path:
    rows = "\n".join(
        f'<li><a href="{path.name}">{path.stem.replace("_", " ")}</a></li>' for path in paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Point cloud phase visualizations</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; line-height: 1.45; }}
    code {{ background: #f2f2f2; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Point cloud phase visualizations</h1>
  <p>Interactive 3D HTML visualizations generated from the pseudo-LiDAR experiment.</p>
  <ul>
    {rows}
  </ul>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8-sig"))
    output_dir = args.output_dir or args.run_summary.parent / "pointcloud_phase_visualizations"
    written = create_phase_figures(run_summary, output_dir)
    index_path = write_index(written, output_dir)

    print(index_path)
    print(json.dumps({"num_figures": len(written), "figures": [str(path) for path in written]}, indent=2))


if __name__ == "__main__":
    main()
