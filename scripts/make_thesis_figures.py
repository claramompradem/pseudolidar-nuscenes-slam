from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


DEFAULT_SUPPORT = {
    "r_min": 2.0,
    "r_max": 35.0,
    "z_min": -2.0,
    "z_max": 2.5,
    "bev_resolution": 0.5,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def infer_run_root(repo_root: Path, manifest: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()

    scene_name = manifest["scene_name"]
    expected_num = int(manifest.get("num_selected") or manifest.get("num_requested") or 0)
    scene_dir = repo_root / "outputs" / scene_name
    candidates = []
    for candidate in sorted(scene_dir.glob("refined_pseudolidar*")):
        index_path = candidate / "comparison_index.json"
        if not index_path.exists():
            continue
        index = load_json(index_path)
        run_summaries = index.get("run_summaries", {})
        if "depth_pro" not in run_summaries or "refined" not in run_summaries:
            continue
        match_num = int(index.get("num_samples", -1)) == expected_num
        is_rgb = bool(index.get("use_rgb", False))
        has_first30 = "first30" in candidate.name
        candidates.append((match_num, is_rgb, has_first30, candidate))

    if not candidates:
        raise FileNotFoundError(f"No refined_pseudolidar run found under {scene_dir}")

    candidates.sort(key=lambda item: (item[0], item[1], item[2], str(item[3])))
    return candidates[-1][3].resolve()


def sample_by_token(run_summary: dict) -> dict[str, dict]:
    return {sample["sample_token"]: sample for sample in run_summary["samples"]}


def sample_by_index(run_summary: dict) -> dict[int, dict]:
    return {int(sample["index"]): sample for sample in run_summary["samples"]}


def choose_representative_sample(
    sample_token: str | None,
    comparison_path: Path,
    depth_pro_summary: dict,
) -> tuple[str, str]:
    if sample_token:
        return sample_token, "provided by --sample-token"

    if comparison_path.exists():
        comparison = load_json(comparison_path)
        rows = [
            row
            for row in comparison.get("rows", [])
            if row.get("depth_source") == "depth_pro"
            and row.get("pseudo_to_gt_mean_nn") is not None
        ]
        if rows:
            rows = sorted(rows, key=lambda row: float(row["pseudo_to_gt_mean_nn"]))
            row = rows[len(rows) // 2]
            return row["sample_token"], "median depth_pro pseudo_to_gt_mean_nn"

    first = sorted(depth_pro_summary["samples"], key=lambda item: int(item["index"]))[0]
    return first["sample_token"], "first sample in run_summary"


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"No points found in point cloud: {path}")
    return points


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def sample_arrays(
    points: np.ndarray,
    values: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, values
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], values[idx]


def filter_common_support(points: np.ndarray, support: dict) -> np.ndarray:
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = (
        (ranges >= float(support["r_min"]))
        & (ranges <= float(support["r_max"]))
        & (points[:, 2] >= float(support["z_min"]))
        & (points[:, 2] <= float(support["z_max"]))
    )
    return points[mask]


def filter_horizontal_regions(points: np.ndarray, support: dict) -> np.ndarray:
    horizontal = np.linalg.norm(points[:, :2], axis=1)
    mask = (
        (horizontal <= float(support["r_max"]))
        & (points[:, 2] >= float(support["z_min"]))
        & (points[:, 2] <= float(support["z_max"]))
    )
    return points[mask]


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


def selector_for_strategy(strategy: str) -> Callable[[np.ndarray], np.ndarray]:
    if "narrow" in strategy:
        return select_front_narrow
    if "front" in strategy:
        return select_front_baseline
    return lambda points: filter_common_support(points, DEFAULT_SUPPORT)


def nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target.astype(np.float32))
    try:
        distances, _ = tree.query(source.astype(np.float32), k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(source.astype(np.float32), k=1)
    return distances.astype(np.float32)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (rotation @ points.T).T + translation


def style_bev_axis(ax: plt.Axes, title: str) -> None:
    if title:
        ax.set_title(title)
    ax.set_xlabel("x ego (m)")
    ax.set_ylabel("y ego (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.5)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_near_mid_far(
    base_points: np.ndarray,
    output_path: Path,
    support: dict,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    points = filter_horizontal_regions(base_points, support)
    horizontal = np.linalg.norm(points[:, :2], axis=1)
    bands = [
        ("cerca <10 m", horizontal < 10.0, "tab:blue"),
        ("medio 10-20 m", (horizontal >= 10.0) & (horizontal < 20.0), "tab:orange"),
        ("lejos 20-35 m", (horizontal >= 20.0) & (horizontal <= 35.0), "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for label, mask, color in bands:
        band_points = sample_points(points[mask], max_points // 3, rng)
        ax.scatter(band_points[:, 0], band_points[:, 1], s=0.7, c=color, alpha=0.75, label=label, linewidths=0)
    ax.scatter([0], [0], marker="x", c="black", s=35)
    style_bev_axis(ax, "Pseudo-LiDAR base por rango horizontal")
    ax.legend(loc="upper right", frameon=True, markerscale=5)
    ax.set_xlim(-36, 36)
    ax.set_ylim(-36, 36)
    save_figure(fig, output_path)


def figure_trajectory(
    base_trajectory_path: Path,
    refined_trajectory_path: Path,
    output_path: Path,
    strategy: str,
) -> None:
    base = load_json(base_trajectory_path)
    refined = load_json(refined_trajectory_path)
    if strategy not in base["trajectories"] or strategy not in refined["trajectories"]:
        available = sorted(set(base["trajectories"]) & set(refined["trajectories"]))
        raise KeyError(f"Strategy {strategy!r} not found. Available common strategies: {available}")

    base_traj = base["trajectories"][strategy]
    refined_traj = refined["trajectories"][strategy]
    gt_xy = np.asarray(base_traj.get("gt_trajectory_xy") or base.get("gt_trajectory_xy"), dtype=np.float32)
    base_xy = np.asarray(base_traj["trajectory_xy"], dtype=np.float32)
    refined_xy = np.asarray(refined_traj["trajectory_xy"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-o", color="black", linewidth=2.0, markersize=2.8, label="referencia nuScenes")
    ax.plot(base_xy[:, 0], base_xy[:, 1], "-o", color="tab:blue", linewidth=1.8, markersize=2.4, label="Depth Pro base")
    ax.plot(refined_xy[:, 0], refined_xy[:, 1], "-o", color="tab:orange", linewidth=1.8, markersize=2.4, label="Depth Pro+RGB refinada")
    ax.scatter(gt_xy[0, 0], gt_xy[0, 1], marker="s", s=45, color="black", label="inicio")

    style_bev_axis(ax, "")
    ax.legend(loc="best", frameon=True)
    save_figure(fig, output_path)


def figure_overlay(
    base_points: np.ndarray,
    lidar_points: np.ndarray,
    output_path: Path,
    support: dict,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    base = sample_points(filter_common_support(base_points, support), max_points, rng)
    lidar = sample_points(filter_common_support(lidar_points, support), max_points, rng)

    fig = plt.figure(figsize=(13, 5.5))
    ax_bev = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    ax_bev.scatter(lidar[:, 0], lidar[:, 1], s=0.8, c="black", alpha=0.45, label="LIDAR_TOP", linewidths=0)
    ax_bev.scatter(base[:, 0], base[:, 1], s=0.8, c="tab:orange", alpha=0.55, label="base pseudo-LiDAR", linewidths=0)
    style_bev_axis(ax_bev, "Superposicion BEV")
    ax_bev.legend(loc="upper right", frameon=True, markerscale=5)

    ax_3d.scatter(lidar[:, 0], lidar[:, 1], lidar[:, 2], s=0.5, c="black", alpha=0.35, linewidths=0)
    ax_3d.scatter(base[:, 0], base[:, 1], base[:, 2], s=0.5, c="tab:orange", alpha=0.55, linewidths=0)
    ax_3d.set_title("Superposicion en perspectiva")
    ax_3d.set_xlabel("x ego (m)")
    ax_3d.set_ylabel("y ego (m)")
    ax_3d.set_zlabel("z ego (m)")
    ax_3d.view_init(elev=24, azim=-58)
    ax_3d.set_box_aspect((1.5, 1.5, 0.45))
    save_figure(fig, output_path)


def figure_pseudolidar_lidar_triptych(
    base_points: np.ndarray,
    lidar_points: np.ndarray,
    output_path: Path,
    support: dict,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    base = sample_points(filter_common_support(base_points, support), max_points, rng)
    lidar = sample_points(filter_common_support(lidar_points, support), max_points, rng)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True, sharex=True, sharey=True)
    panels = [
        ("Pseudo-LiDAR", [(base, "tab:orange", 0.9, 0.62, "pseudo-LiDAR")]),
        ("LIDAR_TOP", [(lidar, "black", 0.9, 0.55, "LIDAR_TOP")]),
        (
            "Superposicion",
            [
                (lidar, "black", 0.75, 0.38, "LIDAR_TOP"),
                (base, "tab:orange", 0.75, 0.55, "pseudo-LiDAR"),
            ],
        ),
    ]

    for ax, (title, clouds) in zip(axes, panels):
        for points, color, size, alpha, label in clouds:
            ax.scatter(points[:, 0], points[:, 1], s=size, c=color, alpha=alpha, label=label, linewidths=0)
        ax.scatter([0], [0], marker="x", c="tab:red", s=38, linewidths=1.4)
        style_bev_axis(ax, title)
        ax.set_xlim(-36, 36)
        ax.set_ylim(-36, 36)
        ax.legend(loc="upper right", frameon=True, markerscale=4)

    for ax in axes[1:]:
        ax.set_ylabel("")
    save_figure(fig, output_path)


def figure_nn_error(
    base_points: np.ndarray,
    refined_points: np.ndarray,
    lidar_points: np.ndarray,
    output_path: Path,
    support: dict,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    base = filter_common_support(base_points, support)
    refined = filter_common_support(refined_points, support)
    lidar = filter_common_support(lidar_points, support)

    base_dist = nearest_neighbor_distances(base, lidar)
    refined_dist = nearest_neighbor_distances(refined, lidar)
    vmax = max(0.5, float(np.percentile(np.concatenate([base_dist, refined_dist]), 95)))

    base, base_dist = sample_arrays(base, base_dist, max_points, rng)
    refined, refined_dist = sample_arrays(refined, refined_dist, max_points, rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    scatters = []
    for ax, title, points, distances in [
        (axes[0], "Base Depth Pro", base, base_dist),
        (axes[1], "Refinada Depth Pro+RGB", refined, refined_dist),
    ]:
        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=np.clip(distances, 0.0, vmax),
            s=0.9,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            alpha=0.85,
            linewidths=0,
        )
        scatters.append(sc)
        style_bev_axis(ax, f"{title}\nNN media={np.mean(distances):.2f} m, p95={np.percentile(distances, 95):.2f} m")
    cbar = fig.colorbar(scatters[-1], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(f"Distancia NN a LIDAR_TOP (m, recortada en {vmax:.2f})")
    save_figure(fig, output_path)


def best_and_worst_pairs(accumulated_path: Path, strategy: str) -> tuple[dict, dict]:
    data = load_json(accumulated_path)
    pairs = [
        pair
        for pair in data["pairwise_results"][strategy]
        if pair.get("metrics") is not None and pair.get("estimated_transform") is not None
    ]
    if not pairs:
        raise ValueError(f"No valid pairs with estimated_transform found for {strategy}")
    good = min(pairs, key=lambda pair: float(pair["metrics"]["translation_error_m"]))
    bad = max(pairs, key=lambda pair: float(pair["metrics"]["translation_error_m"]))
    return good, bad


def figure_registration_good_vs_bad(
    run_summary: dict,
    accumulated_path: Path,
    output_path: Path,
    strategy: str,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    samples = sample_by_token(run_summary)
    selector = selector_for_strategy(strategy)
    good, bad = best_and_worst_pairs(accumulated_path, strategy)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, label, pair in [(axes[0], "Registro bueno", good), (axes[1], "Registro malo", bad)]:
        src_meta = samples[pair["source_token"]]
        tgt_meta = samples[pair["target_token"]]
        source = selector(load_points(Path(src_meta["pseudo_path"])))
        target = selector(load_points(Path(tgt_meta["pseudo_path"])))
        transform = np.asarray(pair["estimated_transform"], dtype=np.float64)
        source_aligned = apply_transform(source, transform).astype(np.float32)

        source_aligned = sample_points(source_aligned, max_points // 2, rng)
        target = sample_points(target, max_points // 2, rng)

        ax.scatter(target[:, 0], target[:, 1], s=0.9, c="tab:green", alpha=0.55, label="objetivo", linewidths=0)
        ax.scatter(source_aligned[:, 0], source_aligned[:, 1], s=0.9, c="tab:purple", alpha=0.55, label="fuente alineada", linewidths=0)
        err_t = pair["metrics"]["translation_error_m"]
        err_r = pair["metrics"]["rotation_error_deg"]
        style_bev_axis(ax, f"{label}\npar {pair['pair_index']}, error trasl.={err_t:.2f} m, error rot.={err_r:.2f} deg")
        ax.legend(loc="best", frameon=True, markerscale=5)
    save_figure(fig, output_path)


def print_located_paths(paths: dict[str, Path]) -> None:
    print("Located input/output paths:")
    for name, path in paths.items():
        status = "OK" if path.exists() or name == "figures_dir" else "MISSING"
        print(f"  [{status}] {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create thesis figures from existing pseudo-LiDAR outputs.")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/scene-0061_first30.json"))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--sample-token", type=str, default=None)
    parser.add_argument("--trajectory-strategy", type=str, default="front_baseline_cfg05")
    parser.add_argument("--registration-strategy", type=str, default="front_narrow_cfg05")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    manifest_path = resolve_path(args.manifest, repo_root)
    manifest = load_json(manifest_path)
    run_root = infer_run_root(repo_root, manifest, args.run_root)
    figures_dir = run_root / "figures" if args.figures_dir is None else resolve_path(args.figures_dir, repo_root)

    comparison_index_path = run_root / "comparison_index.json"
    comparison_path = run_root / "refined_pseudolidar_comparison.json"
    comparison_index = load_json(comparison_index_path)
    depth_pro_summary_path = resolve_path(comparison_index["run_summaries"]["depth_pro"], repo_root)
    refined_summary_path = resolve_path(comparison_index["run_summaries"]["refined"], repo_root)
    depth_pro_summary = load_json(depth_pro_summary_path)
    refined_summary = load_json(refined_summary_path)

    base_trajectory_path = depth_pro_summary_path.parent / "accumulated_trajectory_comparison.json"
    refined_trajectory_path = refined_summary_path.parent / "accumulated_trajectory_comparison.json"
    base_pairwise_path = depth_pro_summary_path.parent / "pairwise_registration_metrics.json"
    refined_pairwise_path = refined_summary_path.parent / "pairwise_registration_metrics.json"
    base_front_path = depth_pro_summary_path.parent / "front_best_combination.json"
    refined_front_path = refined_summary_path.parent / "front_best_combination.json"

    support = DEFAULT_SUPPORT.copy()
    if comparison_path.exists():
        support.update(load_json(comparison_path).get("support", {}))

    sample_token, sample_reason = choose_representative_sample(
        args.sample_token,
        comparison_path,
        depth_pro_summary,
    )
    base_samples = sample_by_token(depth_pro_summary)
    refined_samples = sample_by_token(refined_summary)
    if sample_token not in base_samples or sample_token not in refined_samples:
        raise KeyError(f"Sample token {sample_token} not found in both run summaries.")

    base_sample = base_samples[sample_token]
    refined_sample = refined_samples[sample_token]
    base_pseudo_path = Path(base_sample["pseudo_path"])
    refined_pseudo_path = Path(refined_sample["pseudo_path"])
    lidar_path = Path(base_sample["lidar_path"])

    located_paths = {
        "repo_root": repo_root,
        "manifest": manifest_path,
        "run_root": run_root,
        "comparison_index": comparison_index_path,
        "comparison_metrics": comparison_path,
        "depth_pro_run_summary": depth_pro_summary_path,
        "refined_run_summary": refined_summary_path,
        "depth_pro_trajectory": base_trajectory_path,
        "refined_trajectory": refined_trajectory_path,
        "depth_pro_pairwise": base_pairwise_path,
        "refined_pairwise": refined_pairwise_path,
        "depth_pro_front_best": base_front_path,
        "refined_front_best": refined_front_path,
        "sample_base_pseudo": base_pseudo_path,
        "sample_refined_pseudo": refined_pseudo_path,
        "sample_lidar_top": lidar_path,
        "figures_dir": figures_dir,
    }
    print_located_paths(located_paths)
    print(f"Representative sample: {sample_token} ({sample_reason})")
    print(f"Support: {support}")

    rng = np.random.default_rng(args.seed)
    base_points = load_points(base_pseudo_path)
    refined_points = load_points(refined_pseudo_path)
    lidar_points = load_points(lidar_path)

    outputs = {
        "near_mid_far_regions": figures_dir / "near_mid_far_regions.png",
        "accumulated_trajectory": figures_dir / "accumulated_trajectory_refined_vs_base.png",
        "overlay": figures_dir / "pseudolidar_vs_lidar_top_overlay.png",
        "triptych": figures_dir / "pseudolidar_lidar_top_triptych.png",
        "nn_error": figures_dir / "refined_vs_base_nn_error.png",
        "registration_good_vs_bad": figures_dir / "registration_good_vs_bad.png",
    }

    figure_near_mid_far(base_points, outputs["near_mid_far_regions"], support, rng, args.max_points)
    figure_trajectory(base_trajectory_path, refined_trajectory_path, outputs["accumulated_trajectory"], args.trajectory_strategy)
    figure_overlay(base_points, lidar_points, outputs["overlay"], support, rng, args.max_points)
    figure_pseudolidar_lidar_triptych(base_points, lidar_points, outputs["triptych"], support, rng, args.max_points)
    figure_nn_error(base_points, refined_points, lidar_points, outputs["nn_error"], support, rng, args.max_points)
    figure_registration_good_vs_bad(
        refined_summary,
        refined_trajectory_path,
        outputs["registration_good_vs_bad"],
        args.registration_strategy,
        rng,
        args.max_points,
    )

    print("Generated figures:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
