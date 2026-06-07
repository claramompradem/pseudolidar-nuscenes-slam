from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import open3d as o3d
import torch
from nuscenes.nuscenes import NuScenes

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_pseudolidar_manifest import (  # noqa: E402
    CAMERA_CHANNELS,
    build_outputs_for_sample,
    load_lidar_top_pcd,
)
from train_depth_refiner import SmallDepthUNet  # noqa: E402


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    return device


def load_refiner(checkpoint_path: Path, device: torch.device) -> tuple[SmallDepthUNet, float, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", 1))
    model = SmallDepthUNet(base_channels=int(checkpoint["base_channels"]), in_channels=input_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    use_rgb = bool(checkpoint.get("use_rgb", input_channels == 4))
    return model, float(checkpoint["max_depth"]), use_rgb


def refine_depth(
    depth_pro: np.ndarray,
    rgb: np.ndarray,
    model: SmallDepthUNet,
    max_depth: float,
    use_rgb: bool,
    device: torch.device,
) -> np.ndarray:
    input_channels = [np.clip(depth_pro / max_depth, 0.0, 1.0).astype(np.float32)[None]]
    if use_rgb:
        rgb_chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        input_channels.append(rgb_chw.astype(np.float32))
    model_input = np.concatenate(input_channels, axis=0)
    with torch.inference_mode():
        tensor = torch.from_numpy(model_input)[None].to(device)
        refined = model(tensor).squeeze().cpu().numpy().astype(np.float32) * max_depth
    return np.clip(refined, 0.0, max_depth).astype(np.float32)


def group_entries_by_sample(index: dict, max_samples: int | None) -> List[dict]:
    grouped: Dict[str, dict] = {}
    for entry in index["entries"]:
        token = entry["sample_token"]
        grouped.setdefault(
            token,
            {
                "sample_token": token,
                "index": int(entry["sample_index"]),
                "channels": {},
            },
        )
        grouped[token]["channels"][entry["channel"]] = entry

    samples = sorted(grouped.values(), key=lambda item: item["index"])
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def build_results_for_sample(
    nusc: NuScenes,
    dataset_dir: Path,
    sample_meta: dict,
    model: SmallDepthUNet,
    max_depth: float,
    use_rgb: bool,
    device: torch.device,
) -> tuple[dict, dict]:
    sample = nusc.get("sample", sample_meta["sample_token"])
    missing = [channel for channel in CAMERA_CHANNELS if channel not in sample_meta["channels"]]
    if missing:
        raise ValueError(
            "The refinement dataset must contain all six cameras to rebuild the 360 degree pseudo-LiDAR. "
            f"Missing for sample {sample_meta['sample_token']}: {missing}"
        )

    depth_pro_results = {}
    refined_results = {}
    for channel in CAMERA_CHANNELS:
        entry = sample_meta["channels"][channel]
        data = np.load(dataset_dir / entry["npz_path"])
        depth_pro = data["depth_pro"].astype(np.float32)
        rgb = data["rgb"].astype(np.uint8)
        intrinsic = data["intrinsic"].astype(np.float32)
        refined = refine_depth(depth_pro, rgb, model, max_depth, use_rgb, device)

        camera_sd = nusc.get("sample_data", sample["data"][channel])
        camera_calib = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
        base_record = {
            "image": rgb,
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "calibrated_sensor": camera_calib,
        }
        depth_pro_results[channel] = {**base_record, "depth": depth_pro}
        refined_results[channel] = {**base_record, "depth": refined}

    return depth_pro_results, refined_results


def write_sample_outputs(
    output_dir: Path,
    nusc: NuScenes,
    sample_meta: dict,
    depth_source: str,
    results: dict,
) -> dict:
    sample_dir = output_dir / depth_source / sample_meta["sample_token"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    ring_pcd, pseudo_pcd, summary = build_outputs_for_sample(results)
    lidar_pcd = load_lidar_top_pcd(nusc, sample_meta["sample_token"])

    ring_path = sample_dir / "pcd_ring_6cams_ego.ply"
    pseudo_path = sample_dir / "pcd_pseudolidar_ego.ply"
    lidar_path = sample_dir / "pcd_lidar_top_ego.ply"
    o3d.io.write_point_cloud(str(ring_path), ring_pcd)
    o3d.io.write_point_cloud(str(pseudo_path), pseudo_pcd)
    o3d.io.write_point_cloud(str(lidar_path), lidar_pcd)

    info = {
        "sample_token": sample_meta["sample_token"],
        "index": int(sample_meta["index"]),
        "timestamp_s": None,
        "depth_source": depth_source,
        "ring_path": str(ring_path),
        "pseudo_path": str(pseudo_path),
        "lidar_path": str(lidar_path),
        **summary,
    }
    (sample_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def add_timestamps_from_nuscenes(nusc: NuScenes, processed: List[dict]) -> None:
    for item in processed:
        sample = nusc.get("sample", item["sample_token"])
        item["timestamp_s"] = float(sample["timestamp"] * 1e-6)


def write_run_summary(output_dir: Path, scene_name: str, dataset_dir: Path, depth_source: str, processed: List[dict]) -> Path:
    source_dir = output_dir / depth_source
    source_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "scene_name": scene_name,
        "depth_source": depth_source,
        "dataset_dir": str(dataset_dir),
        "num_processed": len(processed),
        "samples": processed,
    }
    path = source_dir / "run_summary.json"
    path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate native and refined pseudo-LiDAR clouds from the depth refinement dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used by the refiner. Default: auto selects CUDA when available.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    index_path = dataset_dir / "dataset_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    version = args.version or index["version"]
    dataroot = args.dataroot or Path(index["dataroot"])
    scene_name = index["scene_name"]

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    device = resolve_device(args.device)
    model, max_depth, use_rgb = load_refiner(args.checkpoint, device)
    selected_samples = group_entries_by_sample(index, args.max_samples)

    output_dir = (args.output_dir or REPO_ROOT / "outputs" / scene_name / "refined_pseudolidar").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_by_source = {"depth_pro": [], "refined": []}
    print(f"device: {device}")
    print(f"use_rgb: {use_rgb}")

    for sample_meta in selected_samples:
        depth_pro_results, refined_results = build_results_for_sample(
            nusc=nusc,
            dataset_dir=dataset_dir,
            sample_meta=sample_meta,
            model=model,
            max_depth=max_depth,
            use_rgb=use_rgb,
            device=device,
        )
        processed_by_source["depth_pro"].append(
            write_sample_outputs(output_dir, nusc, sample_meta, "depth_pro", depth_pro_results)
        )
        processed_by_source["refined"].append(
            write_sample_outputs(output_dir, nusc, sample_meta, "refined", refined_results)
        )
        print(f"processed sample {sample_meta['index']}: {sample_meta['sample_token']}")

    summary_paths = {}
    for depth_source, processed in processed_by_source.items():
        add_timestamps_from_nuscenes(nusc, processed)
        path = write_run_summary(output_dir, scene_name, dataset_dir, depth_source, processed)
        summary_paths[depth_source] = str(path)

    comparison_index = {
        "task": "refined_pseudolidar_generation",
        "scene_name": scene_name,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "use_rgb": use_rgb,
        "num_samples": len(selected_samples),
        "run_summaries": summary_paths,
        "note": (
            "depth_pro and refined are generated from the same stored Depth Pro maps, same resolution, "
            "same camera calibration and same pseudo-LiDAR conversion. This makes the comparison controlled."
        ),
    }
    comparison_path = output_dir / "comparison_index.json"
    comparison_path.write_text(json.dumps(comparison_index, indent=2), encoding="utf-8")
    print(comparison_path)
    print(json.dumps(comparison_index, indent=2))


if __name__ == "__main__":
    main()
