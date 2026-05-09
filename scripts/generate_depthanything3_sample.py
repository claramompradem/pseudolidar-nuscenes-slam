from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from nuscenes.nuscenes import NuScenes

from generate_pseudolidar_manifest import (
    CAMERA_CHANNELS,
    build_outputs_for_sample,
    load_lidar_top_pcd,
)


def import_depth_anything_3(depth_anything_root: Path | None):
    if depth_anything_root is not None:
        depth_anything_root = depth_anything_root.expanduser().resolve()
        if not depth_anything_root.exists():
            raise FileNotFoundError(f"Depth Anything 3 root not found: {depth_anything_root}")
        sys.path.insert(0, str(depth_anything_root / "src"))
        sys.path.insert(0, str(depth_anything_root))

    from depth_anything_3.api import DepthAnything3  # noqa: PLC0415

    return DepthAnything3


def sample_from_manifest(manifest: dict, sample_index: int) -> dict:
    for sample in manifest["samples"]:
        if int(sample["index"]) == sample_index:
            return sample
    raise ValueError(f"Sample index {sample_index} not found in manifest.")


def run_depth_anything_3_for_sample(
    nusc: NuScenes,
    sample_token: str,
    model,
    device: torch.device,
    sample_dir: Path,
    max_width: int,
    process_res: int,
) -> tuple[dict[str, dict], dict[str, float]]:
    sample = nusc.get("sample", sample_token)
    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for channel in CAMERA_CHANNELS:
        sample_data = nusc.get("sample_data", sample["data"][channel])
        calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        image_path = Path(nusc.dataroot) / sample_data["filename"]
        image_pil = Image.open(image_path).convert("RGB")

        if image_pil.width > max_width:
            new_height = int(image_pil.height * max_width / image_pil.width)
            image_pil = image_pil.resize((max_width, new_height))

        image_np = np.asarray(image_pil)
        intrinsic = np.asarray(calib["camera_intrinsic"], dtype=np.float32)
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        scale_x = image_pil.width / float(sample_data["width"])
        scale_y = image_pil.height / float(sample_data["height"])
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y

        intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        start = time.perf_counter()
        with torch.no_grad():
            prediction = model.inference(
                [image_pil],
                intrinsics=intrinsics[None],
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        depth = prediction.depth[0].astype(np.float32)
        depth = cv2.resize(depth, (image_pil.width, image_pil.height), interpolation=cv2.INTER_LINEAR)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        channel_dir = sample_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        np.save(channel_dir / "rgb.npy", image_np)
        np.save(channel_dir / "depth.npy", depth)

        results[channel] = {
            "depth": depth,
            "image": image_np,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "calibrated_sensor": calib,
        }
        timings[channel] = elapsed

        print(
            f"{channel}: depth_shape={depth.shape} "
            f"depth_min={np.nanmin(depth):.2f} depth_p95={np.nanpercentile(depth, 95):.2f} "
            f"infer_time_s={elapsed:.2f}"
        )

    return results, timings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on one nuScenes sample and save ring/pseudo-LiDAR point clouds."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-id", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--depth-anything-root", type=Path, default=None)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    depth_anything_root = args.depth_anything_root
    if depth_anything_root is None and "DEPTH_ANYTHING3_ROOT" in os.environ:
        depth_anything_root = Path(os.environ["DEPTH_ANYTHING3_ROOT"])

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample_meta = sample_from_manifest(manifest, args.sample_index)
    nusc = NuScenes(version=manifest["version"], dataroot=manifest["dataroot"], verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DepthAnything3 = import_depth_anything_3(depth_anything_root)
    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()

    scene_dir = args.output_root / manifest["scene_name"]
    sample_dir = scene_dir / f"depthanything3_sample_{args.sample_index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    results, timings = run_depth_anything_3_for_sample(
        nusc=nusc,
        sample_token=sample_meta["sample_token"],
        model=model,
        device=device,
        sample_dir=sample_dir,
        max_width=args.max_width,
        process_res=args.process_res,
    )

    ring_pcd, pseudo_pcd, summary = build_outputs_for_sample(results)
    lidar_pcd = load_lidar_top_pcd(nusc, sample_meta["sample_token"])

    ring_path = sample_dir / "pcd_ring_6cams_ego.ply"
    pseudo_path = sample_dir / "pcd_pseudolidar_ego.ply"
    lidar_path = sample_dir / "pcd_lidar_top_ego.ply"
    o3d.io.write_point_cloud(str(ring_path), ring_pcd)
    o3d.io.write_point_cloud(str(pseudo_path), pseudo_pcd)
    o3d.io.write_point_cloud(str(lidar_path), lidar_pcd)

    info = {
        "model": "Depth Anything 3",
        "model_id": args.model_id,
        "sample_token": sample_meta["sample_token"],
        "index": sample_meta["index"],
        "timestamp_s": sample_meta["timestamp_s"],
        "ring_path": str(ring_path),
        "pseudo_path": str(pseudo_path),
        "lidar_path": str(lidar_path),
        "inference_time_s_by_channel": timings,
        "inference_time_s_total": float(sum(timings.values())),
        **summary,
    }
    (sample_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print("saved:", sample_dir)


if __name__ == "__main__":
    main()
