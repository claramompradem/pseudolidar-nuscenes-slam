from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
import torch


CAMERA_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def transform_from_translation_rotation(translation: List[float], rotation: List[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def import_depth_pro(depth_pro_root: Path):
    depth_pro_root = depth_pro_root.expanduser().resolve()
    if not depth_pro_root.exists():
        raise FileNotFoundError(f"Depth Pro root not found: {depth_pro_root}")

    sys.path.insert(0, str(depth_pro_root / "src"))
    sys.path.insert(0, str(depth_pro_root))
    os.chdir(str(depth_pro_root))

    import depth_pro  # noqa: PLC0415

    return depth_pro


def load_samples(args: argparse.Namespace) -> Tuple[dict, List[dict]]:
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        return manifest, manifest["samples"][: args.max_samples] if args.max_samples else manifest["samples"]

    if args.run_summary:
        if args.version is None or args.dataroot is None:
            raise ValueError("--version and --dataroot are required when using --run-summary.")
        run_summary = json.loads(args.run_summary.read_text(encoding="utf-8-sig"))
        samples = [
            {
                "index": item["index"],
                "sample_token": item["sample_token"],
                "timestamp_s": item.get("timestamp_s"),
            }
            for item in run_summary["samples"]
        ]
        manifest = {
            "version": args.version,
            "dataroot": str(args.dataroot),
            "scene_name": run_summary["scene_name"],
            "samples": samples,
        }
        return manifest, samples[: args.max_samples] if args.max_samples else samples

    raise ValueError("Provide either --manifest or --run-summary.")


def resize_image_and_intrinsics(
    image_path: Path,
    intrinsic: np.ndarray,
    original_width: int,
    original_height: int,
    max_width: int,
) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    if image.width != original_width or image.height != original_height:
        raise ValueError(f"Unexpected image size for {image_path}: {image.size}")

    scaled_intrinsic = intrinsic.astype(np.float64).copy()
    if image.width > max_width:
        new_height = int(image.height * max_width / image.width)
        image = image.resize((max_width, new_height))

    scale_x = image.width / float(original_width)
    scale_y = image.height / float(original_height)
    scaled_intrinsic[0, 0] *= scale_x
    scaled_intrinsic[1, 1] *= scale_y
    scaled_intrinsic[0, 2] *= scale_x
    scaled_intrinsic[1, 2] *= scale_y
    return image, scaled_intrinsic


def infer_depth_pro(image: Image.Image, model, depth_transform, device: torch.device, focal_px: float) -> np.ndarray:
    image_tensor = depth_transform(image).to(device)
    f_px_tensor = torch.tensor(float(focal_px), device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=f_px_tensor)
    depth = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)

    if depth.shape != (image.height, image.width):
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((image.width, image.height), Image.BILINEAR)
        depth = np.asarray(depth_img, dtype=np.float32)

    del image_tensor, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth


def load_lidar_points_sensor(nusc: NuScenes, sample: dict) -> Tuple[np.ndarray, dict]:
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_path = Path(nusc.dataroot) / lidar_sd["filename"]
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    return points, lidar_sd


def project_lidar_to_camera_depth(
    nusc: NuScenes,
    sample: dict,
    camera_sd: dict,
    camera_calib: dict,
    intrinsic: np.ndarray,
    image_shape: Tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    lidar_points, lidar_sd = load_lidar_points_sensor(nusc, sample)

    lidar_calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])

    t_ego_from_lidar = transform_from_translation_rotation(lidar_calib["translation"], lidar_calib["rotation"])
    t_global_from_lidar_ego = transform_from_translation_rotation(lidar_pose["translation"], lidar_pose["rotation"])
    t_global_from_camera_ego = transform_from_translation_rotation(camera_pose["translation"], camera_pose["rotation"])
    t_ego_from_camera = transform_from_translation_rotation(camera_calib["translation"], camera_calib["rotation"])

    points_ego_lidar = transform_points(lidar_points, t_ego_from_lidar)
    points_global = transform_points(points_ego_lidar, t_global_from_lidar_ego)
    points_ego_camera = transform_points(points_global, np.linalg.inv(t_global_from_camera_ego))
    points_camera = transform_points(points_ego_camera, np.linalg.inv(t_ego_from_camera))

    z = points_camera[:, 2]
    valid_z = (z > min_depth) & (z < max_depth)
    points_camera = points_camera[valid_z]
    z = z[valid_z]

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = np.round((points_camera[:, 0] * fx / z) + cx).astype(np.int64)
    v = np.round((points_camera[:, 1] * fy / z) + cy).astype(np.int64)

    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_image]
    v = v[in_image]
    z = z[in_image]

    gt_depth = np.full((h, w), np.inf, dtype=np.float32)
    flat_index = v * w + u
    np.minimum.at(gt_depth.reshape(-1), flat_index, z.astype(np.float32))

    valid_mask = np.isfinite(gt_depth)
    gt_depth[~valid_mask] = 0.0
    return gt_depth, valid_mask.astype(np.uint8)


def save_preview(
    image_np: np.ndarray,
    depth_pro: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(image_np)
    axes[0].set_title("RGB")
    axes[1].imshow(depth_pro, cmap="magma")
    axes[1].set_title("Depth Pro")
    axes[2].imshow(image_np)
    ys, xs = np.where(valid_mask > 0)
    colors = gt_depth[ys, xs]
    axes[2].scatter(xs, ys, c=colors, s=1, cmap="magma")
    axes[2].set_title("GT LiDAR proyectado")
    for ax in axes:
        ax.axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_entry_from_arrays(
    sample_meta: dict,
    channel: str,
    npz_path: Path,
    output_dir: Path,
    camera_filename: str,
    depth_pro_map: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    valid_count = int(valid_mask.sum())
    total_pixels = int(valid_mask.size)
    return {
        "sample_index": int(sample_meta["index"]),
        "sample_token": sample_meta["sample_token"],
        "channel": channel,
        "npz_path": str(npz_path.relative_to(output_dir)),
        "image_path": camera_filename,
        "width": int(depth_pro_map.shape[1]),
        "height": int(depth_pro_map.shape[0]),
        "valid_lidar_pixels": valid_count,
        "valid_lidar_ratio": float(valid_count / max(total_pixels, 1)),
        "depth_pro_min": float(np.nanmin(depth_pro_map)),
        "depth_pro_median": float(np.nanmedian(depth_pro_map)),
        "depth_pro_max": float(np.nanmax(depth_pro_map)),
        "gt_depth_median_valid": float(np.median(gt_depth[valid_mask > 0])) if valid_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare sparse supervised pairs for Depth Pro depth refinement."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-summary", type=Path, default=None)
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument(
        "--depth-pro-root",
        type=Path,
        default=Path(os.environ["DEPTH_PRO_ROOT"]) if "DEPTH_PRO_ROOT" in os.environ else Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=CAMERA_CHANNELS,
        choices=CAMERA_CHANNELS,
        help="Camera channels to process. By default, all six nuScenes cameras are used.",
    )
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--preview-limit", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest, selected_samples = load_samples(args)
    output_dir = args.output_dir or Path("outputs") / manifest["scene_name"] / "depth_refinement_dataset"
    output_dir = output_dir.resolve()
    npz_dir = output_dir / "npz"
    preview_dir = output_dir / "previews"
    npz_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=manifest["version"], dataroot=manifest["dataroot"], verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_cwd = Path.cwd()
    model = None
    depth_transform = None

    entries = []
    preview_count = 0
    for sample_meta in selected_samples:
        sample = nusc.get("sample", sample_meta["sample_token"])
        for channel in args.channels:
            camera_sd = nusc.get("sample_data", sample["data"][channel])
            camera_calib = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
            stem = f"{sample_meta['index']:03d}_{sample_meta['sample_token']}_{channel}"
            npz_path = npz_dir / f"{stem}.npz"

            if npz_path.exists() and not args.overwrite:
                data = np.load(npz_path)
                entry = build_entry_from_arrays(
                    sample_meta=sample_meta,
                    channel=channel,
                    npz_path=npz_path,
                    output_dir=output_dir,
                    camera_filename=camera_sd["filename"],
                    depth_pro_map=data["depth_pro"].astype(np.float32),
                    gt_depth=data["gt_depth"].astype(np.float32),
                    valid_mask=data["valid_mask"].astype(np.uint8),
                )
                preview_path = preview_dir / f"{stem}.png"
                if preview_path.exists():
                    entry["preview_path"] = str(preview_path.relative_to(output_dir))
                entries.append(entry)
                print(f"indexed existing: {npz_path}")
                continue

            if model is None or depth_transform is None:
                depth_pro = import_depth_pro(args.depth_pro_root)
                model, depth_transform = depth_pro.create_model_and_transforms()
                model.eval().to(device)
                os.chdir(original_cwd)

            intrinsic = np.asarray(camera_calib["camera_intrinsic"], dtype=np.float64)
            image_path = Path(nusc.dataroot) / camera_sd["filename"]
            image, scaled_intrinsic = resize_image_and_intrinsics(
                image_path,
                intrinsic,
                camera_sd["width"],
                camera_sd["height"],
                args.max_width,
            )
            image_np = np.asarray(image, dtype=np.uint8)
            focal_px = float((scaled_intrinsic[0, 0] + scaled_intrinsic[1, 1]) * 0.5)
            depth_pro_map = infer_depth_pro(image, model, depth_transform, device, focal_px)
            gt_depth, valid_mask = project_lidar_to_camera_depth(
                nusc,
                sample,
                camera_sd,
                camera_calib,
                scaled_intrinsic,
                depth_pro_map.shape,
                args.min_depth,
                args.max_depth,
            )

            np.savez_compressed(
                npz_path,
                depth_pro=depth_pro_map.astype(np.float32),
                gt_depth=gt_depth.astype(np.float32),
                valid_mask=valid_mask.astype(np.uint8),
                rgb=image_np,
                intrinsic=scaled_intrinsic.astype(np.float32),
            )

            entry = build_entry_from_arrays(
                sample_meta=sample_meta,
                channel=channel,
                npz_path=npz_path,
                output_dir=output_dir,
                camera_filename=camera_sd["filename"],
                depth_pro_map=depth_pro_map,
                gt_depth=gt_depth,
                valid_mask=valid_mask,
            )
            entries.append(entry)

            if preview_count < args.preview_limit:
                preview_path = preview_dir / f"{stem}.png"
                save_preview(image_np, depth_pro_map, gt_depth, valid_mask, preview_path)
                entry["preview_path"] = str(preview_path.relative_to(output_dir))
                preview_count += 1

            print(json.dumps(entry, indent=2))

    index = {
        "task": "depth_refinement_sparse_lidar_supervision",
        "scene_name": manifest["scene_name"],
        "version": manifest["version"],
        "dataroot": manifest["dataroot"],
        "num_entries": len(entries),
        "max_width": args.max_width,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "channels": args.channels,
        "entries": entries,
    }
    index_path = output_dir / "dataset_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("saved:", index_path)


if __name__ == "__main__":
    main()
