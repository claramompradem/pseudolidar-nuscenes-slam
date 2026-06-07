from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class DepthRefinementDataset(Dataset):
    def __init__(self, dataset_dir: Path, max_depth: float = 80.0, use_rgb: bool = False) -> None:
        self.dataset_dir = dataset_dir
        self.index = json.loads((dataset_dir / "dataset_index.json").read_text(encoding="utf-8-sig"))
        self.entries = self.index["entries"]
        self.max_depth = float(max_depth)
        self.use_rgb = bool(use_rgb)

        if not self.entries:
            raise ValueError(f"No entries found in {dataset_dir / 'dataset_index.json'}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        entry = self.entries[idx]
        data = np.load(self.dataset_dir / entry["npz_path"])

        depth_pro = data["depth_pro"].astype(np.float32)
        gt_depth = data["gt_depth"].astype(np.float32)
        valid_mask = data["valid_mask"].astype(np.float32)

        depth_norm = np.clip(depth_pro / self.max_depth, 0.0, 1.0)
        gt_norm = np.clip(gt_depth / self.max_depth, 0.0, 1.0)
        input_channels = [depth_norm[None]]
        if self.use_rgb:
            rgb = data["rgb"].astype(np.float32) / 255.0
            input_channels.append(np.transpose(rgb, (2, 0, 1)))
        model_input = np.concatenate(input_channels, axis=0).astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "depth": torch.from_numpy(depth_norm[None]),
            "target": torch.from_numpy(gt_norm[None]),
            "mask": torch.from_numpy(valid_mask[None]),
            "sample_id": f"{entry['sample_index']:03d}_{entry['channel']}",
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallDepthUNet(nn.Module):
    """Small residual U-Net for Depth Pro refinement."""

    def __init__(self, base_channels: int = 32, residual_scale: float = 0.25, in_channels: int = 1) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.in_channels = int(in_channels)

        self.enc1 = ConvBlock(self.in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        depth = model_input[:, :1]
        e1 = self.enc1(model_input)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = torch.tanh(self.out(d1)) * self.residual_scale
        return torch.clamp(depth + residual, 0.0, 1.0)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid])


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, max_depth: float) -> Dict[str, float]:
    valid = mask > 0.5
    if not torch.any(valid):
        return {
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "valid_pixels": 0,
            "abs_error_sum_m": 0.0,
            "sq_error_sum_m2": 0.0,
        }

    diff_m = (pred[valid] - target[valid]) * max_depth
    valid_pixels = int(valid.sum().item())
    mae = torch.mean(torch.abs(diff_m)).item()
    rmse = torch.sqrt(torch.mean(diff_m**2)).item()
    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "valid_pixels": valid_pixels,
        "abs_error_sum_m": float(torch.sum(torch.abs(diff_m)).item()),
        "sq_error_sum_m2": float(torch.sum(diff_m**2).item()),
    }


def summarize_metric_totals(abs_error_sum_m: float, sq_error_sum_m2: float, valid_pixels: int) -> Dict[str, float]:
    if valid_pixels == 0:
        return {"mae_m": float("nan"), "rmse_m": float("nan"), "valid_pixels": 0}
    return {
        "mae_m": float(abs_error_sum_m / valid_pixels),
        "rmse_m": float(np.sqrt(sq_error_sum_m2 / valid_pixels)),
        "valid_pixels": int(valid_pixels),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_depth: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    losses: List[float] = []
    valid_pixels = 0
    abs_error_sum_m = 0.0
    sq_error_sum_m2 = 0.0

    for batch in loader:
        model_input = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            pred = model(model_input)
            loss = masked_smooth_l1(pred, target, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = masked_metrics(pred.detach(), target, mask, max_depth)
        losses.append(float(loss.item()))
        valid_pixels += metrics["valid_pixels"]
        abs_error_sum_m += metrics["abs_error_sum_m"]
        sq_error_sum_m2 += metrics["sq_error_sum_m2"]

    metrics = summarize_metric_totals(abs_error_sum_m, sq_error_sum_m2, valid_pixels)
    metrics["loss"] = float(np.nanmean(losses))
    return metrics


def evaluate_depth_pro_baseline(loader: DataLoader, max_depth: float) -> Dict[str, float]:
    valid_pixels = 0
    abs_error_sum_m = 0.0
    sq_error_sum_m2 = 0.0

    for batch in loader:
        depth = batch["depth"]
        target = batch["target"]
        mask = batch["mask"]
        metrics = masked_metrics(depth, target, mask, max_depth)
        valid_pixels += metrics["valid_pixels"]
        abs_error_sum_m += metrics["abs_error_sum_m"]
        sq_error_sum_m2 += metrics["sq_error_sum_m2"]

    return summarize_metric_totals(abs_error_sum_m, sq_error_sum_m2, valid_pixels)


def split_dataset(
    dataset: DepthRefinementDataset,
    val_ratio: float,
    seed: int,
    split_by_sample: bool,
):
    if len(dataset) == 1:
        # Smoke-test mode: this is only meant to verify that the training code runs.
        # It must not be reported as a real train/validation experiment.
        split_details = {
            "train_indices": [0],
            "val_indices": [0],
            "train_sample_tokens": [dataset.entries[0]["sample_token"]],
            "val_sample_tokens": [dataset.entries[0]["sample_token"]],
        }
        return dataset, dataset, "single_entry_smoke_test_same_sample_for_train_and_val", split_details

    if not split_by_sample:
        val_len = max(1, int(round(len(dataset) * val_ratio)))
        train_len = len(dataset) - val_len
        if train_len < 1:
            train_len = len(dataset) - 1
            val_len = 1
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
        train_indices = list(train_set.indices)
        val_indices = list(val_set.indices)
        split_details = {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_sample_tokens": sorted({dataset.entries[idx]["sample_token"] for idx in train_indices}),
            "val_sample_tokens": sorted({dataset.entries[idx]["sample_token"] for idx in val_indices}),
        }
        return train_set, val_set, "entry_random_split", split_details

    rng = random.Random(seed)
    sample_tokens = sorted({entry["sample_token"] for entry in dataset.entries})
    if len(sample_tokens) == 1:
        split_details = {
            "train_indices": list(range(len(dataset))),
            "val_indices": list(range(len(dataset))),
            "train_sample_tokens": sample_tokens,
            "val_sample_tokens": sample_tokens,
        }
        return dataset, dataset, "single_sample_smoke_test_same_sample_for_train_and_val", split_details

    rng.shuffle(sample_tokens)
    val_sample_count = max(1, int(round(len(sample_tokens) * val_ratio)))
    if val_sample_count >= len(sample_tokens):
        val_sample_count = len(sample_tokens) - 1

    val_tokens = set(sample_tokens[:val_sample_count])
    train_indices = [
        idx for idx, entry in enumerate(dataset.entries) if entry["sample_token"] not in val_tokens
    ]
    val_indices = [
        idx for idx, entry in enumerate(dataset.entries) if entry["sample_token"] in val_tokens
    ]
    split_details = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_tokens": sorted(
            {dataset.entries[idx]["sample_token"] for idx in train_indices}
        ),
        "val_sample_tokens": sorted(
            {dataset.entries[idx]["sample_token"] for idx in val_indices}
        ),
    }
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        "sample_token_grouped_split",
        split_details,
    )


def save_history(history: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_mae_m"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_mae_m"] for row in history], label="val")
    axes[0].set_title("Masked MAE [m]")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_rmse_m"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_rmse_m"] for row in history], label="val")
    axes[1].set_title("Masked RMSE [m]")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small Depth Pro refinement U-Net.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--split-by-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep all cameras from the same nuScenes sample in the same split. "
            "Use --no-split-by-sample to reproduce the older per-entry random split."
        ),
    )
    parser.add_argument(
        "--use-rgb",
        action="store_true",
        help="Use RGB channels together with the Depth Pro map as network input.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DepthRefinementDataset(args.dataset_dir, max_depth=args.max_depth, use_rgb=args.use_rgb)
    train_set, val_set, split_note, split_details = split_dataset(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_by_sample=args.split_by_sample,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_dir = args.output_dir or args.dataset_dir / "refiner_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 4 if args.use_rgb else 1
    model = SmallDepthUNet(base_channels=args.base_channels, in_channels=input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    baseline_metrics = {
        "train_depth_pro": evaluate_depth_pro_baseline(train_loader, args.max_depth),
        "val_depth_pro": evaluate_depth_pro_baseline(val_loader, args.max_depth),
        "split_note": split_note,
        "num_entries": len(dataset),
        "num_train_entries": len(train_set),
        "num_val_entries": len(val_set),
        "split_by_sample": args.split_by_sample,
        "split_details": split_details,
        "use_rgb": args.use_rgb,
        "input_channels": input_channels,
        "metric_note": "MAE and RMSE are weighted by the number of valid sparse LiDAR pixels.",
    }
    (output_dir / "baseline_depth_pro_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2),
        encoding="utf-8",
    )
    print("Depth Pro baseline:")
    print(json.dumps(baseline_metrics, indent=2))

    history: List[dict] = []
    best_val = float("inf")
    best_path = output_dir / "best_depth_refiner.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.max_depth)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, args.max_depth)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_m": train_metrics["mae_m"],
            "train_rmse_m": train_metrics["rmse_m"],
            "val_loss": val_metrics["loss"],
            "val_mae_m": val_metrics["mae_m"],
            "val_rmse_m": val_metrics["rmse_m"],
            "train_valid_pixels": train_metrics["valid_pixels"],
            "val_valid_pixels": val_metrics["valid_pixels"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if row["val_mae_m"] < best_val:
            best_val = row["val_mae_m"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": "SmallDepthUNet",
                    "base_channels": args.base_channels,
                    "input_channels": input_channels,
                    "use_rgb": args.use_rgb,
                    "max_depth": args.max_depth,
                    "epoch": epoch,
                    "val_mae_m": best_val,
                    "dataset_dir": str(args.dataset_dir),
                    "split_note": split_note,
                    "split_by_sample": args.split_by_sample,
                    "split_details": split_details,
                },
                best_path,
            )

    save_history(history, output_dir)
    print("saved:", best_path)
    print("saved:", output_dir / "training_history.json")
    print("saved:", output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
