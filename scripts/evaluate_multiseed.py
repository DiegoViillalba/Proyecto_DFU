#!/usr/bin/env python3
"""Evaluation script for DFUTissue multi-seed checkpoints."""
from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError as error:
    raise ImportError(
        "segmentation_models_pytorch is required for evaluate_multiseed.py. "
        "Install it with pip install segmentation-models-pytorch"
    ) from error

try:
    from medpy.metric.binary import hd95
except ImportError:
    hd95 = None  # type: ignore[assignment]

try:
    from monai.metrics import compute_hausdorff_distance
except ImportError:
    compute_hausdorff_distance = None  # type: ignore[assignment]

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]
MODEL_NAME_MAP = {
    "MANet_MiT-b3": "MANet_MiT-b3",
    "UNet_MiT-b3": "UNet_MiT-b3",
    "SegFormer_MiT-b3": "SegFormer_MiT-b3",
    "ResUNet": "ResUNet",
    "UNet_MobNetV2": "UNet_MobNetV2",
}
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "dfu_tissue"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
CLASS_NAMES = ["Granulation", "Esfacelo", "Necrotic"]
CLASS_LABELS = [1, 2, 3]
IMAGE_SIZE = 256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DFUTissueTestDataset(torch.utils.data.Dataset):
    """Lightweight DFUTissue test dataset loader."""

    def __init__(self, image_dir: Path, mask_dir: Path) -> None:
        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Image and mask counts do not match for test set.")
        self._verify_indexing()

    def _verify_indexing(self) -> None:
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            if image_path.stem != mask_path.stem:
                raise ValueError(
                    f"Image and mask file names must match: {image_path.name} != {mask_path.name}"
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(self.mask_paths[index]).convert("L"), dtype=np.int64)
        filename = self.image_paths[index].name
        return image, mask, filename


def get_model(model_name: str, num_classes: int = 4) -> nn.Module:
    """Instantiate the requested architecture for inference."""
    key = model_name.replace(" ", "_").replace("-", "_")
    if key == "MANet_MiT_b3" or key.lower().startswith("manet"):
        return smp.MAnet(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if key == "UNet_MiT_b3" or key.lower().startswith("unet_mit"):
        return smp.Unet(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if key == "SegFormer_MiT_b3" or key.lower().startswith("segformer"):
        return smp.Segformer(
            encoder_name="mit_b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if key == "ResUNet" or key.lower() == "resunet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if key == "UNet_MobNetV2" or key.lower().startswith("unet_mobilenet"):
        return smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def compute_range_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_label: int,
) -> Dict[str, float]:
    """Compute Dice, IoU, precision, recall and boundary metrics for a single class."""
    pred_binary = pred_mask == class_label
    gt_binary = gt_mask == class_label

    intersection = float(np.logical_and(pred_binary, gt_binary).sum())
    pred_sum = float(pred_binary.sum())
    gt_sum = float(gt_binary.sum())

    if gt_sum == 0.0 and pred_sum == 0.0:
        dsc = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dsc = 2.0 * intersection / max(pred_sum + gt_sum, 1e-8)
        iou = intersection / max(float(np.logical_or(pred_binary, gt_binary).sum()), 1e-8)
        precision = intersection / max(pred_sum, 1e-8)
        recall = intersection / max(gt_sum, 1e-8)

    hd95_px = 0.0
    assd_px = 0.0
    if np.any(gt_binary) and np.any(pred_binary):
        hd95_px = _compute_hd95(pred_binary, gt_binary)
        assd_px = _compute_assd(pred_binary, gt_binary)
    elif np.any(gt_binary) != np.any(pred_binary):
        hd95_px = float(max(pred_mask.shape))
        assd_px = float(max(pred_mask.shape)) / 2.0

    return {
        "dice": dsc,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95": hd95_px,
        "assd": assd_px,
    }


def _compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance using MedPy or MONAI."""
    if hd95 is not None:
        return float(hd95(pred.astype(np.uint8), gt.astype(np.uint8)))
    if compute_hausdorff_distance is not None:
        return float(compute_hausdorff_distance(pred.astype(np.uint8), gt.astype(np.uint8), percentile=95))
    from scipy.spatial.distance import cdist
    pred_border = _extract_border(pred)
    gt_border = _extract_border(gt)
    if pred_border.size == 0 or gt_border.size == 0:
        return float(max(pred.shape))
    dist_matrix = cdist(pred_border, gt_border)
    all_dists = np.concatenate([dist_matrix.min(axis=1), dist_matrix.min(axis=0)])
    return float(np.percentile(all_dists, 95))


def _compute_assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute average symmetric surface distance in pixels."""
    from scipy.spatial.distance import cdist
    pred_border = _extract_border(pred)
    gt_border = _extract_border(gt)
    if pred_border.size == 0 or gt_border.size == 0:
        return float(max(pred.shape)) / 2.0
    dist_pred = cdist(pred_border, gt_border).min(axis=1)
    dist_gt = cdist(pred_border, gt_border).min(axis=0)
    return float(np.concatenate([dist_pred, dist_gt]).mean())


def _extract_border(mask: np.ndarray) -> np.ndarray:
    """Return border coordinates for a binary mask."""
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask)
    border = np.logical_xor(mask, eroded)
    return np.argwhere(border)


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Load `state_dict` from a `.pth` weight file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)


def run_inference(
    model: nn.Module,
    dataset: DFUTissueTestDataset,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Run inference and compute metrics for every test image."""
    model.eval()
    metrics_by_image: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for image, gt_mask, filename in dataset:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            for label, class_name in zip(CLASS_LABELS, CLASS_NAMES):
                class_metrics = compute_range_metrics(pred_mask, gt_mask, label)
                if filename not in metrics_by_image:
                    metrics_by_image[filename] = {}
                suffix = f"_{class_name}"
                metrics_by_image[filename].update({f"dice{suffix}": class_metrics["dice"],
                                                   f"iou{suffix}": class_metrics["iou"],
                                                   f"precision{suffix}": class_metrics["precision"],
                                                   f"recall{suffix}": class_metrics["recall"],
                                                   f"hd95{suffix}": class_metrics["hd95"],
                                                   f"assd{suffix}": class_metrics["assd"]})
    return metrics_by_image


def aggregate_model_metrics(
    model_name: str,
    seed_paths: List[Path],
    dataset: DFUTissueTestDataset,
    device: torch.device,
) -> pd.DataFrame:
    """Aggregate metrics across all seeds for a single model."""
    records = []
    for checkpoint_path in seed_paths:
        seed = int(checkpoint_path.stem.split("seed")[-1])
        model = get_model(model_name, num_classes=4).to(device)
        load_checkpoint_weights(model, checkpoint_path)
        image_metrics = run_inference(model, dataset, device)
        for filename, metrics in image_metrics.items():
            record = {
                "model": model_name,
                "seed": seed,
                "image": filename,
            }
            record.update(metrics)
            records.append(record)
    return pd.DataFrame.from_records(records)


def format_mean_std(values: Iterable[float]) -> str:
    """Format mean and standard deviation as a string."""
    values_list = list(values)
    mean_value = float(np.mean(values_list))
    std_value = float(np.std(values_list, ddof=1)) if len(values_list) > 1 else 0.0
    return f"{mean_value:.4f} ± {std_value:.4f}"


def summarize_metrics(df: pd.DataFrame, output_path: Path) -> None:
    """Write a consolidated CSV with mean ± std for each metric and class."""
    summary_rows = []
    metric_columns = [
        "dice_Granulation",
        "iou_Granulation",
        "precision_Granulation",
        "recall_Granulation",
        "hd95_Granulation",
        "assd_Granulation",
        "dice_Esfacelo",
        "iou_Esfacelo",
        "precision_Esfacelo",
        "recall_Esfacelo",
        "hd95_Esfacelo",
        "assd_Esfacelo",
        "dice_Necrotic",
        "iou_Necrotic",
        "precision_Necrotic",
        "recall_Necrotic",
        "hd95_Necrotic",
        "assd_Necrotic",
    ]

    for model_name, model_df in df.groupby("model"):
        row = {"model": model_name}
        for metric in metric_columns:
            row[metric] = format_mean_std(model_df[metric].values)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    logging.info("Saved consolidated metrics to %s", output_path)


def find_seed_checkpoints(model_name: str) -> List[Path]:
    """Find checkpoint files for the requested model."""
    pattern = f"{model_name}_seed*.pth"
    return sorted(WEIGHTS_DIR.glob(pattern))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DFUTissue multi-seed checkpoints.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAME_MAP.keys()),
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seed ids to evaluate.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DATA_ROOT,
        help="Root path for DFUTissue test images and masks.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "dfutissue_multiseed_metrics.csv",
        help="Path to the consolidated metrics CSV output.",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    weights = find_seed_checkpoints(args.model)
    if not weights:
        raise FileNotFoundError(
            f"No checkpoint files found for {args.model} in {WEIGHTS_DIR}"
        )
    filtered_weights = [p for p in weights if any(f"seed{seed}" in p.stem for seed in args.seeds)]
    if not filtered_weights:
        raise ValueError(
            f"No checkpoint files matched requested seeds {args.seeds} for model {args.model}."
        )
    if not (args.test_root / "test_images").exists() or not (args.test_root / "test_masks").exists():
        raise FileNotFoundError("DFUTissue test directories not found under the provided test root.")

    dataset = DFUTissueTestDataset(
        image_dir=args.test_root / "test_images",
        mask_dir=args.test_root / "test_masks",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Running inference on %s images using %s", len(dataset), device)
    metrics_df = aggregate_model_metrics(args.model, filtered_weights, dataset, device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_csv, index=False)
    logging.info("Saved per-seed metrics to %s", args.output_csv)
    summarize_metrics(metrics_df, args.output_csv.parent / f"{args.model}_summary.csv")


if __name__ == "__main__":
    main()
