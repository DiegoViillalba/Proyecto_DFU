#!/usr/bin/env python3
"""Training script for DFUTissue segmentation models using multiple random seeds."""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError as error:
    raise ImportError(
        "segmentation_models_pytorch is required for train_multiseed.py. "
        "Install it with pip install segmentation-models-pytorch"
    ) from error

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]
MODEL_NAME_MAP = {
    "MANet_MiT-b3": "MANet_MiT-b3",
    "UNet_MiT-b3": "UNet_MiT-b3",
    "SegFormer_MiT-b3": "SegFormer_MiT-b3",
    "ResUNet": "ResUNet",
    "UNet_MobNetV2": "UNet_MobNetV2",
}
CLASS_COUNT = 4
IMAGE_SIZE = 256
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "dfu_tissue"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DFUTissueDataset(Dataset):
    """Dataset for DFUTissue images and multi-class masks."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: Optional[A.BasicTransform] = None,
    ) -> None:
        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        self.transform = transform

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image count ({len(self.image_paths)}) does not match mask count "
                f"({len(self.mask_paths)}) in {image_dir} and {mask_dir}"
            )

        self._verify_indexing()

    def _verify_indexing(self) -> None:
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            if image_path.stem != mask_path.stem:
                raise ValueError(
                    f"Image and mask names must match: {image_path.name} != {mask_path.name}"
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(self.mask_paths[index]).convert("L"), dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float()
            mask = torch.from_numpy(mask).long()

        return image, mask


def build_transforms(image_size: int) -> Dict[str, A.Compose]:
    """Build augmentation and validation transforms."""
    train_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return {"train": train_transform, "val": val_transform}


def make_loaders(
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    transforms = build_transforms(image_size=image_size)
    train_dataset = DFUTissueDataset(
        image_dir=DATA_ROOT / "train_images",
        mask_dir=DATA_ROOT / "train_masks",
        transform=transforms["train"],
    )
    val_dataset = DFUTissueDataset(
        image_dir=DATA_ROOT / "val_images",
        mask_dir=DATA_ROOT / "val_masks",
        transform=transforms["val"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_model(model_name: str, num_classes: int = CLASS_COUNT) -> nn.Module:
    """Build a segmentation model from the requested name."""
    key = model_name.replace(" ", "_").replace("-", "_")
    if key == "MANet_MiT_b3" or key == "MANet_MiT_b3".replace("_", "_"):
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


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute training loss and integer predictions."""
    loss = loss_fn(outputs, targets)
    predictions = torch.argmax(outputs, dim=1)
    return loss, predictions


def evaluate_epoch(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluate model performance on validation set."""
    model.eval()
    running_loss = 0.0
    running_pixels = 0
    correct_pixels = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            running_pixels += masks.numel()
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct_pixels / max(running_pixels, 1)
    return avg_loss, accuracy


def train_model(
    model_name: str,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    image_size: int,
) -> None:
    """Train the requested model across multiple seeds and save weights."""
    weights_dir = WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        logging.info("Training %s for seed %s", model_name, seed)
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss()
        dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader, val_loader = make_loaders(batch_size=batch_size, num_workers=num_workers, image_size=image_size)

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks) + dice_loss(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
            epoch_loss /= len(train_loader.dataset)

            val_loss, val_accuracy = evaluate_epoch(val_loader, model, device, criterion)
            logging.info(
                "Seed %s epoch %s/%s - train_loss %.4f - val_loss %.4f - val_acc %.4f",
                seed,
                epoch,
                epochs,
                epoch_loss,
                val_loss,
                val_accuracy,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                filename = weights_dir / f"{model_name}_seed{seed}.pth"
                torch.save(model.state_dict(), filename)
                logging.info("Saved weights to %s", filename)

        logging.info("Finished training %s for seed %s", model_name, seed)


def parse_seeds(args: argparse.Namespace) -> List[int]:
    """Parse seed list or default seed count."""
    if args.seed_list:
        return [int(value) for value in args.seed_list]
    if args.seeds_count:
        return DEFAULT_SEEDS[: args.seeds_count]
    return DEFAULT_SEEDS


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DFUTissue models across multiple seeds.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAME_MAP.keys()),
        required=True,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--seeds-count",
        type=int,
        default=len(DEFAULT_SEEDS),
        help="Number of default seeds to use from the preconfigured list.",
    )
    parser.add_argument(
        "--seed-list",
        nargs="+",
        help="Explicit list of integer seeds to use.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs per seed.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input image size.")
    return parser


def main() -> None:
    parser = get_argument_parser()
    args = parser.parse_args()
    selected_seeds = parse_seeds(args)
    logging.info("Starting multi-seed training for %s", args.model)
    train_model(
        model_name=args.model,
        seeds=selected_seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
