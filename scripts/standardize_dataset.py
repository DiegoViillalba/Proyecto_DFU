#!/usr/bin/env python3
"""Standardize DFUTissue images and masks to 256x256 using letterboxing.

This script resizes images and masks while preserving the original aspect ratio.
Padding is added with zeros to avoid medical deformation of ulcer regions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

VALID_FOLDERS = [
    "train_images",
    "train_masks",
    "val_images",
    "val_masks",
    "test_images",
    "test_masks",
]
TARGET_SIZE = (256, 256)
MASK_LABELS = {0, 1, 2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize data/dfu_tissue to 256x256 with padding (letterbox)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dfu_tissue",
        help="Root folder of the DFUTissue dataset.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Target pixel size for width and height.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only report dataset sizes without modifying files.",
    )
    return parser.parse_args()


def letterbox_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width, target_height = target_size

    scale = min(target_width / width, target_height / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = image.resize((new_width, new_height), resample=Image.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    padded.paste(resized, (pad_x, pad_y))
    return padded


def letterbox_mask(mask: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    width, height = mask.size
    target_width, target_height = target_size

    scale = min(target_width / width, target_height / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = mask.resize((new_width, new_height), resample=Image.NEAREST)
    padded = Image.new("L", target_size, 0)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    padded.paste(resized, (pad_x, pad_y))
    return padded


def normalize_path_list(data_root: Path, folders: Iterable[str]) -> List[Path]:
    validated = []
    for folder in folders:
        folder_path = data_root / folder
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")
        validated.append(folder_path)
    return validated


def process_folder(folder_path: Path, target_size: Tuple[int, int], summary_only: bool) -> Tuple[int, int, int]:
    files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    total = len(files)
    changed = 0
    warnings = 0

    iterator = tqdm(files, desc=f"Processing {folder_path.name}", unit="file") if tqdm else files
    for path in iterator:
        try:
            if folder_path.name.endswith("_images"):
                image = Image.open(path).convert("RGB")
                if image.size != target_size:
                    changed += 1
                    if not summary_only:
                        output = letterbox_image(image, target_size)
                        output.save(path)
                else:
                    if not summary_only:
                        image.save(path)
            else:
                mask = Image.open(path).convert("L")
                if mask.size != target_size:
                    changed += 1
                    if not summary_only:
                        output = letterbox_mask(mask, target_size)
                        output.save(path)
                else:
                    if not summary_only:
                        mask.save(path)

                unique_values = set(int(v) for v in mask.getdata())
                if not unique_values.issubset(MASK_LABELS):
                    warnings += 1
                    print(
                        f"Warning: {path.name} contains labels {sorted(unique_values - MASK_LABELS)}",
                        file=sys.stderr,
                    )
        except Exception as exc:
            warnings += 1
            print(f"Failed to process {path}: {exc}", file=sys.stderr)
    return total, changed, warnings


def main() -> None:
    args = parse_args()
    target_size = (args.size, args.size)
    data_root = args.data_root

    print("Starting DFUTissue standardization with letterboxing")
    print(f"Data root: {data_root}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    folder_paths = normalize_path_list(data_root, VALID_FOLDERS)
    overall_total = 0
    overall_changed = 0
    overall_warnings = 0

    for folder_path in folder_paths:
        total, changed, warnings = process_folder(folder_path, target_size, args.summary_only)
        overall_total += total
        overall_changed += changed
        overall_warnings += warnings
        print(
            f"{folder_path.name}: {total} files checked, {changed} resized/re-saved, {warnings} warnings"
        )

    print("\nStandardization summary")
    print(f"Total files checked: {overall_total}")
    print(f"Files resized or re-saved: {overall_changed}")
    print(f"Warnings / failures: {overall_warnings}")
    print("Note: If files were modified and are tracked by DVC, run `dvc add` again on the changed paths.")


if __name__ == "__main__":
    main()
