#!/usr/bin/env python3
"""Local data sanity checker for DFUTissue validation and test sets."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

VALID_LABELS = {0, 1, 2, 3}
EXPECTED_SIZE = (256, 256)
COLORMAP = {
    0: (0.0, 0.0, 0.0, 1.0),
    1: (1.0, 0.2, 0.2, 1.0),
    2: (1.0, 1.0, 0.0, 1.0),
    3: (0.3, 0.0, 0.0, 1.0),
}


def load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask, dtype=np.uint8)


def verify_mask(mask: np.ndarray, filename: str, expected_size: Tuple[int, int]) -> List[str]:
    errors: List[str] = []
    unique_values = set(np.unique(mask).tolist())
    if not unique_values.issubset(VALID_LABELS):
        errors.append(
            f"{filename}: máscara contiene valores inválidos {sorted(unique_values - VALID_LABELS)}"
        )
    if mask.shape != expected_size[::-1]:
        errors.append(
            f"{filename}: tamaño incorrecto {mask.shape}, se esperaba {expected_size[::-1]}"
        )
    return errors


def plot_sample(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for label, color in COLORMAP.items():
        mask_rgb[mask == label] = color[:3]
    overlay = (0.6 * image.astype(np.float32) / 255.0 + 0.4 * mask_rgb).clip(0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("RGB Original")
    axes[1].imshow(mask, cmap=plt.cm.get_cmap("tab10", len(VALID_LABELS)))
    axes[1].set_title("Ground Truth (RYB)")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def get_image_mask_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = root / f"{split}_images"
    mask_dir = root / f"{split}_masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"No se encontró {split} set en {root}. Esperado: {image_dir}, {mask_dir}"
        )
    image_files = sorted(image_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.png"))
    return [(image_path, mask_dir / image_path.name) for image_path in image_files]


def run_sanity_check(root: Path, split: str, output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = get_image_mask_pairs(root, split)
    errors: List[str] = []
    for image_path, mask_path in pairs:
        image = load_image(image_path)
        mask = load_mask(mask_path)
        errors.extend(verify_mask(mask, mask_path.name, EXPECTED_SIZE))
    if len(pairs) > 0:
        sample_paths = random.sample(pairs, min(4, len(pairs)))
        for idx, (image_path, mask_path) in enumerate(sample_paths, start=1):
            image = load_image(image_path)
            mask = load_mask(mask_path)
            plot_sample(
                image=image,
                mask=mask,
                output_path=output_dir / f"{split}_sample_{idx}.png",
                title=f"{split.capitalize()} sample {idx}: {image_path.name}",
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local DFUTissue data sanity check.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dfu_tissue",
        help="Ruta raíz del dataset DFUTissue.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "figures" / "local_data_sanity",
        help="Directorio donde guardar las imágenes de muestra generadas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: List[str] = []
    for split in ["val", "test"]:
        print(f"Validando split: {split}")
        try:
            split_errors = run_sanity_check(args.root, split, args.output_dir)
            errors.extend(split_errors)
        except FileNotFoundError as exc:
            print(exc)
            continue
    if errors:
        print("\nSe encontraron los siguientes problemas de integridad de datos:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nSanity check completado: no se detectaron errores en las máscaras ni en la resolución.")
    print(f"Imágenes de ejemplo guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
