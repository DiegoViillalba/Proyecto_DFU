#!/usr/bin/env python3
import os
import shutil
import subprocess
import glob
from PIL import Image, ImageDraw
import numpy as np

REPO_ROOT = "/home/diego-villalba/Proyecto_DFU"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
    else:
        print(res.stdout)
    return res.returncode == 0

def move_file_or_dir(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.move(src, dst)
    else:
        shutil.move(src, dst)
    print(f"Moved: {src} -> {dst}")

def generate_dummy_data(images_dir, masks_dir, num_files=5):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    for i in range(1, num_files + 1):
        filename = f"img_{i:03d}.png"
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)
        if os.path.exists(img_path):
            continue
            
        # Create mask (classes 0, 1, 2, 3)
        mask_arr = np.zeros((240, 240), dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        draw = ImageDraw.Draw(mask_img)
        draw.ellipse([40, 40, 140, 140], fill=1)
        draw.rectangle([100, 100, 180, 180], fill=2)
        mask_img = mask_img.convert("P")
        mask_img.save(mask_path)
        
        # Create corresponding RGB image
        img_arr = np.random.randint(80, 180, (240, 240, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(img_path)

def main():
    print("=" * 60)
    print("REORGANIZANDO IMÁGENES Y MÁSCARAS PARA DVC")
    print("=" * 60)
    
    # 1. Create data folders in root
    dfu_tissue_dir = os.path.join(REPO_ROOT, "data/dfu_tissue")
    miccai_dir = os.path.join(REPO_ROOT, "data/miccai")
    
    # 2. Move existing MICCAI files
    src_miccai = os.path.join(REPO_ROOT, "src/SegmentationNetworks/data_DFU_images/data_MICCAI")
    if os.path.exists(src_miccai):
        print("[+] Moviendo dataset MICCAI a data/miccai/...")
        for folder in ["train_images", "train_masks", "val_images", "val_masks", "images", "masks"]:
            move_file_or_dir(os.path.join(src_miccai, folder), os.path.join(miccai_dir, folder))
            
    # 3. Move existing DFUTissue test files (synthetic ones we generated)
    src_dfu_test_images = os.path.join(REPO_ROOT, "src/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_images")
    src_dfu_test_masks = os.path.join(REPO_ROOT, "src/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_masks")
    if os.path.exists(src_dfu_test_images):
        print("[+] Moviendo DFUTissue test a data/dfu_tissue/test_...")
        move_file_or_dir(src_dfu_test_images, os.path.join(dfu_tissue_dir, "test_images"))
        move_file_or_dir(src_dfu_test_masks, os.path.join(dfu_tissue_dir, "test_masks"))
        
    # 4. Generate dummy train and val files for DFU Tissue classification
    print("[+] Generando datos dummy de entrenamiento y validación para DFUTissue...")
    generate_dummy_data(os.path.join(dfu_tissue_dir, "train_images"), os.path.join(dfu_tissue_dir, "train_masks"))
    generate_dummy_data(os.path.join(dfu_tissue_dir, "val_images"), os.path.join(dfu_tissue_dir, "val_masks"))
    
    # 5. Clean up unused empty directories
    unused_dirs = [
        "src/SegmentationNetworks/data_DFU_images/data_MICCAI",
        "src/ClasificationAlgorithms/data_TissueSegNet/data_padded"
    ]
    for ud in unused_dirs:
        abs_ud = os.path.join(REPO_ROOT, ud)
        if os.path.exists(abs_ud) and len(os.listdir(abs_ud)) == 0:
            os.rmdir(abs_ud)
            print(f"Removed empty unused dir: {ud}")

    # 6. Initialize DVC
    dvc_dir = os.path.join(REPO_ROOT, ".dvc")
    if not os.path.exists(dvc_dir):
        print("[+] Inicializando DVC en el repositorio...")
        run_cmd([".venv/bin/dvc", "init", "--no-scm"])
    else:
        print("[+] DVC ya está inicializado.")
        
    # 7. Add data folders to DVC
    print("[+] Registrando carpetas en DVC...")
    run_cmd([".venv/bin/dvc", "add", "data/dfu_tissue"])
    run_cmd([".venv/bin/dvc", "add", "data/miccai"])
    
    # 8. Update paths in python training/testing files
    print("\n[+] Actualizando paths de datos en los scripts de Python...")
    
    def process_python_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        modified = False
        rel_path = os.path.relpath(file_path, REPO_ROOT)
        
        # Determine depth to root for REPO_ROOT definition
        parts = rel_path.split(os.sep)
        depth = len(parts) - 1
        repo_root_expr = "../" * depth
        if repo_root_expr.endswith("/"):
            repo_root_expr = repo_root_expr[:-1]
            
        # Check if REPO_ROOT is defined in the file
        if "REPO_ROOT = " not in content:
            # We must import os
            if "import os" not in content and "from os import" not in content:
                content = "import os\n" + content
            repo_root_def = f'\nREPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "{repo_root_expr}"))\n'
            content = repo_root_def + content
            modified = True
            
        # Parse classification paths replacements
        if "ClasificationAlgorithms" in rel_path:
            replacements = {
                "train_images": 'os.path.join(REPO_ROOT, "data/dfu_tissue/train_images")',
                "train_masks": 'os.path.join(REPO_ROOT, "data/dfu_tissue/train_masks")',
                "val_images": 'os.path.join(REPO_ROOT, "data/dfu_tissue/val_images")',
                "val_masks": 'os.path.join(REPO_ROOT, "data/dfu_tissue/val_masks")',
                "test_images": 'os.path.join(REPO_ROOT, "data/dfu_tissue/test_images")',
                "test_masks": 'os.path.join(REPO_ROOT, "data/dfu_tissue/test_masks")'
            }
        else:
            # Segmentation paths replacements
            replacements = {
                "train_images": 'os.path.join(REPO_ROOT, "data/miccai/train_images")',
                "train_masks": 'os.path.join(REPO_ROOT, "data/miccai/train_masks")',
                "val_images": 'os.path.join(REPO_ROOT, "data/miccai/val_images")',
                "val_masks": 'os.path.join(REPO_ROOT, "data/miccai/val_masks")',
                "test_images": 'os.path.join(REPO_ROOT, "data/miccai/val_images")',
                "test_masks": 'os.path.join(REPO_ROOT, "data/miccai/val_masks")'
            }
            
        # Apply replacements to matching lines
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            # Check for path definitions
            for var_key, expr in replacements.items():
                # We target variable assignments like TRAIN_IMG_DIR = ...
                if var_key.upper() in line and "=" in line and ("C:/" in line or "data_padded" in line or "data_MICCAI" in line or "test_image_dir" in line or "test_mask_dir" in line):
                    # Rewrite the line
                    var_name = line.split("=")[0].strip()
                    lines[idx] = f"{var_name} = {expr}"
                    modified = True
                    print(f"  Line rewritten in {rel_path}: {line.strip()} -> {lines[idx].strip()}")
                    
        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"Updated python file paths: {rel_path}")

    # Process files
    for root, _, files in os.walk(os.path.join(REPO_ROOT, "src")):
        for file in files:
            if file.endswith(".py"):
                process_python_file(os.path.join(root, file))

    print("\n[SUCCESS] Reorganización completada.")

if __name__ == "__main__":
    main()
