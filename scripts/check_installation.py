#!/usr/bin/env python3
import sys

# Define expected versions
EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "segmentation_models_pytorch": "0.3.3",
    "optuna": "3.6.1",
    "albumentations": "1.4.17",
    "cv2": "4.9.0", # Can also accept different opencv-python minor revisions
    "numpy": "1.26.4",
    "sklearn": "1.4.2"
}

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title} ".center(70, "="))
    print("=" * 70)

def main():
    print_header("PROYECTO DFU - VERIFICACIÓN DE ENTORNO")
    
    # Python Version Check
    py_version = sys.version.split()[0]
    print(f"Python Version: {py_version} (Expected: 3.11.7)")
    if sys.version_info[:2] != (3, 11):
        print("[-] WARNING: Recommended Python version is 3.11.x.")
    else:
        print("[+] PASS: Python version is 3.11.x.")

    results = {}
    
    # Check PyTorch
    try:
        import torch
        results["PyTorch"] = {
            "installed": True,
            "version": torch.__version__,
            "expected": EXPECTED_VERSIONS["torch"],
            "pass": torch.__version__.startswith(EXPECTED_VERSIONS["torch"])
        }
    except ImportError:
        results["PyTorch"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["torch"], "pass": False}

    # Check CUDA
    if results["PyTorch"]["installed"]:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        results["CUDA"] = {
            "installed": cuda_available,
            "version": f"Available (CUDA {cuda_version}) - GPU: {gpu_name}" if cuda_available else "Not Available",
            "expected": "Available (with GPU support)",
            "pass": cuda_available
        }
    else:
        results["CUDA"] = {"installed": False, "version": "N/A (PyTorch missing)", "expected": "Available", "pass": False}

    # Check segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp
        results["segmentation_models_pytorch"] = {
            "installed": True,
            "version": smp.__version__,
            "expected": EXPECTED_VERSIONS["segmentation_models_pytorch"],
            "pass": smp.__version__.startswith(EXPECTED_VERSIONS["segmentation_models_pytorch"])
        }
    except ImportError:
        results["segmentation_models_pytorch"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["segmentation_models_pytorch"], "pass": False}

    # Check Optuna
    try:
        import optuna
        results["Optuna"] = {
            "installed": True,
            "version": optuna.__version__,
            "expected": EXPECTED_VERSIONS["optuna"],
            "pass": optuna.__version__.startswith(EXPECTED_VERSIONS["optuna"])
        }
    except ImportError:
        results["Optuna"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["optuna"], "pass": False}

    # Check Albumentations
    try:
        import albumentations as A
        results["Albumentations"] = {
            "installed": True,
            "version": A.__version__,
            "expected": EXPECTED_VERSIONS["albumentations"],
            "pass": A.__version__.startswith(EXPECTED_VERSIONS["albumentations"])
        }
    except ImportError:
        results["Albumentations"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["albumentations"], "pass": False}

    # Check OpenCV
    try:
        import cv2
        results["OpenCV"] = {
            "installed": True,
            "version": cv2.__version__,
            "expected": EXPECTED_VERSIONS["cv2"],
            "pass": cv2.__version__.startswith(EXPECTED_VERSIONS["cv2"]) or cv2.__version__.startswith("4.")
        }
    except ImportError:
        results["OpenCV"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["cv2"], "pass": False}

    # Check NumPy
    try:
        import numpy as np
        results["NumPy"] = {
            "installed": True,
            "version": np.__version__,
            "expected": EXPECTED_VERSIONS["numpy"],
            "pass": np.__version__.startswith(EXPECTED_VERSIONS["numpy"])
        }
    except ImportError:
        results["NumPy"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["numpy"], "pass": False}

    # Check Scikit-learn
    try:
        import sklearn
        results["Scikit-learn"] = {
            "installed": True,
            "version": sklearn.__version__,
            "expected": EXPECTED_VERSIONS["sklearn"],
            "pass": sklearn.__version__.startswith(EXPECTED_VERSIONS["sklearn"])
        }
    except ImportError:
        results["Scikit-learn"] = {"installed": False, "version": "None", "expected": EXPECTED_VERSIONS["sklearn"], "pass": False}

    print_header("DETALLES DE LIBRERÍAS")
    all_passed = True
    
    for lib, info in results.items():
        status_str = "[+] PASS" if info["pass"] else "[-] FAIL"
        if not info["pass"]:
            all_passed = False
            
        print(f"{lib:<30} | {status_str:<8} | Inst: {info['version']:<35} (Exp: {info['expected']})")

    print_header("DIAGNÓSTICO FINAL")
    if all_passed:
        print("[SUCCESS] ¡El entorno está configurado correctamente y listo para ejecutar el proyecto!")
        sys.exit(0)
    else:
        print("[ERROR] Faltan dependencias críticas o no coinciden las versiones requeridas.")
        print("Por favor, ejecute el script 'scripts/setup_environment.sh' para configurar el entorno.")
        sys.exit(1)

if __name__ == "__main__":
    main()
