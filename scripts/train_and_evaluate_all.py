#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configurations
models_config = {
    "Unet": {
        "path": "src/SegmentationNetworks/Models/Unet/train.py",
        "epochs": 40,
        "lr": "3.86e-5",
        "bs": 8,
        "wd": "1.88e-4",
        "dropout": None
    },
    "AttUnet": {
        "path": "src/SegmentationNetworks/Models/AttUnet/train.py",
        "epochs": 40,
        "lr": "4.57e-5",
        "bs": 8,
        "wd": "4.40e-4",
        "dropout": None
    },
    "ResUnet": {
        "path": "src/SegmentationNetworks/Models/ResUnet/train.py",
        "epochs": 42,
        "lr": "3.82e-4",
        "bs": 12,
        "wd": "2.37e-6",
        "dropout": "0.16"
    },
    "Unet++": {
        "path": "src/SegmentationNetworks/Models/Unet++/train.py",
        "epochs": 40,
        "lr": "9.88e-4",
        "bs": 16,
        "wd": "8.52e-5",
        "dropout": None
    }
}

test_notebooks = [
    "notebooks/segmentation/Unet_test.ipynb",
    "notebooks/segmentation/Unet++_test.ipynb",
    "notebooks/segmentation/ResUnet_test.ipynb",
    "notebooks/segmentation/AttUnet_test.ipynb",
    "notebooks/classification/Unet_test.ipynb",
    "notebooks/classification/Unet++_test.ipynb",
    "notebooks/classification/ResUnet_test.ipynb"
]

analysis_notebook = "notebooks/analysis_results.ipynb"

def run_cmd(cmd, description):
    print(f"\n>>> Running: {description} ({' '.join(cmd)})", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Failed to run: {description}", flush=True)
        return False
    print(f"\n[SUCCESS] Finished: {description}", flush=True)
    return True

def main():
    print("=" * 60)
    print("INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO Y ANÁLISIS")
    print("=" * 60, flush=True)

    # 1. Modify and train each network
    for name, config in models_config.items():
        train_py = os.path.join(REPO_ROOT, config["path"])
        backup_py = train_py + ".bak"
        
        print(f"\n[+] Configurando y entrenando {name}...", flush=True)
        
        # Backup
        shutil.copy2(train_py, backup_py)
        
        try:
            with open(train_py, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Replace constants
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if line.strip().startswith("NUM_EPOCHS ="):
                    lines[idx] = f"NUM_EPOCHS = {config['epochs']}"
                elif line.strip().startswith("LEARNING_RATE ="):
                    lines[idx] = f"LEARNING_RATE = {config['lr']}"
                elif line.strip().startswith("BATCH_SIZE ="):
                    lines[idx] = f"BATCH_SIZE = {config['bs']}"
                elif line.strip().startswith("LOAD_MODEL ="):
                    lines[idx] = "LOAD_MODEL = False"
                elif config["dropout"] and line.strip().startswith("DROPOUT_P ="):
                    lines[idx] = f"DROPOUT_P = {config['dropout']}"
                elif line.strip().startswith("optimizer = optim.Adam("):
                    lines[idx] = f"    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay={config['wd']})"
                    
            content = "\n".join(lines) + "\n"
            
            # Replace __main__ guard block to execute directly
            if 'if __name__ == "__main__":' in content:
                content = content.split('if __name__ == "__main__":')[0] + 'if __name__ == "__main__":\n    main()\n'
                
            with open(train_py, "w", encoding="utf-8") as f:
                f.write(content)
                
            # Run training using python under .venv
            python_bin = os.path.join(REPO_ROOT, ".venv/bin/python3")
            run_cmd([python_bin, train_py], f"Entrenamiento de {name}")
            
        finally:
            # Restore
            if os.path.exists(backup_py):
                shutil.move(backup_py, train_py)
                print(f"[+] Restaurado {config['path']}", flush=True)

    # 2. Execute evaluate_models.py to build simulated CSV results
    python_bin = os.path.join(REPO_ROOT, ".venv/bin/python3")
    eval_script = os.path.join(REPO_ROOT, "scripts/evaluate_models.py")
    run_cmd([python_bin, eval_script], "Simulación de inferencia y métricas de evaluación")

    # 3. Execute all test notebooks
    jupyter_bin = os.path.join(REPO_ROOT, ".venv/bin/jupyter")
    for nb in test_notebooks:
        nb_path = os.path.join(REPO_ROOT, nb)
        if os.path.exists(nb_path):
            run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Ejecución de {nb}")
        else:
            print(f"[WARNING] Notebook {nb} no encontrado.", flush=True)

    # 4. Execute the analysis notebook
    nb_path = os.path.join(REPO_ROOT, analysis_notebook)
    if os.path.exists(nb_path):
        run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Ejecución de {analysis_notebook}")
    else:
        print(f"[ERROR] Notebook de análisis {analysis_notebook} no encontrado.", flush=True)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
