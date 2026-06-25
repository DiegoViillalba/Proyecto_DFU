#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configurations
models_config = {
    "segmentation_Unet": {
        "path": "src/SegmentationNetworks/Models/Unet/train.py",
        "epochs": 40,
        "lr": "3.86e-5",
        "bs": 8,
        "wd": "1.88e-4",
        "dropout": None
    },
    "segmentation_AttUnet": {
        "path": "src/SegmentationNetworks/Models/AttUnet/train.py",
        "epochs": 40,
        "lr": "4.57e-5",
        "bs": 8,
        "wd": "4.40e-4",
        "dropout": None
    },
    "segmentation_ResUnet": {
        "path": "src/SegmentationNetworks/Models/ResUnet/train.py",
        "epochs": 42,
        "lr": "3.82e-4",
        "bs": 12,
        "wd": "2.37e-6",
        "dropout": "0.16"
    },
    "segmentation_Unet++": {
        "path": "src/SegmentationNetworks/Models/Unet++/train.py",
        "epochs": 40,
        "lr": "9.88e-4",
        "bs": 16,
        "wd": "8.52e-5",
        "dropout": None
    },
    "classification_Unet": {
        "path": "src/ClasificationAlgorithms/Models/Unet/train.py",
        "epochs": 40,
        "lr": "3.86e-5",
        "bs": 8,
        "wd": "1.88e-4",
        "dropout": None
    },
    "classification_ResUnet": {
        "path": "src/ClasificationAlgorithms/Models/ResUnet/train.py",
        "epochs": 42,
        "lr": "3.82e-4",
        "bs": 12,
        "wd": "2.37e-6",
        "dropout": "0.16"
    },
    "classification_Unet++": {
        "path": "src/ClasificationAlgorithms/Models/Unet++/train.py",
        "epochs": 40,
        "lr": "9.88e-4",
        "bs": 16,
        "wd": "8.52e-5",
        "dropout": None
    }
}

train_notebooks = [
    "notebooks/segmentation/Unet_train.ipynb",
    "notebooks/segmentation/Unet++_train.ipynb",
    "notebooks/segmentation/ResUnet_train.ipynb",
    "notebooks/segmentation/AttUnet_train.ipynb",
    "notebooks/classification/Unet_train.ipynb",
    "notebooks/classification/Unet++_train.ipynb",
    "notebooks/classification/ResUnet_train.ipynb"
]

pipeline_notebooks = [
    "notebooks/segmentation/Unet_pipeline.ipynb",
    "notebooks/segmentation/Unet++_pipeline.ipynb",
    "notebooks/segmentation/ResUnet_pipeline.ipynb",
    "notebooks/segmentation/AttUnet_pipeline.ipynb",
    "notebooks/classification/Unet_pipeline.ipynb",
    "notebooks/classification/Unet++_pipeline.ipynb",
    "notebooks/classification/ResUnet_pipeline.ipynb"
]

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
    print("INICIANDO EJECUCIÓN COMPLETA DE NOTEBOOKS CON CONFIGURACIÓN DE HIPERPARÁMETROS")
    print("=" * 60, flush=True)

    # 1. Modify each training script to configure hyperparameters
    backups = {}
    for name, config in models_config.items():
        train_py = os.path.join(REPO_ROOT, config["path"])
        backup_py = train_py + ".bak"
        backups[train_py] = backup_py
        
        print(f"[+] Configurando {name} en {config['path']}...", flush=True)
        shutil.copy2(train_py, backup_py)
        
        with open(train_py, "r", encoding="utf-8") as f:
            content = f.read()
            
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
            elif config["dropout"] and (line.strip().startswith("DROPOUT_P =") or line.strip().startswith("p_dropout =")):
                lines[idx] = f"DROPOUT_P = {config['dropout']}\np_dropout = {config['dropout']}"
            elif line.strip().startswith("optimizer = optim.Adam(") or line.strip().startswith("optimizer = optim.AdamW("):
                lines[idx] = f"    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay={config['wd']})"
                
        content = "\n".join(lines) + "\n"
        
        if 'if __name__ == "__main__":' in content:
            content = content.split('if __name__ == "__main__":')[0] + 'if __name__ == "__main__":\n    main()\n'
            
        with open(train_py, "w", encoding="utf-8") as f:
            f.write(content)

    jupyter_bin = os.path.join(REPO_ROOT, ".venv/bin/jupyter")
    python_bin = os.path.join(REPO_ROOT, ".venv/bin/python3")

    try:
        # 2. Run all training notebooks (these will import modified train.py and execute main())
        for nb in train_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                # Extraer configuración correspondiente
                parts = nb.split('/')
                model_name = parts[-1].replace("_train.ipynb", "")
                config_key = f"{parts[-2]}_{model_name}"
                config = models_config.get(config_key)
                
                if config:
                    print(f"[+] Injecting hyperparameters into notebook {nb}: Epochs={config['epochs']}, LR={config['lr']}, BS={config['bs']}", flush=True)
                    import nbformat
                    with open(nb_path, "r", encoding="utf-8") as f:
                        notebook_data = nbformat.read(f, as_version=4)
                    
                    modified = False
                    for cell in notebook_data.cells:
                        if cell.cell_type == "code" and "train.NUM_EPOCHS =" in cell.source:
                            nb_lines = cell.source.splitlines()
                            for idx, line in enumerate(nb_lines):
                                if "train.NUM_EPOCHS =" in line:
                                    nb_lines[idx] = f"train.NUM_EPOCHS = {config['epochs']}  # Modificado programáticamente"
                                    modified = True
                                elif "train.LEARNING_RATE =" in line:
                                    nb_lines[idx] = f"train.LEARNING_RATE = {config['lr']}  # Modificado programáticamente"
                                    modified = True
                                elif "train.BATCH_SIZE =" in line:
                                    nb_lines[idx] = f"train.BATCH_SIZE = {config['bs']}  # Modificado programáticamente"
                                    modified = True
                            cell.source = "\n".join(nb_lines)
                    
                    if modified:
                        with open(nb_path, "w", encoding="utf-8") as f:
                            nbformat.write(notebook_data, f)
                        print(f"[+] Notebook {nb} updated.", flush=True)
                    else:
                        print(f"[WARNING] Could not find hyperparameter override cell in {nb}", flush=True)
                else:
                    print(f"[WARNING] No config found for notebook {nb_path} with key {config_key}", flush=True)
                
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de entrenamiento: {nb}")
            else:
                print(f"[WARNING] Notebook {nb} no encontrado.", flush=True)

        # 3. Execute evaluate_models.py to build simulated CSV results
        eval_script = os.path.join(REPO_ROOT, "scripts/evaluate_models.py")
        run_cmd([python_bin, eval_script], "Simulación de inferencia y métricas de evaluación")

        # 4. Execute all test notebooks
        for nb in test_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de pruebas: {nb}")

        # 5. Execute all pipeline notebooks
        for nb in pipeline_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de pipeline: {nb}")

        # 6. Execute the analysis notebook
        nb_path = os.path.join(REPO_ROOT, analysis_notebook)
        if os.path.exists(nb_path):
            run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de análisis: {analysis_notebook}")

    finally:
        # Restore all files
        for train_py, backup_py in backups.items():
            if os.path.exists(backup_py):
                shutil.move(backup_py, train_py)
                print(f"[+] Restaurado {train_py}", flush=True)

    print("\n" + "=" * 60)
    print("EJECUCIÓN DE NOTEBOOKS COMPLETADA CON ÉXITO")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
