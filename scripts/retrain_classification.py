#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configurations for Classification
classification_models = {
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
    "notebooks/classification/Unet_train.ipynb",
    "notebooks/classification/Unet++_train.ipynb",
    "notebooks/classification/ResUnet_train.ipynb"
]

test_notebooks = [
    "notebooks/classification/Unet_test.ipynb",
    "notebooks/classification/Unet++_test.ipynb",
    "notebooks/classification/ResUnet_test.ipynb"
]

pipeline_notebooks = [
    "notebooks/classification/Unet_pipeline.ipynb",
    "notebooks/classification/Unet++_pipeline.ipynb",
    "notebooks/classification/ResUnet_pipeline.ipynb"
]

checkpoint_mappings = [
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint2.pth",
        "src/ClasificationAlgorithms/Models/Unet/output_assets_model/best_model_checkpoint2.pth"
    ),
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint2.zip",
        "src/ClasificationAlgorithms/Models/Unet/output_assets_model/best_model_checkpoint2.zip"
    ),
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint_Unet++.pth",
        "src/ClasificationAlgorithms/Models/Unet++/output_assets_model/best_model_checkpoint_Unet++.pth"
    ),
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint_Unet++.zip",
        "src/ClasificationAlgorithms/Models/Unet++/output_assets_model/best_model_checkpoint_Unet++.zip"
    ),
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint_ResUnet.pth",
        "src/ClasificationAlgorithms/Models/ResUnet/output_assets_model/best_model_checkpoint_ResUnet.pth"
    ),
    (
        "notebooks/classification/output_assets_model/best_model_checkpoint_ResUnet.zip",
        "src/ClasificationAlgorithms/Models/ResUnet/output_assets_model/best_model_checkpoint_ResUnet.zip"
    ),
]

analysis_notebooks = [
    "notebooks/analysis_results.ipynb",
    "notebooks/model_analysis/analyze_models.ipynb",
    "notebooks/model_analysis/qualitative_analysis.ipynb"
]

jupyter_bin = os.path.join(REPO_ROOT, ".venv/bin/jupyter")
python_bin = os.path.join(REPO_ROOT, ".venv/bin/python3")

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
    print("INICIANDO RE-ENTRENAMIENTO Y RE-EVALUACIÓN DE CLASIFICACIÓN (DFU TISSUE)")
    print("=" * 60, flush=True)

    # 1. Modify each training script on disk to configure hyperparameters
    backups = {}
    for name, config in classification_models.items():
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

    try:
        # 2. Run classification training notebooks
        for nb in train_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                # Extraer configuración correspondiente
                parts = nb.split('/')
                model_name = parts[-1].replace("_train.ipynb", "")
                config_key = f"classification_{model_name}"
                config = classification_models.get(config_key)
                
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
                
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de entrenamiento: {nb}")
            else:
                print(f"[WARNING] Notebook {nb} no encontrado.", flush=True)

        # 3. Copy newly trained checkpoints to src folder
        print("\n[+] Sincronizando checkpoints con la carpeta src...", flush=True)
        for src_rel, dst_rel in checkpoint_mappings:
            src_path = os.path.join(REPO_ROOT, src_rel)
            dst_path = os.path.join(REPO_ROOT, dst_rel)
            
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_rel} -> {dst_rel}", flush=True)
            else:
                print(f"Warning: Checkpoint not found at {src_rel}", flush=True)

        # 4. Run classification test notebooks
        for nb in test_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de pruebas: {nb}")

        # 5. Run classification pipeline notebooks
        for nb in pipeline_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de pipeline: {nb}")

        # 6. Run evaluate_models.py to rebuild consolidated metrics
        eval_script = os.path.join(REPO_ROOT, "scripts/evaluate_models.py")
        run_cmd([python_bin, eval_script], "Evaluación comparativa de inferencia")

        # 7. Execute all analysis notebooks to update figures and reports
        for nb in analysis_notebooks:
            nb_path = os.path.join(REPO_ROOT, nb)
            if os.path.exists(nb_path):
                run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de análisis: {nb}")

        # 8. Re-generate validation report text file
        print("\n[+] Regenerando el reporte validation_report.md con métricas actualizadas...", flush=True)
        df_metrics = pd.read_csv(os.path.join(REPO_ROOT, "results/tables/dfutissue_metrics.csv"))
        summary_stats = df_metrics.groupby("model")[["dsc", "iou", "precision", "recall", "hd95", "assd"]].mean().reset_index()
        summary_stats = summary_stats.sort_values(by="dsc", ascending=False)
        
        # Read validation_report template content and update the tables
        # Let's read the report, replace the table, and write it back
        report_file_path = os.path.join(REPO_ROOT, "results/report/validation_report.md")
        if os.path.exists(report_file_path):
            with open(report_file_path, "r", encoding="utf-8") as f:
                rep_lines = f.readlines()
            
            # Find the table boundaries
            start_table_idx = -1
            end_table_idx = -1
            for idx, line in enumerate(rep_lines):
                if "| Model | DSC ↑ | IoU |" in line or "| Model | DSC ↑ | IoU ↑ |" in line:
                    start_table_idx = idx
                elif start_table_idx != -1 and line.strip() == "" and idx > start_table_idx + 2:
                    end_table_idx = idx
                    break
            
            if start_table_idx != -1:
                # Generate new table
                new_table = [
                    "| Model | DSC ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 (px) ↓ | ASSD (px) ↓ |\n",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
                ]
                for _, row in summary_stats.iterrows():
                    new_table.append(f"| **{row['model']}** | {row['dsc']:.4f} | {row['iou']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['hd95']:.2f} px | {row['assd']:.2f} px |\n")
                
                # Replace
                rep_lines = rep_lines[:start_table_idx] + new_table + rep_lines[end_table_idx:]
                with open(report_file_path, "w", encoding="utf-8") as f:
                    f.writelines(rep_lines)
                print("[SUCCESS] results/report/validation_report.md updated with new metrics.", flush=True)

    finally:
        # Restore backups
        for train_py, backup_py in backups.items():
            if os.path.exists(backup_py):
                shutil.move(backup_py, train_py)
                print(f"[+] Restaurado {train_py}", flush=True)

    print("\n" + "=" * 60)
    print("PROCESO DE RE-ENTRENAMIENTO Y RE-EVALUACIÓN COMPLETADO CON ÉXITO")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
