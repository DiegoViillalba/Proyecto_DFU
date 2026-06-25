#!/usr/bin/env python3
import os
import subprocess
import pandas as pd

REPO_ROOT = "/home/diego-villalba/Proyecto_DFU"
jupyter_bin = os.path.join(REPO_ROOT, ".venv/bin/jupyter")
python_bin = os.path.join(REPO_ROOT, ".venv/bin/python3")

analysis_notebooks = [
    "notebooks/analysis_results.ipynb",
    "notebooks/model_analysis/analyze_models.ipynb",
    "notebooks/model_analysis/qualitative_analysis.ipynb"
]

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
    print("EJECUTANDO EVALUACIONES FINALES DE NOTEBOOKS DESPUÉS DE RE-ENTRENAMIENTO")
    print("=" * 60, flush=True)
    
    # 1. Run evaluate_models.py to update metrics and boxplots
    eval_script = os.path.join(REPO_ROOT, "scripts/evaluate_models.py")
    run_cmd([python_bin, eval_script], "Evaluación comparativa de inferencia")

    # 2. Execute all analysis notebooks to update figures and reports
    for nb in list(analysis_notebooks):
        nb_path = os.path.join(REPO_ROOT, nb)
        if os.path.exists(nb_path):
            run_cmd([jupyter_bin, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], f"Notebook de análisis: {nb}")

    # 3. Update results/report/validation_report.md
    print("\n[+] Actualizando validation_report.md con métricas de re-entrenamiento...", flush=True)
    df_metrics = pd.read_csv(os.path.join(REPO_ROOT, "results/tables/dfutissue_metrics.csv"))
    summary_stats = df_metrics.groupby("model")[["dsc", "iou", "precision", "recall", "hd95", "assd"]].mean().reset_index()
    summary_stats = summary_stats.sort_values(by="dsc", ascending=False)
    
    report_file_path = os.path.join(REPO_ROOT, "results/report/validation_report.md")
    if os.path.exists(report_file_path):
        with open(report_file_path, "r", encoding="utf-8") as f:
            rep_lines = f.readlines()
        
        start_table_idx = -1
        end_table_idx = -1
        for idx, line in enumerate(rep_lines):
            if "| Model | DSC ↑ | IoU |" in line or "| Model | DSC ↑ | IoU ↑ |" in line:
                start_table_idx = idx
            elif start_table_idx != -1 and line.strip() == "" and idx > start_table_idx + 2:
                end_table_idx = idx
                break
        
        if start_table_idx != -1:
            new_table = [
                "| Model | DSC ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 (px) ↓ | ASSD (px) ↓ |\n",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            ]
            for _, row in summary_stats.iterrows():
                new_table.append(f"| **{row['model']}** | {row['dsc']:.4f} | {row['iou']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['hd95']:.2f} px | {row['assd']:.2f} px |\n")
            
            rep_lines = rep_lines[:start_table_idx] + new_table + rep_lines[end_table_idx:]
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.writelines(rep_lines)
            print("[SUCCESS] results/report/validation_report.md updated.", flush=True)

if __name__ == "__main__":
    main()
