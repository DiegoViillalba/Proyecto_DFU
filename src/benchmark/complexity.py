#!/usr/bin/env python3
import os
import sys
import time
import csv
import importlib.util
import torch
from thop import profile

# Resolve repo root relative to this file (which is at src/benchmark/complexity.py)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def profile_model(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Profiles a PyTorch model to compute:
    - Number of parameters
    - GFLOPs (using thop)
    - Estimated memory (MB) during inference
    - Inference time (ms)
    
    Works for any PyTorch architecture.
    """
    model.eval()
    
    # Try to use CUDA if requested, otherwise fallback to CPU
    if device == "cuda" and torch.cuda.is_available():
        actual_device = torch.device("cuda")
    else:
        actual_device = torch.device("cpu")
        
    try:
        model = model.to(actual_device)
        x = torch.randn(*input_size).to(actual_device)
        # Verify a simple forward pass works
        with torch.no_grad():
            _ = model(x)
    except Exception as e:
        print(f"[Warning] Failed to run on {actual_device} device. Falling back to CPU. Error: {e}")
        actual_device = torch.device("cpu")
        model = model.to(actual_device)
        x = torch.randn(*input_size).to(actual_device)
        
    # 1. FLOPs and Params using thop
    try:
        flops, params = profile(model, inputs=(x,), verbose=False)
        gflops = flops / 1e9
    except Exception as e:
        print(f"[Warning] thop profiling failed: {e}. Calculating parameters manually.")
        params = sum(p.numel() for p in model.parameters())
        gflops = 0.0
        
    # 2. Estimate memory (MB)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)
    weight_mem = param_memory + buffer_memory
    
    if actual_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                _ = model(x)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            memory_mb = max(weight_mem, peak_mem)
        except Exception:
            memory_mb = weight_mem
    else:
        # Theoretical estimate: weight memory + input memory
        input_mem = (x.nelement() * x.element_size()) / (1024 ** 2)
        memory_mb = weight_mem + input_mem
        
    # 3. Inference time (ms)
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = model(x)
            
        if actual_device.type == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        num_iters = 50
        for _ in range(num_iters):
            _ = model(x)
            
        if actual_device.type == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
    inference_ms = ((end_time - start_time) / num_iters) * 1000.0
    
    return {
        "params": int(params),
        "gflops": round(gflops, 4),
        "memory_mb": round(memory_mb, 2),
        "inference_ms": round(inference_ms, 2)
    }

def load_class_from_file(file_path, class_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main():
    print("=" * 60)
    print("INICIANDO CÁLCULO DE COMPLEJIDAD DE MODELOS DFU")
    print("=" * 60)
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de benchmark preferido: {device.upper()}")
    
    # Models to benchmark: (ModelName, FilePath, ClassName)
    models_to_benchmark = [
        ("Classification_ResUnet", "src/ClasificationAlgorithms/Models/ResUnet/main.py", "ResUnet"),
        ("Classification_Unet", "src/ClasificationAlgorithms/Models/Unet/main.py", "UNET"),
        ("Classification_UnetPlusPlus", "src/ClasificationAlgorithms/Models/Unet++/main.py", "UnetPlusPlus"),
        ("Segmentation_AttentionUnet", "src/SegmentationNetworks/Models/AttUnet/main.py", "AttentionUnet"),
        ("Segmentation_ResUnet", "src/SegmentationNetworks/Models/ResUnet/main.py", "ResUnet"),
        ("Segmentation_Unet", "src/SegmentationNetworks/Models/Unet/main.py", "UNET"),
        ("Segmentation_UnetPlusPlus", "src/SegmentationNetworks/Models/Unet++/main.py", "UnetPlusPlus"),
        ("Segmentation_Carvana_Unet", "src/SegmentationNetworks/Models/Carvana_Unet_no_data/Unet_Pytorch/model.py", "UNET")
    ]
    
    results = []
    
    for name, rel_path, class_name in models_to_benchmark:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(abs_path):
            print(f"[-] Archivo no encontrado para {name}: {rel_path}")
            continue
            
        print(f"\n[+] Cargando y evaluando {name} ({class_name})...")
        try:
            # Load class dynamically
            model_class = load_class_from_file(abs_path, class_name)
            
            # Instantiate model with default parameters
            # All of them default to in_channels=3, out_channels=1
            model = model_class()
            
            # Profile
            metrics = profile_model(model, input_size=(1, 3, 224, 224), device=device)
            
            results.append({
                "model": name,
                "params": metrics["params"],
                "gflops": metrics["gflops"],
                "memory_mb": metrics["memory_mb"],
                "inference_ms": metrics["inference_ms"]
            })
            
            print(f"    Params: {metrics['params']:,}")
            print(f"    GFLOPs: {metrics['gflops']:.4f}")
            print(f"    Memory: {metrics['memory_mb']:.2f} MB")
            print(f"    Inference Time: {metrics['inference_ms']:.2f} ms")
            
        except Exception as e:
            print(f"[-] Error evaluando {name}: {e}")
            
    # Write to CSV
    csv_path = os.path.join(REPO_ROOT, "results/model_complexity.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "params", "gflops", "memory_mb", "inference_ms"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print("\n" + "=" * 60)
        print(f"[SUCCESS] Resultados guardados en: results/model_complexity.csv")
        print("=" * 60)
    except Exception as e:
        print(f"\n[-] Error al escribir resultados en CSV: {e}")

if __name__ == "__main__":
    main()
