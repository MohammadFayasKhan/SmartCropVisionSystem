#!/usr/bin/env python3
"""
train_plantdoc_yolo_efficient.py
---------------------------------
Memory-Safe, Power-Resilient YOLOv8-nano Training on PlantDoc Field Dataset.

Features:
- Isolated Python process (zero memory pollution from prior models)
- Explicit MPS cache clearing and garbage collection
- Low-memory footprint (batch=8, workers=2, imgsz=256)
- Resilient to battery throttling on Apple Silicon MacBooks
- Automatic export to cv/models/yolov8n_plantdoc_best.pt
"""

import os
import gc
import sys
import shutil
import time
from pathlib import Path
import torch

# Ensure MPS fallback for unsupported operators
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def print_banner(text):
    line = "=" * 80
    print(f"\n{line}\n{text}\n{line}")

def train_plantdoc_efficient():
    print_banner("🌱 EFFICIENT, MEMORY-SAFE PLANTDOC YOLOV8-NANO TRAINING")
    
    # 1. Device & Memory Diagnostics
    device_target = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Compute Target Device  : {device_target.upper()}")
    print(f"[*] Active Workspace       : {Path.cwd()}")
    
    # Garbage collection before training
    gc.collect()
    if device_target == "mps":
        torch.mps.empty_cache()
    print("[+] Cleared active GPU/MPS cache buffers.")

    # 2. Config Verification
    yaml_path = Path("cv/configs/yolo_plantdoc.yaml")
    if not yaml_path.exists():
        print(f"[!] Error: Configuration file {yaml_path} not found.")
        sys.exit(1)
        
    print(f"[*] PlantDoc YAML Path     : {yaml_path.resolve()}")
    
    # Check dataset existence
    train_dir = Path("cv/datasets/processed/yolo_plantdoc/images/train")
    val_dir = Path("cv/datasets/processed/yolo_plantdoc/images/val")
    num_train = len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0
    num_val = len(list(val_dir.glob("*.*"))) if val_dir.exists() else 0
    print(f"[*] Train Partition Size   : {num_train:,} real-world field images")
    print(f"[*] Val Partition Size     : {num_val:,} field validation images")
    
    if num_train == 0:
        print("[!] Error: No training images found in yolo_plantdoc directory.")
        sys.exit(1)

    # 3. Load YOLOv8-nano Foundation Weights
    from ultralytics import YOLO
    
    pretrained_weights = Path("cv/models/yolov8n.pt")
    if not pretrained_weights.exists():
        print("[*] Downloading baseline yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    else:
        print(f"[*] Using local baseline   : {pretrained_weights.resolve()}")
        model = YOLO(str(pretrained_weights))

    # 4. Train with Strict Memory & Battery Safety Guardrails
    epochs = 10
    batch_size = 8      # Safe footprint: ~1.2 GB peak RAM vs 6 GB+ at batch=32
    imgsz = 256         # Optimal trade-off for mobile foliar pathology
    workers = 2         # Prevents socket connection exhaustion on macOS
    
    print(f"\n[*] Training Hyperparameters:")
    print(f"    - Epochs       : {epochs}")
    print(f"    - Batch Size   : {batch_size} (Memory Safe)")
    print(f"    - Image Size   : {imgsz}x{imgsz}")
    print(f"    - Workers      : {workers}")
    print(f"    - Device       : {device_target}")
    
    t0 = time.time()
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device_target,
        project="cv/models",
        name="yolov8n_plantdoc_efficient",
        exist_ok=True,
        save=True,
        plots=False,        # Disable matplotlib rendering during training to avoid memory leaks
        verbose=True,
        amp=False           # Disable AMP on MPS to prevent instability
    )
    training_duration = time.time() - t0
    
    # 5. Export Best Weights
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    dest_weights = Path("cv/models/yolov8n_plantdoc_best.pt")
    
    if best_weights.exists():
        shutil.copy(str(best_weights), str(dest_weights))
        model_size_mb = dest_weights.stat().st_size / (1024 * 1024)
        print_banner(f"✅ TRAINING COMPLETE & EXPORTED SUCCESSFULLY\n"
                     f"[*] Duration          : {training_duration / 60:.2f} minutes\n"
                     f"[*] Exported Checkpoint: {dest_weights.resolve()}\n"
                     f"[*] Model Footprint   : {model_size_mb:.2f} MB")
    else:
        print(f"[!] Warning: {best_weights} not found. Check {save_dir}")

    # Final cleanup
    gc.collect()
    if device_target == "mps":
        torch.mps.empty_cache()
    print("[+] Cleaned up memory cache.")

if __name__ == "__main__":
    train_plantdoc_efficient()
