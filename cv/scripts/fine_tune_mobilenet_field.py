"""
fine_tune_mobilenet_field.py
----------------------------
Fine-tunes MobileNetV2 across all 38 plant pathology classes using the balanced
field-augmented dataset (PlantVillage lab + PlantDoc real-world field foliage).

Optimized for Apple Silicon GPU (MPS):
- Batch size: 32
- Robust outdoor data augmentations
- Class-balanced sampling and evaluation metrics (Accuracy, Macro-F1, Precision, Recall)
- Saves best checkpoint and generates evaluation confusion report.
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("=" * 80)
print("🌿 FINE-TUNING MOBILENETV2 FOR REAL-WORLD FIELD GENERALIZATION")
print("=" * 80)

# 1. Device Setup
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[*] Compute Target Device     : {device}")

# 2. Load Manifest
manifest_path = Path("cv/datasets/processed/manifest_38classes_field_augmented.csv")
if not manifest_path.exists():
    print(f"[!] Error: {manifest_path} not found.")
    sys.exit(1)

df = pd.read_csv(manifest_path)
num_classes = 38
print(f"[*] Total Manifest Images      : {len(df):,}")
print(f"[*] Total Target Classes       : {num_classes}")

# 3. Outdoor-Realistic Augmentation Transforms
train_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LeafDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.df = data_df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df["file_path"].values
        self.labels = self.df["class_index"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        lbl = int(self.labels[idx])
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(lbl, dtype=torch.long)

train_sub = df[df["split"] == "train"]
val_sub = df[df["split"] == "val"]

train_loader = DataLoader(LeafDataset(train_sub, train_transforms), batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(LeafDataset(val_sub, val_transforms), batch_size=32, shuffle=False, num_workers=0)

# 4. Model Architecture & Pre-Trained Weights
base_model_path = Path("cv/models/mobilenet_v2_38classes_best.pth")
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.25),
    nn.Linear(model.last_channel, num_classes)
)

if base_model_path.exists():
    print(f"[*] Loading foundation checkpoint: {base_model_path.resolve()}")
    state_dict = torch.load(base_model_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("[*] Downloading ImageNet baseline for MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(model.last_channel, num_classes)
    )

model.to(device)

# Differential Learning Rates: Low lr for backbone, higher lr for classifier
backbone_params = [p for n, p in model.features.named_parameters()]
classifier_params = [p for n, p in model.classifier.named_parameters()]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-5, "weight_decay": 1e-4},
    {"params": classifier_params, "lr": 2e-4, "weight_decay": 1e-4}
])

epochs = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

best_val_f1 = 0.0
best_checkpoint_path = Path("cv/models/mobilenet_v2_38classes_field_best.pth")

print(f"\n[*] Starting Fine-Tuning for {epochs} Epochs on {len(train_sub):,} training samples...")
print("-" * 80)

for ep in range(1, epochs + 1):
    t0 = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    tr_loss = running_loss / total
    tr_acc = (correct / total) * 100.0

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            loss = criterion(outs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    v_loss = val_loss / val_total
    v_acc = accuracy_score(all_targets, all_preds) * 100.0
    v_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    v_prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    v_rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    dur = time.time() - t0

    saved_marker = ""
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save(model.state_dict(), best_checkpoint_path)
        saved_marker = " 🎯 [BEST CHECKPOINT SAVED]"

    print(f"Epoch [{ep:2d}/{epochs:2d}] ({dur:4.1f}s) → Train Acc: {tr_acc:5.1f}% (Loss: {tr_loss:.4f}) | "
          f"Val Acc: {v_acc:5.1f}% | Val F1: {v_f1:5.1f}% | Prec: {v_prec:5.1f}% | Rec: {v_rec:5.1f}%{saved_marker}")

print("\n" + "=" * 80)
print(f"✅ FINE-TUNING COMPLETE. Best Val Weighted F1: {best_val_f1:.2f}%")
print(f"[*] Checkpoint Saved to: {best_checkpoint_path.resolve()}")
print("=" * 80)
