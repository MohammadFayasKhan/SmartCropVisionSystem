"""
train_server_grade_models.py
----------------------------
Modular, reproducible training engine for server-grade crop disease classifiers.
Features:
1. Leakage-free multi-domain dataset (PlantVillage lab + PlantDoc field).
2. Field-realistic outdoor augmentations (blur, noise, shadows, perspective, erasing).
3. Class-weighted Focal Loss with label smoothing.
4. 2-Phase Progressive Fine-Tuning (Head warmup followed by differential unfreezing).
5. Early stopping on Validation Macro F1.
6. Dual held-out evaluation:
   - Standard PlantVillage Lab Test Set (8,146 images)
   - Real-world PlantDoc Field Test Set (236 images)
7. Full metrics export (Top-1, Top-3, Macro/Micro F1, Precision, Recall, Balanced Acc).
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# ── Class-Weighted Focal Loss with Label Smoothing ────────────────────────────
class ClassWeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Label smoothing target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Focal term: (1 - p_t)^gamma
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - pt) ** self.gamma

        # Cross entropy loss per sample
        loss = -(smooth_targets * log_probs).sum(dim=1)
        loss = focal_factor * loss

        if self.weights is not None:
            w = self.weights[targets]
            loss = loss * w

        return loss.mean()

# ── Leaf Dataset ──────────────────────────────────────────────────────────────
class LeafMultiDomainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df["file_path"].values
        self.labels = self.df["class_index"].values
        self.domains = self.df["domain"].values if "domain" in self.df.columns else ["unknown"] * len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        lbl = int(self.labels[idx])
        domain = self.domains[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(lbl, dtype=torch.long), domain

# ── Model Builder ─────────────────────────────────────────────────────────────
def build_model(arch_name: str, num_classes: int = 38):
    arch_name = arch_name.lower().replace("-", "_")
    if arch_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        backbone_modules = [model.features]
        classifier_modules = [model.classifier]

    elif arch_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        backbone_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
        classifier_modules = [model.fc]

    elif arch_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        backbone_modules = [model.features]
        classifier_modules = [model.classifier]

    elif arch_name in ["efficientnet_v2_s", "efficientnet_v2"]:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        backbone_modules = [model.features]
        classifier_modules = [model.classifier]

    elif arch_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(model.last_channel, num_classes)
        )
        backbone_modules = [model.features]
        classifier_modules = [model.classifier]
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    return model, backbone_modules, classifier_modules

# ── Evaluator Function ────────────────────────────────────────────────────────
def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds, all_targets, all_domains = [], [], []
    top3_correct = 0

    with torch.no_grad():
        for imgs, lbls, domains in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            if criterion:
                loss = criterion(outs, lbls)
                total_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outs, dim=1)
            _, top3_p = probs.topk(3, dim=1)
            top3_correct += (top3_p == lbls.unsqueeze(1)).any(dim=1).sum().item()

            _, preds = outs.max(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(lbls.cpu().numpy().tolist())
            all_domains.extend(domains)
            total_count += lbls.size(0)

    avg_loss = (total_loss / total_count) if total_count > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    top3_acc = (top3_correct / total_count) * 100.0 if total_count > 0 else 0.0
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    # Domain breakdown
    field_mask = [d == "field" for d in all_domains]
    lab_mask = [d == "lab" for d in all_domains]

    field_acc = 0.0
    if any(field_mask):
        f_true = [t for t, m in zip(all_targets, field_mask) if m]
        f_pred = [p for p, m in zip(all_preds, field_mask) if m]
        field_acc = accuracy_score(f_true, f_pred) * 100.0
        field_f1 = f1_score(f_true, f_pred, average="macro", zero_division=0) * 100.0
    else:
        field_f1 = 0.0

    lab_acc = 0.0
    if any(lab_mask):
        l_true = [t for t, m in zip(all_targets, lab_mask) if m]
        l_pred = [p for p, m in zip(all_preds, lab_mask) if m]
        lab_acc = accuracy_score(l_true, l_pred) * 100.0
        lab_f1 = f1_score(l_true, l_pred, average="macro", zero_division=0) * 100.0
    else:
        lab_f1 = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "field_accuracy": field_acc,
        "field_macro_f1": field_f1,
        "lab_accuracy": lab_acc,
        "lab_macro_f1": lab_f1,
        "predictions": all_preds,
        "targets": all_targets
    }

# ── Main Training Loop ────────────────────────────────────────────────────────
def train_model(
    arch_name: str = "efficientnet_b2",
    img_size: int = 288,
    batch_size: int = 32,
    warmup_epochs: int = 2,
    fine_tune_epochs: int = 6,
    lr_head: float = 3e-4,
    lr_backbone: float = 3e-5,
    device_str: str = "mps",
    resume_from: str = None,
    manifest_path_str: str = "cv/datasets/processed/manifest_38classes_tridataset.csv"
):
    print("=" * 80)
    print(f"🚀 TRAINING SERVER-GRADE ARCHITECTURE: {arch_name.upper()}")
    print(f"[*] Resolution: {img_size}x{img_size} | Batch Size: {batch_size}")
    print(f"[*] Warmup Epochs: {warmup_epochs} | Fine-Tune Epochs: {fine_tune_epochs}")
    print(f"[*] Dataset Manifest: {manifest_path_str}")
    if resume_from:
        print(f"[*] Resuming from Checkpoint: {resume_from}")
    print("=" * 80)

    device = torch.device(device_str if (device_str == "mps" and torch.backends.mps.is_available()) or (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[*] Compute Target Device: {device}")

    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        manifest_path = Path("cv/datasets/processed/manifest_38classes_multidomain.csv")
    df = pd.read_csv(manifest_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_field_df = df[(df["split"] == "test") & (df["domain"] == "field")]
    test_lab_df = df[(df["split"] == "test") & (df["domain"] == "lab")]

    print(f"[*] Dataset Splits: Train={len(train_df):,}, Val={len(val_df):,}, Test Field={len(test_field_df):,}, Test Lab={len(test_lab_df):,}")

    # Compute smoothed class weights
    class_counts = train_df["class_index"].value_counts().sort_index()
    total_samples = len(train_df)
    num_classes = 38
    raw_weights = total_samples / (num_classes * class_counts.values.astype(float))
    smoothed_weights = np.sqrt(raw_weights)
    smoothed_weights = smoothed_weights / smoothed_weights.mean()
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float, device=device)

    criterion = ClassWeightedFocalLoss(weights=weights_tensor, gamma=2.0, label_smoothing=0.05)

    # Outdoor-Realistic Augmentation Pipeline
    train_transforms = transforms.Compose([
        transforms.Resize((int(img_size * 1.05), int(img_size * 1.05))),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random")
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(LeafMultiDomainDataset(train_df, train_transforms), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LeafMultiDomainDataset(val_df, val_transforms), batch_size=batch_size, shuffle=False, num_workers=0)

    test_field_loader = DataLoader(LeafMultiDomainDataset(test_field_df, val_transforms), batch_size=batch_size, shuffle=False, num_workers=0)
    test_lab_loader = DataLoader(LeafMultiDomainDataset(test_lab_df, val_transforms), batch_size=64, shuffle=False, num_workers=0)

    # Model Init
    model, backbone_modules, classifier_modules = build_model(arch_name, num_classes=38)
    model.to(device)

    best_checkpoint_path = Path(f"cv/models/{arch_name}_38classes_best.pth")
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_macro_f1 = 0.0
    history = []

    if resume_from and Path(resume_from).exists():
        print(f"[*] Loading checkpoint state: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
        warmup_epochs = 0
        print("[*] Skipping Phase 1 warmup since model is resumed from existing checkpoint.")
        print("[*] Evaluating baseline checkpoint performance on validation set...")
        init_val_res = evaluate_loader(model, val_loader, device, criterion)
        best_val_macro_f1 = init_val_res["macro_f1"]
        print(f"[*] Resumed Checkpoint Val Acc: {init_val_res['accuracy']:.2f}% | Macro F1: {best_val_macro_f1:.2f}% | Field Val Acc: {init_val_res['field_accuracy']:.2f}%")

    # ── PHASE 1: HEAD WARMUP (Backbone Frozen) ─────────────────────────────────
    if warmup_epochs > 0:
        print(f"\n[*] PHASE 1: Classifier Head Warmup ({warmup_epochs} epochs, backbone frozen)...")
        for m in backbone_modules:
            for p in m.parameters():
                p.requires_grad = False

        warmup_params = [p for m in classifier_modules for p in m.parameters() if p.requires_grad]
        optimizer_warmup = torch.optim.AdamW(warmup_params, lr=lr_head, weight_decay=1e-4)

        for ep in range(1, warmup_epochs + 1):
            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer_warmup.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, lbls)
                loss.backward()
                optimizer_warmup.step()

                running_loss += loss.item() * imgs.size(0)
                _, preds = outs.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)

            val_res = evaluate_loader(model, val_loader, device, criterion)
            dur = time.time() - t0
            tr_loss = running_loss / total
            tr_acc = (correct / total) * 100.0

            if val_res["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_res["macro_f1"]
                torch.save(model.state_dict(), best_checkpoint_path)
                saved_tag = " 🎯 [BEST]"
            else:
                saved_tag = ""

            print(f"Warmup [{ep:2d}/{warmup_epochs:2d}] ({dur:4.1f}s) | Train Acc: {tr_acc:5.1f}% | "
                  f"Val Acc: {val_res['accuracy']:5.1f}% | Val Macro-F1: {val_res['macro_f1']:5.1f}% | "
                  f"Field-Val Acc: {val_res['field_accuracy']:5.1f}%{saved_tag}")

    # ── PHASE 2: PROGRESSIVE UNFREEZING & DIFFERENTIAL FINE-TUNING ──────────────
    print(f"\n[*] PHASE 2: Differential Fine-Tuning ({fine_tune_epochs} epochs, full network unfreezing)...")
    for m in backbone_modules:
        for p in m.parameters():
            p.requires_grad = True

    backbone_params = [p for m in backbone_modules for p in m.parameters() if p.requires_grad]
    classifier_params = [p for m in classifier_modules for p in m.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": classifier_params, "lr": lr_head, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)

    for ep in range(1, fine_tune_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outs.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)

        scheduler.step()
        val_res = evaluate_loader(model, val_loader, device, criterion)
        dur = time.time() - t0
        tr_loss = running_loss / total
        tr_acc = (correct / total) * 100.0

        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            saved_tag = " 🎯 [BEST CHECKPOINT SAVED]"
        else:
            saved_tag = ""

        print(f"Epoch  [{ep:2d}/{fine_tune_epochs:2d}] ({dur:4.1f}s) | Train Acc: {tr_acc:5.1f}% (Loss: {tr_loss:.4f}) | "
              f"Val Acc: {val_res['accuracy']:5.1f}% | Val Macro-F1: {val_res['macro_f1']:5.1f}% | "
              f"Field-Val Acc: {val_res['field_accuracy']:5.1f}% | Field F1: {val_res['field_macro_f1']:5.1f}%{saved_tag}")

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "field_val_acc": val_res["field_accuracy"],
            "field_val_f1": val_res["field_macro_f1"]
        })

    # ── FINAL EVALUATION ON STRICTLY HELD-OUT TEST SETS ────────────────────────
    print("\n" + "=" * 80)
    print(f"📊 EVALUATING BEST CHECKPOINT: {best_checkpoint_path.resolve()}")
    print("=" * 80)

    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    model.eval()

    # 1. Held-out Field Test
    field_test_res = evaluate_loader(model, test_field_loader, device)
    print(f"[1] Real-World PlantDoc Field Held-Out Test Set (N={len(test_field_df)}):")
    print(f"    • Top-1 Accuracy       : {field_test_res['accuracy']:.2f}%")
    print(f"    • Top-3 Accuracy       : {field_test_res['top3_accuracy']:.2f}%")
    print(f"    • Balanced Accuracy    : {field_test_res['balanced_accuracy']:.2f}%")
    print(f"    • Macro F1 Score       : {field_test_res['macro_f1']:.2f}%")
    print(f"    • Macro Precision      : {field_test_res['macro_precision']:.2f}%")
    print(f"    • Macro Recall         : {field_test_res['macro_recall']:.2f}%")

    # 2. Held-out Lab Test
    lab_test_res = evaluate_loader(model, test_lab_loader, device)
    print(f"\n[2] PlantVillage Laboratory Held-Out Test Set (N={len(test_lab_df)}):")
    print(f"    • Top-1 Accuracy       : {lab_test_res['accuracy']:.2f}%")
    print(f"    • Top-3 Accuracy       : {lab_test_res['top3_accuracy']:.2f}%")
    print(f"    • Balanced Accuracy    : {lab_test_res['balanced_accuracy']:.2f}%")
    print(f"    • Macro F1 Score       : {lab_test_res['macro_f1']:.2f}%")

    # Export Experiment Summary
    result_payload = {
        "model": arch_name,
        "resolution": img_size,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "best_val_macro_f1": best_val_macro_f1,
        "field_test": {
            "top1_acc": field_test_res["accuracy"],
            "top3_acc": field_test_res["top3_accuracy"],
            "balanced_acc": field_test_res["balanced_accuracy"],
            "macro_f1": field_test_res["macro_f1"],
            "macro_prec": field_test_res["macro_precision"],
            "macro_rec": field_test_res["macro_recall"]
        },
        "lab_test": {
            "top1_acc": lab_test_res["accuracy"],
            "top3_acc": lab_test_res["top3_accuracy"],
            "balanced_acc": lab_test_res["balanced_accuracy"],
            "macro_f1": lab_test_res["macro_f1"]
        },
        "history": history
    }

    res_path = Path(f"cv/logs/experiment_{arch_name}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    print(f"\n[+] Saved Experiment Telemetry to: {res_path.resolve()}")
    print("=" * 80)
    return result_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="efficientnet_b2", help="Model architecture")
    parser.add_argument("--img_size", type=int, default=288, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=6, help="Fine-tuning epochs")
    parser.add_argument("--device", type=str, default="mps", help="Compute device")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--manifest", type=str, default="cv/datasets/processed/manifest_38classes_tridataset.csv", help="Path to manifest CSV")
    args = parser.parse_args()

    train_model(
        arch_name=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        device_str=args.device,
        resume_from=args.resume_from,
        manifest_path_str=args.manifest
    )
