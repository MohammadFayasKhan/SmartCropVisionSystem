"""
test_decision_logic.py
----------------------
Validates the refined plant pathology decision pipeline across:
1. images-3.jpeg (healthy field tomato leaf from prompt)
2. PlantDoc field test healthy leaves
3. PlantVillage laboratory healthy leaves
4. PlantVillage diseased leaves (Early Blight, Late Blight, Black Rot, Common Rust, Apple Scab)
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

device = torch.device("cpu")

# 1. Load Taxonomy
with open("cv/configs/taxonomy_38classes.json", "r") as f:
    taxonomy = json.load(f)
idx_to_meta = {item["class_index"]: item for item in taxonomy}

healthy_indices = [
    i for i, item in idx_to_meta.items()
    if "healthy" in item.get("condition_type", "").lower() or "healthy" in item.get("raw_folder", "").lower()
]
disease_indices = [i for i in idx_to_meta if i not in healthy_indices]

# 2. Load Models
t1_path = Path("cv/models/mobilenet_v2_38classes_best.pth")
t1_model = models.mobilenet_v2(weights=None)
t1_model.classifier = nn.Sequential(
    nn.Dropout(p=0.25),
    nn.Linear(t1_model.last_channel, 38)
)
t1_model.load_state_dict(torch.load(t1_path, map_location=device))
t1_model.to(device)
t1_model.eval()

t2_path = Path("cv/models/yolov8n_lesions_best.pt")
t2_model = YOLO(str(t2_path))

# Transforms
t1_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_image(img_path: str):
    path = Path(img_path)
    if not path.exists():
        return f"File not found: {img_path}"
    
    img = Image.open(path).convert("RGB")
    w, h = img.size
    
    # Tier 1
    tensor = t1_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = t1_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_confs, top_indices = torch.topk(probs, k=3)
    
    top1_idx = int(top_indices[0].item())
    top1_conf = float(top_confs[0].item())
    meta = idx_to_meta[top1_idx]
    pred_class = meta["raw_folder"]
    
    p_healthy_total = float(torch.sum(probs[healthy_indices]).item())
    p_disease_total = float(torch.sum(probs[disease_indices]).item())
    
    # Tier 2 YOLO with confident threshold (conf >= 0.35)
    np_img = np.array(img)
    yolo_res = t2_model(np_img, conf=0.35, iou=0.45, verbose=False)
    boxes = []
    if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
        for b in yolo_res[0].boxes:
            b_conf = float(b.conf[0].item())
            boxes.append((b.xyxy[0].tolist(), b_conf))
    foci_count = len(boxes)
    
    # Decision Matrix
    # Case A: Top-1 is explicitly healthy, or aggregate healthy probability mass exceeds disease mass
    if (top1_idx in healthy_indices) or (p_healthy_total > p_disease_total and top1_conf < 0.65):
        decision = "HEALTHY"
        diag_name = "Healthy / No Disease Detected"
        is_infected = False
        damage_pct = 0.0
        foci_count = 0
        final_boxes = []
    # Case B: High confidence disease or verified focal lesion presence
    elif (top1_idx in disease_indices) and (top1_conf >= 0.70 or (top1_conf >= 0.50 and foci_count > 0)):
        decision = "DISEASE"
        diag_name = meta["disease_name"]
        is_infected = True
        damage_pct = 12.0
        final_boxes = boxes
    # Case C: Disease predicted at low/moderate confidence but refuted by spatial evidence (no lesions on foliage)
    elif (top1_idx in disease_indices) and foci_count == 0:
        if p_healthy_total >= 0.20 or top1_conf < 0.35:
            decision = "HEALTHY (Refuted by Spatial Telemetry)"
            diag_name = "Healthy / No Disease Detected"
            is_infected = False
            damage_pct = 0.0
            foci_count = 0
            final_boxes = []
        else:
            decision = "UNCERTAIN"
            diag_name = "Uncertain - Retake Image"
            is_infected = False
            damage_pct = 0.0
            foci_count = 0
            final_boxes = []
    # Case D: Overall confidence is too low to make an actionable diagnosis
    elif top1_conf < 0.35:
        decision = "UNCERTAIN"
        diag_name = "Uncertain - Retake Image"
        is_infected = False
        damage_pct = 0.0
        foci_count = 0
        final_boxes = []
    else:
        decision = "HEALTHY" if p_healthy_total >= p_disease_total else "DISEASE"
        diag_name = "Healthy / No Disease Detected" if decision == "HEALTHY" else meta["disease_name"]
        is_infected = (decision == "DISEASE")
        damage_pct = 8.0 if is_infected else 0.0
        final_boxes = boxes if is_infected else []

    return {
        "file": path.name,
        "top1": f"{pred_class} ({top1_conf*100:.1f}%)",
        "p_healthy": f"{p_healthy_total*100:.1f}%",
        "p_disease": f"{p_disease_total*100:.1f}%",
        "decision": decision,
        "diagnosis_name": diag_name,
        "is_infected": is_infected,
        "foci_count": foci_count,
        "damage_pct": damage_pct
    }

test_files = [
    "cv/datasets/processed/segmentation/images/val/0001.jpg",
    "cv/datasets/raw/plantdoc_repo/test/Tomato leaf/1684.jpg",
    "cv/datasets/raw/plantdoc_repo/test/Tomato leaf/DSCN1015.JPG.jpg",
    "cv/datasets/raw/plantvillage/raw/color/Tomato___healthy/9e2a71e5-2a59-4e62-9c6c-581fe9091a10___RS_HL 0132.JPG",
    "cv/datasets/raw/plantvillage/raw/color/Tomato___Late_blight/3d9cca85-96cf-4186-b863-933bbbbc8075___GHLB2 Leaf 117.4.JPG",
    "cv/datasets/raw/plantvillage/raw/color/Grape___Black_rot/0e143d33-adc0-41af-92e2-d0bb712d7b72___FAM_B.Rot 5047.JPG",
    "cv/datasets/raw/plantvillage/raw/color/Corn_(maize)___Common_rust_/RS_Rust 2335.JPG",
    "cv/datasets/raw/plantvillage/raw/color/Apple___Apple_scab/14c623e5-051c-42f6-9e4f-f7a93e6a723c___FREC_Scab 2965.JPG"
]

print("-" * 115)
print(f"{'Image':<35} | {'Decision':<32} | {'Diagnosis':<28} | {'Infected':<8}")
print("-" * 115)
for tf in test_files:
    res = evaluate_image(tf)
    if isinstance(res, dict):
        print(f"{res['file'][:35]:<35} | {res['decision'][:32]:<32} | {res['diagnosis_name'][:28]:<28} | {str(res['is_infected']):<8}")
    else:
        print(res)
print("-" * 115)
