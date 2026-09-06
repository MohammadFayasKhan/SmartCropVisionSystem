"""
vision_engine.py
----------------
Dedicated Computer Vision Inference Engine for Smart Plant Intelligence System.
Executes the authentic 3-Tier cascade:
Tier 1: MobileNetV2 38-class universal crop disease classifier with healthy leaf gating.
Tier 2: YOLOv8-nano spatial necrotic lesion localization with multi-foci detection.
Tier 3: Mobile-UNet sub-pixel pathology segmentation and Botanical Damage Index.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import base64
import io
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ── Paths resolution ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Check local application directories first, then workspace fallbacks
if (BASE_DIR / "models").exists():
    CV_MODELS_DIR = BASE_DIR / "models"
elif (BASE_DIR / "cv" / "models").exists():
    CV_MODELS_DIR = BASE_DIR / "cv" / "models"
else:
    CV_MODELS_DIR = BASE_DIR.parent / "cv" / "models"

if (BASE_DIR / "configs").exists():
    CV_CONFIGS_DIR = BASE_DIR / "configs"
elif (BASE_DIR / "cv" / "configs").exists():
    CV_CONFIGS_DIR = BASE_DIR / "cv" / "configs"
else:
    CV_CONFIGS_DIR = BASE_DIR.parent / "cv" / "configs"

TAXONOMY_PATH = CV_CONFIGS_DIR / "taxonomy_38classes.json"
TIER1_MODEL_PATH = CV_MODELS_DIR / "mobilenet_v2_38classes_best.pth"
TIER2_MODEL_PATH = CV_MODELS_DIR / "yolov8n_lesions_best.pt"
TIER3_MODEL_PATH = CV_MODELS_DIR / "mobile_unet_lesions_best.pth"
PLANTDOC_MODEL_PATH = CV_MODELS_DIR / "yolov8n_plantdoc_best.pt"

# ── Advisory Knowledge Base ──────────────────────────────────────────────────
ADVISORY_DB = {
    "Tomato___Early_blight": {
        "immediate": "Isolate affected tomato vines and prune lower diseased foliage immediately.",
        "treatment": "Apply Chlorothalonil 720 SC (2.0 mL/L water) or Mancozeb 75% WP (2.5 g/L). Alternate with Azoxystrobin every 7 to 10 days.",
        "cultural": "Avoid overhead sprinkler irrigation to keep canopy dry. Space plants 60 cm apart for adequate airflow. Remove infected debris."
    },
    "Tomato___Late_blight": {
        "immediate": "Urgent fungal threat: rogue out severely infected plants and bag immediately to stop spore transmission.",
        "treatment": "Apply Cymoxanil + Mancozeb (Curzate M8, 2.5 g/L) or Dimethomorph (1.5 mL/L). Spray preventatively before rain events.",
        "cultural": "Destroy volunteer solanaceous weeds (nightshade). Water strictly at base through drip tubes."
    },
    "Tomato___healthy": {
        "immediate": "No pathological symptoms detected. Foliage displays robust photosynthetic vigor.",
        "treatment": "No chemical fungicide application warranted. Apply balanced organic foliar nutrients.",
        "cultural": "Maintain regular drip scheduling, root aeration, and weed sanitation."
    },
    "Corn_(maize)___Common_rust_": {
        "immediate": "Inspect field boundary rows for powdery reddish-brown uredinial pustules.",
        "treatment": "Apply Pyraclostrobin + Fluxapyroxad (Priaxor, 0.6 mL/L) or Tebuconazole 250 EC (1.0 mL/L).",
        "cultural": "Select resistant corn hybrids for next planting cycle. Ensure soil potassium levels are balanced."
    },
    "Grape___Black_rot": {
        "immediate": "Remove mummified berry clusters and black-spotted leaves from trellis canopy.",
        "treatment": "Apply Myclobutanil (Nova 40W, 1.2 g/L) or Captan 50 WP (2.5 g/L) starting at early bud break.",
        "cultural": "Prune grape canopy to optimize solar penetration and accelerate morning foliage drying."
    },
    "Apple___Apple_scab": {
        "immediate": "Rake and compost or burn fallen leaf litter to eliminate overwintering fungal ascospores.",
        "treatment": "Apply Captan 80 WDG (1.8 g/L) or Difenoconazole (Score 250 EC, 0.3 mL/L) at green tip stage.",
        "cultural": "Prune internal water sprouts to open tree canopy to sunlight and wind drying."
    }
}

DEFAULT_INFECTED_ADVISORY = {
    "immediate": "Isolate symptomatic plants and inspect adjacent vegetation for early lesion propagation.",
    "treatment": "Apply broad-spectrum protective bio-fungicide (Copper Oxychloride 50 WP at 3.0 g/L or cold-pressed Neem oil at 5 mL/L).",
    "cultural": "Reduce canopy humidity by improving spacing and pruning. Switch completely to ground-level drip irrigation."
}

DEFAULT_HEALTHY_ADVISORY = {
    "immediate": "No intervention required. Foliar tissue exhibits healthy photosynthetic green pigmentation.",
    "treatment": "Maintain preventative organic biostimulants. No fungicide application needed.",
    "cultural": "Continue standard soil moisture regulation, balanced fertilization, and IPM scouting."
}

# ── Mobile-UNet Architecture ────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """(Conv2D → BatchNorm → ReLU) x 2 Block"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class MobileUNet(nn.Module):
    """Lightweight 4-stage U-Net with skip connections for lesion segmentation"""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.inc = DoubleConv(3, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConv(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConv(64 + 32, 32)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = DoubleConv(32 + 16, 16)

        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = torch.cat([self.up1(x4), x3], dim=1)
        d1 = self.conv_up1(u1)

        u2 = torch.cat([self.up2(d1), x2], dim=1)
        d2 = self.conv_up2(u2)

        u3 = torch.cat([self.up3(d2), x1], dim=1)
        d3 = self.conv_up3(u3)

        return self.outc(d3)

# ── Vision Engine Singleton ──────────────────────────────────────────────────
class VisionInferenceEngine:
    _instance: Optional["VisionInferenceEngine"] = None

    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device_name = (
            "Apple Silicon GPU (MPS)" if self.device.type == "mps"
            else ("NVIDIA GPU (CUDA)" if self.device.type == "cuda" else "Host CPU")
        )
        self.is_loaded = False
        self.taxonomy: Dict[str, Any] = {}
        self.model_tier1: Optional[nn.Module] = None
        self.model_tier2: Optional[YOLO] = None
        self.model_tier3: Optional[nn.Module] = None

        self.t1_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.t3_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def get_instance(cls) -> "VisionInferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        """Loads and pre-caches the 3 vision tiers onto target compute hardware."""
        if self.is_loaded:
            return

        print(f"[*] Initializing Computer Vision Suite on: {self.device_name}...")

        # 1. Taxonomy
        if not TAXONOMY_PATH.exists():
            raise FileNotFoundError(f"Taxonomy config not found at: {TAXONOMY_PATH}")
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        self.idx_to_meta = {item["class_index"]: item for item in self.taxonomy}

        # 2. Tier 1: MobileNetV2
        if not TIER1_MODEL_PATH.exists():
            raise FileNotFoundError(f"Tier 1 checkpoint missing at: {TIER1_MODEL_PATH}")
        t1 = models.mobilenet_v2(weights=None)
        t1.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(t1.last_channel, 38)
        )
        state_dict_t1 = torch.load(TIER1_MODEL_PATH, map_location=self.device)
        t1.load_state_dict(state_dict_t1)
        t1.to(self.device)
        t1.eval()
        self.model_tier1 = t1

        # 3. Tier 2: YOLOv8-nano
        if not TIER2_MODEL_PATH.exists():
            raise FileNotFoundError(f"Tier 2 checkpoint missing at: {TIER2_MODEL_PATH}")
        self.model_tier2 = YOLO(str(TIER2_MODEL_PATH))

        # 4. Tier 3: Mobile-UNet
        if not TIER3_MODEL_PATH.exists():
            raise FileNotFoundError(f"Tier 3 checkpoint missing at: {TIER3_MODEL_PATH}")
        t3 = MobileUNet(num_classes=3)
        state_dict_t3 = torch.load(TIER3_MODEL_PATH, map_location=self.device)
        t3.load_state_dict(state_dict_t3)
        t3.to(self.device)
        t3.eval()
        self.model_tier3 = t3

        self.is_loaded = True
        print(f"[*] Computer Vision Suite ready on {self.device_name}. Total taxonomy classes: {len(self.taxonomy)}")

    def validate_image_bytes(self, image_bytes: bytes) -> Image.Image:
        """Validates payload bounds and decodes bytes into PIL RGB image."""
        if len(image_bytes) < 2048:
            raise ValueError("Uploaded file is too small (minimum 2 KB required for foliar analysis).")
        if len(image_bytes) > 15 * 1024 * 1024:
            raise ValueError("Uploaded image exceeds 15 MB limit.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
        except Exception as e:
            raise ValueError(f"Corrupt or invalid image file: {e}")

        # Reopen after verify
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        if w < 100 or h < 100:
            raise ValueError(f"Image resolution {w}x{h} is too low for lesion analysis (minimum 100x100).")
        return image

    @staticmethod
    def _merge_spatial_boxes(candidate_boxes: List[Dict[str, Any]], img_w: int, img_h: int, max_foci: int = 8) -> List[Dict[str, Any]]:
        """
        Consolidates candidate bounding boxes from heterogeneous sources:
        1. YOLOv8 object detector proposals (spatial lesion anchors)
        2. Morphological connected components from Mobile-UNet segmentation masks

        Both Intersection over Union (IoU > 0.20) and containment (containment > 0.40)
        are evaluated. Containment is vital for agricultural pathology because necrotic
        lesions often develop satellite spots or concentric rings where a larger lesion
        encompasses smaller sub-foci. A single cohesive focus envelope is retained.
        """
        if not candidate_boxes:
            return []

        def box_area(b):
            coords = b["bbox_xyxy"]
            return max(1, (coords[2] - coords[0]) * (coords[3] - coords[1]))

        # Sort by area descending
        sorted_boxes = sorted(candidate_boxes, key=box_area, reverse=True)
        merged: List[Dict[str, Any]] = []

        for cand in sorted_boxes:
            c = cand["bbox_xyxy"]
            merged_into = False

            for m in merged:
                mc = m["bbox_xyxy"]
                ix1 = max(c[0], mc[0])
                iy1 = max(c[1], mc[1])
                ix2 = min(c[2], mc[2])
                iy2 = min(c[3], mc[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ac = box_area(cand)
                am = box_area(m)
                union = ac + am - inter
                iou = inter / union if union > 0 else 0.0
                containment = inter / min(ac, am) if min(ac, am) > 0 else 0.0

                # Merge if significant IoU or one box is inside another
                if iou > 0.20 or containment > 0.40:
                    m["bbox_xyxy"] = [
                        max(0, min(c[0], mc[0])),
                        max(0, min(c[1], mc[1])),
                        min(img_w, max(c[2], mc[2])),
                        min(img_h, max(c[3], mc[3]))
                    ]
                    m["confidence"] = max(m.get("confidence", 0.85), cand.get("confidence", 0.85))
                    merged_into = True
                    break

            if not merged_into:
                merged.append(dict(cand))

        # Second consolidation pass
        consolidated: List[Dict[str, Any]] = []
        for b in merged:
            c = b["bbox_xyxy"]
            merged_into = False
            for cb in consolidated:
                mc = cb["bbox_xyxy"]
                ix1 = max(c[0], mc[0])
                iy1 = max(c[1], mc[1])
                ix2 = min(c[2], mc[2])
                iy2 = min(c[3], mc[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ac = box_area(b)
                am = box_area(cb)
                union = ac + am - inter
                iou = inter / union if union > 0 else 0.0
                containment = inter / min(ac, am) if min(ac, am) > 0 else 0.0

                if iou > 0.25 or containment > 0.45:
                    cb["bbox_xyxy"] = [
                        max(0, min(c[0], mc[0])),
                        max(0, min(c[1], mc[1])),
                        min(img_w, max(c[2], mc[2])),
                        min(img_h, max(c[3], mc[3]))
                    ]
                    cb["confidence"] = max(cb.get("confidence", 0.85), b.get("confidence", 0.85))
                    merged_into = True
                    break
            if not merged_into:
                consolidated.append(b)

        # Filter out tiny slivers (< 16x16 pixels)
        filtered = [b for b in consolidated if (b["bbox_xyxy"][2] - b["bbox_xyxy"][0]) >= 16 and (b["bbox_xyxy"][3] - b["bbox_xyxy"][1]) >= 16]
        if not filtered and consolidated:
            filtered = consolidated

        # Cap to top max_foci (e.g. 8)
        top_foci = sorted(filtered, key=box_area, reverse=True)[:max_foci]

        # Format output
        results = []
        for idx, item in enumerate(top_foci):
            coords = item["bbox_xyxy"]
            cx = round(float((coords[0] + coords[2]) / (2.0 * img_w)), 3)
            cy = round(float((coords[1] + coords[3]) / (2.0 * img_h)), 3)
            results.append({
                "label": f"Infection Focus #{idx + 1}",
                "bbox_xyxy": coords,
                "confidence": round(float(item.get("confidence", 0.85)), 2),
                "centroid_norm": [cx, cy]
            })

        return results

    def diagnose(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs the complete 3-Tier diagnostic inference cascade."""
        if not self.is_loaded:
            self.load_models()

        t_total_start = time.time()
        image = self.validate_image_bytes(image_bytes)
        img_w, img_h = image.size

        # ── TIER 1: Screening & Gating ──────────────────────────────────────
        t1_start = time.time()
        tensor_t1 = self.t1_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_t1 = self.model_tier1(tensor_t1)
            probas_t1 = torch.softmax(logits_t1, dim=1)[0]
            top_confs, top_indices = torch.topk(probas_t1, k=3)

        t1_ms = round((time.time() - t1_start) * 1000, 2)

        # Precompute healthy and diseased index sets across 38 classes
        healthy_indices = [
            i for i, item in self.idx_to_meta.items()
            if "healthy" in item.get("condition_type", "").lower() or "healthy" in item.get("raw_folder", "").lower()
        ]
        disease_indices = [i for i in self.idx_to_meta if i not in healthy_indices]

        p_healthy_total = float(torch.sum(probas_t1[healthy_indices]).item())
        p_disease_total = float(torch.sum(probas_t1[disease_indices]).item())

        top1_idx = int(top_indices[0].item())
        top1_conf = float(top_confs[0].item())
        top1_conf_pct = round(top1_conf * 100, 2)

        meta = self.idx_to_meta.get(top1_idx, {
            "crop": "Crop",
            "condition_type": "unknown",
            "disease_name": f"Class {top1_idx}",
            "raw_folder": f"Class_{top1_idx}"
        })
        detected_crop = meta.get("crop", "Crop")

        # Find crop-specific healthy class metadata
        crop_healthy_meta = next(
            (item for item in self.taxonomy
             if item.get("crop", "").lower() == detected_crop.lower()
             and ("healthy" in item.get("condition_type", "").lower() or "healthy" in item.get("raw_folder", "").lower())),
            None
        )

        top3 = []
        for i in range(3):
            idx_i = int(top_indices[i].item())
            meta_i = self.idx_to_meta.get(idx_i, {})
            cls_i = meta_i.get("raw_folder", f"Class_{idx_i}")
            conf_i = round(float(top_confs[i].item()) * 100, 2)
            lbl_i = meta_i.get("disease_name", cls_i.replace("_", " "))
            top3.append({
                "class_id": cls_i,
                "label": lbl_i,
                "confidence_pct": conf_i
            })

        # ── TIER 2: YOLOv8-nano Spatial Lesion Localization ─────────────────
        t2_start = time.time()
        np_img = np.array(image)
        candidate_boxes: List[Dict[str, Any]] = []

        try:
            # Execute with robust confidence threshold (conf >= 0.35) to eliminate background shadows
            results = self.model_tier2(np_img, conf=0.35, iou=0.45, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    box_conf = round(float(b.conf[0].item()), 2)
                    candidate_boxes.append({
                        "bbox_xyxy": coords,
                        "confidence": box_conf,
                        "source": "yolo"
                    })
        except Exception as e:
            print(f"[!] Tier 2 YOLO execution note: {e}")

        t2_ms = round((time.time() - t2_start) * 1000, 2)

        # ── TIER 3: Mobile-UNet Sub-Pixel Segmentation ──────────────────────
        t3_start = time.time()
        tensor_t3 = self.t3_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_t3 = self.model_tier3(tensor_t3)
            mask_t3 = torch.argmax(out_t3, dim=1)[0].cpu().numpy().astype(np.uint8)

        mask_orig = cv2.resize(mask_t3, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        leaf_pixels = int(np.sum(mask_orig >= 1))
        lesion_pixels = int(np.sum(mask_orig == 2))

        damage_pct = round(float((lesion_pixels / max(leaf_pixels, 1)) * 100), 2)
        t3_ms = round((time.time() - t3_start) * 1000, 2)

        # ── Multi-Scale Morphological Lesion Clustering ─────────────────────
        # Only extract segmentation clusters if lesion pixels are physically substantial
        if np.any(mask_orig == 2) and damage_pct >= 4.0:
            lesion_bin = (mask_orig == 2).astype(np.uint8) * 255
            k_size = max(7, int(min(img_w, img_h) * 0.045))
            if k_size % 2 == 0:
                k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            dilated = cv2.dilate(lesion_bin, kernel)

            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(dilated)
            min_area = max(80, int(img_w * img_h * 0.002))

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    bx = int(stats[i, cv2.CC_STAT_LEFT])
                    by = int(stats[i, cv2.CC_STAT_TOP])
                    bw = int(stats[i, cv2.CC_STAT_WIDTH])
                    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    if bw >= 16 and bh >= 16:
                        candidate_boxes.append({
                            "bbox_xyxy": [bx, by, bx + bw, by + bh],
                            "confidence": 0.88,
                            "source": "segmenter"
                        })

        # ── Spatial Box Merging & NMS ───────────────────────────────────────
        boxes_out = self._merge_spatial_boxes(candidate_boxes, img_w, img_h, max_foci=8)
        foci_count = len(boxes_out)

        # ── MULTI-FACTOR DECISION MATRIX ────────────────────────────────────
        # Field specimens frequently exhibit natural lighting shifts, soil background,
        # and leaf venation that can skew laboratory-trained softmax probabilities.
        # To prevent false disease alerts on healthy crops, we cross-reference the
        # classification distribution with spatial lesion counts and segmented damage area.
        #
        # Case A: Explicit Healthy (Top-1 is healthy or aggregate healthy probability mass exceeds highest disease confidence)
        if (top1_idx in healthy_indices) or (p_healthy_total > top1_conf and top1_conf < 0.65):
            decision = "HEALTHY"
        # Case B: Confirmed Disease (High statistical confidence or verified focal lesion presence)
        elif (top1_idx in disease_indices) and (top1_conf >= 0.68 or (top1_conf >= 0.45 and foci_count > 0 and damage_pct >= 4.0)):
            decision = "DISEASE"
        # Case C: Spatial Refutation Guard (Disease predicted at moderate/low confidence but refuted by zero lesions on foliage)
        elif (top1_idx in disease_indices) and foci_count == 0:
            if p_healthy_total >= 0.15 or top1_conf < 0.35:
                decision = "HEALTHY"
            else:
                decision = "UNCERTAIN"
        # Case D: Overall confidence is too low to make a responsible agronomic determination
        elif top1_conf < 0.40:
            decision = "UNCERTAIN"
        else:
            decision = "HEALTHY" if p_healthy_total >= p_disease_total else "DISEASE"

        total_ms = round((time.time() - t_total_start) * 1000, 2)

        # ── SYNCHRONIZE DOWNSTREAM SECTIONS CONSISTENT WITH DECISION ─────────
        if decision == "HEALTHY":
            healthy_class_name = crop_healthy_meta.get("raw_folder") if crop_healthy_meta else "Tomato___healthy"
            advisory = ADVISORY_DB.get(healthy_class_name, DEFAULT_HEALTHY_ADVISORY)
            return {
                "status": "success",
                "diagnosis": {
                    "predicted_class": healthy_class_name,
                    "disease_common_name": "Healthy / No Disease Detected",
                    "crop": detected_crop,
                    "condition_type": "healthy",
                    "confidence_pct": round(max(top1_conf, p_healthy_total) * 100, 2),
                    "confidence_level": "CONFIRMED" if p_healthy_total >= 0.50 else "MODERATE",
                    "top3_predictions": top3,
                    "is_infected": False,
                    "triage_stage": "STAGE 0: Optimal Health",
                    "foliar_damage_pct": 0.0,
                    "lesion_foci_count": 0
                },
                "spatial_telemetry": {
                    "detection_engine": "Fused MobileNetV2 + Spatial Verification (Healthy Verified)",
                    "bounding_boxes": [],
                    "nozzle_actuation_targets": 0,
                    "variable_rate_dosage_multiplier": 1.0
                },
                "segmentation_mask_b64": None,
                "advisory": advisory,
                "latency_ms": {
                    "tier1_ms": t1_ms,
                    "tier2_ms": t2_ms,
                    "tier3_ms": t3_ms,
                    "total_ms": total_ms,
                    "device": self.device_name
                }
            }

        elif decision == "UNCERTAIN":
            uncertain_advisory = {
                "immediate": "Diagnostic certainty below required decision threshold. Retake a well-lit close-up photograph of the leaf.",
                "treatment": "No chemical fungicide application warranted while diagnosis is unconfirmed. Re-evaluate with a clearer photograph.",
                "cultural": "Ensure uniform illumination without harsh shadows or background clutter. Center the leaf lamina in frame."
            }
            return {
                "status": "success",
                "diagnosis": {
                    "predicted_class": "Uncertain",
                    "disease_common_name": "Uncertain - Retake Image",
                    "crop": detected_crop,
                    "condition_type": "uncertain",
                    "confidence_pct": top1_conf_pct,
                    "confidence_level": "LOW_UNCERTAIN",
                    "top3_predictions": top3,
                    "is_infected": False,
                    "triage_stage": "STAGE 0: Retake Specimen Image",
                    "foliar_damage_pct": 0.0,
                    "lesion_foci_count": 0
                },
                "spatial_telemetry": {
                    "detection_engine": "Multi-Tier Uncertainty Resolver (Specimen Image Ambiguous)",
                    "bounding_boxes": [],
                    "nozzle_actuation_targets": 0,
                    "variable_rate_dosage_multiplier": 1.0
                },
                "segmentation_mask_b64": None,
                "advisory": uncertain_advisory,
                "latency_ms": {
                    "tier1_ms": t1_ms,
                    "tier2_ms": t2_ms,
                    "tier3_ms": t3_ms,
                    "total_ms": total_ms,
                    "device": self.device_name
                }
            }

        # Confirmed Disease Branch
        pred_class = meta.get("raw_folder", f"Class_{top1_idx}")
        conf_level = "HIGH" if top1_conf_pct >= 80.0 else "MEDIUM"

        # Translucent Segmentation Mask Overlay in Base64
        overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        overlay[mask_orig == 2] = [239, 35, 60, 160]  # Translucent crimson #ef233c
        _, png_buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
        mask_b64 = f"data:image/png;base64,{base64.b64encode(png_buffer).decode('utf-8')}"

        if damage_pct >= 25.0:
            triage_stage = "STAGE 3: Severe Systemic Infection"
            multiplier = 2.0
        elif damage_pct >= 8.0:
            triage_stage = "STAGE 2: Moderate Progressive Lesions"
            multiplier = 1.5
        elif damage_pct >= 1.0:
            triage_stage = "STAGE 1: Early Localized Onset"
            multiplier = 1.2
        else:
            triage_stage = "STAGE 0: Incipient Infection / Trace Damage"
            multiplier = 1.0

        advisory = ADVISORY_DB.get(pred_class, DEFAULT_INFECTED_ADVISORY)

        return {
            "status": "success",
            "diagnosis": {
                "predicted_class": pred_class,
                "disease_common_name": meta.get("disease_name", pred_class.replace("_", " ")),
                "crop": meta.get("crop", "Solanaceous Crop"),
                "condition_type": meta.get("condition_type", "pathology"),
                "confidence_pct": top1_conf_pct,
                "confidence_level": conf_level,
                "top3_predictions": top3,
                "is_infected": True,
                "triage_stage": triage_stage,
                "foliar_damage_pct": damage_pct,
                "lesion_foci_count": foci_count
            },
            "spatial_telemetry": {
                "detection_engine": "Fused YOLOv8-nano + Mobile-UNet",
                "bounding_boxes": boxes_out,
                "nozzle_actuation_targets": foci_count,
                "variable_rate_dosage_multiplier": multiplier
            },
            "segmentation_mask_b64": mask_b64,
            "advisory": advisory,
            "latency_ms": {
                "tier1_ms": t1_ms,
                "tier2_ms": t2_ms,
                "tier3_ms": t3_ms,
                "total_ms": total_ms,
                "device": self.device_name
            }
        }

vision_engine = VisionInferenceEngine.get_instance()
