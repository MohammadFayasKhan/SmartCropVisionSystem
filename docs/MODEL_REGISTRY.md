# SmartCropVision Production Model Registry & Provenance Guide

## 1. Authoritative Model Registry Overview

The SmartCropVision platform maintains an authoritative model registry loaded at application startup by `backend/app/services/model_registry.py`. All neural network weight files reside exclusively on the server filesystem in `cv/models/` and are validated against cryptographic SHA 256 signatures declared in `RELEASE_MANIFEST.json`.

```
cv/models/
├── efficientnetv2_s_best.pt              # Universal Classifier (Server Grade)
├── efficientnetv2_s_tri_domain_best.pt   # Gen 2 Tri Domain Universal Classifier
├── yolo_plantdoc_best.pt                 # Specimen Foliage Canopy Detector
├── yolov8n_lesions_best.pt               # Granular Necrotic Lesion Spot Detector
├── mobile_unet_best.pt                   # Sub Pixel Foliar Damage Segmenter
└── crop_recommender/
    ├── crop_model.pkl                    # Agro Climatic Random Forest Classifier
    ├── scaler.pkl                        # Feature Standard Scaler
    ├── label_encoder.pkl                 # Categorical Label Encoder
    └── feature_names.pkl                 # Tabular Input Column Index
```

## 2. Model Checkpoint Ledger & Verification Signatures

| Model Role | Filepath | Size | Architecture | Supported Task | SHA 256 Signature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Primary Classifier | `cv/models/efficientnetv2_s_best.pt` | 78.03 MB | EfficientNetV2 S | Universal Pathology Classification | `07e84d39481f0757898c5088cc7fb9e9d28817fbd71e16769fea5eb056eace74` |
| Tri Domain Classifier | `cv/models/efficientnetv2_s_tri_domain_best.pt` | 78.02 MB | EfficientNetV2 S | Multi Domain Pathology Classification | `7ebe13a085819231878c776092c112b2c0babf678d39e4d0f2a2e1ee258f25c7` |
| Specimen Detector | `cv/models/yolo_plantdoc_best.pt` | 5.19 MB | YOLO DetectionModel | Whole Leaf Canopy Bounding | `e98545716c774d796a4796f620138022f6388a063bf8a9d36dfc5937372366b6` |
| Lesion Spot Detector | `cv/models/yolov8n_lesions_best.pt` | 5.91 MB | YOLO DetectionModel | Granular Spot Level Lesion Localization | `f528dc7cf1ccafa85e63fb4f75ea3097b9197da7c89c4380b93b04fe708aacf7` |
| Foliar Segmenter | `cv/models/mobile_unet_best.pt` | 1.90 MB | Mobile UNet | Sub Pixel Foliar Damage Segmentation | `86bba6df9222b92c2d8b019fb2a10eacc2654f1a61b6e2b5996d8c8ee3cc04cc` |
| Tabular Recommender | `cv/models/crop_recommender/crop_model.pkl` | 19.45 MB | RandomForestClassifier | Agro Climatic Crop Recommendation | `cb20fc5908efe34d4e5e6e999b62e576d962ff2e1d00381db09c912bb57aa1d3` |

## 3. Individual Model Specifications

### 3.1 Primary Vision Classifier: EfficientNetV2 S
* **Architecture**: EfficientNetV2 S with fused inverted bottleneck blocks and progressive regularization.
* **Checkpoint**: `cv/models/efficientnetv2_s_best.pt`
* **Input Specifications**: Bilinear resized 256x256 Float32 tensor normalized via standard ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
* **Taxonomy Dependency**: Canonical 38 class mapping defined in `cv/models/taxonomy_38.json`.
* **Verified Benchmark Evidence**:
  * Test Top 1 Accuracy: 95.13%
  * Test Macro F1 Score: 0.9354
  * Expected Calibration Error: 0.0803
* **Operational Role**: Primary diagnostic engine that evaluates the entire photographed leaf blade to produce calibrated probability distributions and botanical Grad CAM saliency activations.
* **Limitations**: Operates on single crop leaf specimens. In complex outdoor canopies containing overlapping multiple crop species, Tier 2 spatial bounding should precede classification.

### 3.2 Specimen Foliage Canopy Detector: YOLO PlantDoc
* **Architecture**: Ultralytics YOLO fine tuned on the PlantDoc agricultural dataset.
* **Checkpoint**: `cv/models/yolo_plantdoc_best.pt`
* **Input Specifications**: Letterbox padded 640x640 RGB tensor preserving aspect ratios.
* **Taxonomy Dependency**: 29 spatial foliar classes defined in `cv/configs/yolo_plantdoc.yaml`.
* **Verified Benchmark Evidence**:
  * Validation mAP at 50: 0.3362
  * Validation mAP at 50 to 95: 0.2361
* **Operational Role**: Identifies and bounds individual leaves and foliar canopy units in outdoor cluttered environments.
* **Truthful Semantic Boundary**: PlantDoc ground truth annotations enclose whole leaves and diseased foliage units rather than micro spots. These detections are rendered as the emerald specimen layer and never presented as spot level lesion foci.

### 3.3 Granular Lesion Spot Detector: YOLOv8n Lesions
* **Architecture**: Ultralytics YOLO fine tuned on the micro spot lesion dataset.
* **Checkpoint**: `cv/models/yolov8n_lesions_best.pt`
* **Input Specifications**: Letterbox padded 640x640 RGB tensor.
* **Taxonomy Dependency**: Target class `early_blight_lesion` defined in `cv/configs/yolo_lesions.yaml`.
* **Dataset Evidence**: Trained on 330 images containing 1,746 micro spot annotations with median bounding box area ratio of 0.0051 (0.5% of total image area).
* **Operational Role**: Localizes individual necrotic lesion spots on affected foliage. Rendered as the coral pathology layer in the dual layer frontend canvas.

### 3.4 Sub Pixel Foliar Damage Segmenter: Mobile UNet
* **Architecture**: Lightweight MobileNetV2 inverted bottleneck encoder coupled with a bilinear upsampling decoder and skip connections.
* **Checkpoint**: `cv/models/mobile_unet_best.pt`
* **Input Specifications**: 256x256 RGB tensor.
* **Output Semantics**: 3 class discrete pixel mask where index 0 represents background, index 1 represents healthy foliar tissue, and index 2 represents necrotic lesions.
* **Verified Benchmark Evidence**:
  * Mean Intersection over Union: 0.7485
  * Necrotic Lesion Dice Coefficient: 0.7012
* **Operational Role**: Generates pixel masks for calculating the Foliar Damage Percentage.
* **Spectral Cleaning Safeguard**: Output masks undergo non plant background rejection in HSV and Excess Green color spaces to prevent shadows, wood tables, human hands, and bare soil from being mislabeled as necrotic tissue.

### 3.5 Agro Climatic Tabular Recommender: Random Forest
* **Architecture**: Scikit Learn Random Forest Classifier (100 estimators).
* **Checkpoint**: `cv/models/crop_recommender/crop_model.pkl`
* **Input Features**: 7 continuous and binary environmental parameters (Nitrogen, Phosphorus, Potassium, Temperature, Relative Humidity, Soil pH, and Rainfall).
* **Taxonomy Dependency**: 22 agricultural crop candidate species.
* **Operational Role**: Recommends optimal crop selection based on microclimate telemetry gathered from IoT soil and weather sensors.
