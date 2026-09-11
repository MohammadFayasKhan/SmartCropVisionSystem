# SmartCropVision Final Production Release Report

Release Identifier: SmartCropVision Production Release
Authoritative Version: 2.2.0
Release Tag: v2.2.0
Release Date: 2026 September 11
Engineering Team: Innovex
Evaluation Context: Smart India Hackathon Production Gate

## Executive Summary

SmartCropVision version 2.2.0 is an end to end agricultural computer vision and decision intelligence platform. It provides automated foliar disease diagnosis, spatial canopy localization, granular necrotic lesion detection, subpixel foliage segmentation, 9 stage Grad CAM visual explainability, agro climatic crop recommendation, and authenticated IoT rover telemetry.

This release candidate is delivered as a verified production package. All production model weights are frozen, read only, cryptographically validated, and hosted strictly server side. Browser clients, AgriRover IoT microcontrollers, and XiaoZhi voice nodes never download or execute local weight checkpoints. All reported metrics reflect verified historical evaluation benchmarks rather than live synthetic scores.

## Release Architecture

The production architecture separates responsibilities into distinct decoupled tiers:

```
[ AgriRover ESP32 / Field Camera / User Browser / XiaoZhi Node ]
                              │
                    HTTPS / WSS Request
                              ▼
           [ FastAPI Production Gateway (Uvicorn) ]
                              │
     ┌────────────────────────┴────────────────────────┐
     ▼                                                 ▼
[ Model Registry (Immutable) ]             [ Advisory & Decision Engine ]
  * EfficientNetV2 S (38 classes)            * Tabular Crop Recommender RF
  * EfficientNetV2 S Tri Domain (38 classes) * Agronomic Advisory Engine
  * YOLO PlantDoc Canopy Detector (29 classes)* Image Quality & OOD Guard
  * YOLOv8n Granular Lesion Detector (1 class)
  * Mobile UNet Foliar Segmenter (3 classes)
  * 9 Stage Explainability Pipeline (Grad CAM)
```

Data flows strictly in one direction during inference:
Client image upload → Input validation & Decompression bomb guard → Image quality & Out of distribution screening → Server primary classification → Optional spatial detection → Optional foliar segmentation → Optional 9 stage Grad CAM activation map → Agronomic treatment advisory → Structured response.

## Authoritative Model Artifact Ledger

Every production checkpoint on disk has been cryptographically validated using complete SHA 256 hashes. Checkpoints are stored under `cv/models/` and are strictly immutable during runtime inference.

| Model Role | Architecture | Checkpoint File | Size | Parameters / Classes | SHA 256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Server Primary Classifier | EfficientNetV2 S | `cv/models/efficientnetv2_s_best.pt` | 78.03 MB | 38 Classes | `07e84d39481f0757898c5088cc7fb9e9d28817fbd71e16769fea5eb056eace74` | Verified Immutable |
| Gen2 Multi Domain Classifier | EfficientNetV2 S Tri Domain | `cv/models/efficientnetv2_s_tri_domain_best.pt` | 78.02 MB | 38 Classes | `7ebe13a085819231878c776092c112b2c0babf678d39e4d0f2a2e1ee258f25c7` | Verified Immutable |
| Specimen Canopy Detector | YOLO PlantDoc | `cv/models/yolo_plantdoc_best.pt` | 5.19 MB | 29 Classes | `e98545716c774d796a4796f620138022f6388a063bf8a9d36dfc5937372366b6` | Verified Immutable |
| Granular Lesion Spot Detector | YOLOv8n Lesions | `cv/models/yolov8n_lesions_best.pt` | 5.91 MB | 1 Class | `f528dc7cf1ccafa85e63fb4f75ea3097b9197da7c89c4380b93b04fe708aacf7` | Verified Immutable |
| Subpixel Foliar Segmenter | Mobile UNet | `cv/models/mobile_unet_best.pt` | 1.90 MB | 3 Classes | `86bba6df9222b92c2d8b019fb2a10eacc2654f1a61b6e2b5996d8c8ee3cc04cc` | Verified Immutable |
| Agro Climatic Recommender | Random Forest | `cv/models/crop_recommender/crop_model.pkl` | 19.45 MB | 22 Crops (7 Features) | `cb20fc5908efe34d4e5e6e999b62e576d962ff2e1d00381db09c912bb57aa1d3` | Verified Immutable |

Ancillary Model Metadata Files:
* `cv/models/taxonomy_38.json`: Canonical 38 class biological taxonomy mapping class indices to crop species and pathological conditions.
* `cv/models/preprocessing_config.json`: Standardized input dimensions (256x256), channel order (RGB), mean values ([0.485, 0.456, 0.406]), and standard deviations ([0.229, 0.224, 0.225]).

## Verified Historical Benchmark Metrics

All accuracy, mAP, and segmentation metrics reported below represent empirical measurements obtained during offline holdout test set evaluations. These numbers serve as documented baseline capabilities and are never fabricated dynamically during live inference.

1. EfficientNetV2 S (Primary Server Classifier):
   * Test Top 1 Accuracy: 95.13%
   * Test Macro F1 Score: 0.9354
   * Expected Calibration Error (ECE): 0.0803 (Post temperature scaling calibration)
   * Input Resolution: 256x256 RGB
   * Evaluation Dataset: PlantVillage 38 Class Stratified Holdout Split

2. EfficientNetV2 S Tri Domain (Gen2 Multi Domain Classifier):
   * Test Top 1 Accuracy: 95.42%
   * Test Macro F1 Score: 0.9388
   * Expected Calibration Error (ECE): 0.0765
   * Evaluation Dataset: Tri Domain Stratified Holdout (PlantVillage, PlantDoc, PlantWild)

3. YOLO PlantDoc (Canopy Specimen Detector):
   * Validation mAP50: 0.3362 (Gen1) / 0.3584 (Gen2 Tri Domain)
   * Validation mAP50 95: 0.2361 (Gen1) / 0.2512 (Gen2 Tri Domain)
   * Annotation Level: Specimen foliage and canopy bounding

4. YOLOv8n Lesions (Granular Spot Detector):
   * Target Task: High resolution necrotic spot bounding
   * Classes: 1 (Foliar Lesion Spot)

5. Mobile UNet (Foliar Segmenter):
   * Mean Intersection over Union (mIoU): 0.7485
   * Lesion Dice Coefficient: 0.7012
   * Classes: 3 (Background, Healthy Foliar Tissue, Necrotic Lesion Tissue)

6. Crop Recommendation Model:
   * Algorithm: Random Forest Classifier
   * Features: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, Rainfall
   * Crops Supported: 22 agricultural crop categories

## Production Directory Layout

The final repository tree is structured cleanly without redundant development archives or temporary folders:

```
CvProject/
├── Dockerfile                             # Container build definition
├── docker-compose.yml                     # Production service composition
├── requirements.txt                       # Locked production runtime dependencies
├── README.md                              # Authoritative documentation index
├── RELEASE_MANIFEST.json                  # Authoritative release ledger
├── RELEASE_MANIFEST_GEN2.json             # Gen2 multi domain specification
├── .env.example                           # Documented safe configuration defaults
├── .dockerignore                          # Build context exclusion rules
├── .gitignore                             # Source control exclusion rules
├── smartcropvision_final_learning_reference.ipynb # Complete 32 stage educational notebook
├── backend/                               # FastAPI application root
│   ├── run.py                             # Server launch entry point
│   ├── app/
│   │   ├── config.py                      # Application settings and path bindings
│   │   ├── main.py                        # FastAPI application initialization
│   │   ├── api/v1/                        # Endpoint routers
│   │   ├── schemas/                       # Pydantic request and response schemas
│   │   ├── services/                      # Model registry and inference services
│   │   └── utils/                         # Explainability, metrics, and image processors
├── cv/                                    # Computer vision assets
│   ├── configs/                           # YOLO and training YAML specifications
│   ├── models/                            # Frozen production model checkpoints
│   └── test_images/                       # Authentic evaluation test images
├── data/                                  # Tabular datasets and dataset audits
├── docs/                                  # Technical architecture and API contracts
├── frontend/                              # Pure Vanilla CSS and JavaScript web interface
│   ├── index.html                         # User interface markup
│   ├── app.js                             # Client side state machine and API client
│   ├── style.css                          # Modern agricultural dark theme styling
│   └── test_frontend.js                   # Frontend state machine unit tests
├── iot/                                   # Hardware integration firmware
│   ├── esp32_rover/                       # AgriRover motor control and telemetry
│   └── xiaozhi/                           # XiaoZhi voice assistant audio protocol
├── notebooks/                             # Research notebooks
│   └── historical/                        # Preserved training records and experiments
├── packages/                              # Offline dependency wheels for isolated hosts
└── tests/                                 # Automated PyTest suite (71 passing tests)
```

## Runtime Requirements & Configuration

### Hardware and OS Requirements
* Operating System: Linux (Ubuntu 22.04 LTS recommended), macOS (Apple Silicon MPS supported), Windows (WSL2)
* Python Runtime: Python 3.10 through 3.14 compatible
* Minimum RAM: 4.0 GB (8.0 GB recommended for multi model inference)
* GPU Acceleration: NVIDIA CUDA enabled GPU (Compute 7.0 or higher), Apple Silicon Metal (MPS), or fallback CPU

### Environment Variables
All configuration options provide secure production defaults in `.env.example`:

| Variable Name | Default Value | Required / Optional | Purpose |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | `production` | Required | Enforces production security policies |
| `DEBUG` | `false` | Required | Disables verbose exception traces and debug routes |
| `HOST` | `0.0.0.0` | Optional | Network binding address |
| `PORT` | `8000` | Optional | HTTP service port |
| `CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Required | Permitted browser origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Optional | Denial of service upload ceiling |
| `MAX_IMAGE_PIXELS` | `16000000` | Optional | Decompression bomb protection ceiling |
| `ENABLE_GRADCAM` | `true` | Optional | Toggles optional visual explainability stage |
| `ENABLE_DETECTION` | `true` | Optional | Toggles optional spatial canopy detection |
| `ENABLE_SEGMENTATION` | `true` | Optional | Toggles optional foliar segmentation |
| `PRIMARY_MODEL_PATH` | `cv/models/efficientnetv2_s_best.pt` | Optional | Path to server primary classification checkpoint |
| `DETECTOR_MODEL_PATH` | `cv/models/yolo_plantdoc_best.pt` | Optional | Path to canopy specimen detector checkpoint |
| `LESION_DETECTOR_MODEL_PATH` | `cv/models/yolov8n_lesions_best.pt` | Optional | Path to granular lesion detector checkpoint |
| `SEGMENTER_MODEL_PATH` | `cv/models/mobile_unet_best.pt` | Optional | Path to subpixel foliar segmenter checkpoint |

## Truth in Telemetry and Scientific Limitations

SmartCropVision adheres strictly to truthful representation of all artificial intelligence capabilities:

1. Specimen Canopy Localization vs Granular Lesions:
   * The primary YOLO detector is trained on PlantDoc imagery where annotations delineate entire leaf specimens and foliage canopies.
   * The system explicitly labels these outputs as Specimen Foliage Canopy.
   * Granular necrotic spot coordinates are produced only when the specialized lesion model is invoked.
   * The system never converts Grad CAM heatmaps into synthetic detection boxes or applies heuristic contours to fabricate bounding boxes.

2. Foliar Segmentation Semantics:
   * Mobile UNet segments pixels into background, foliar canopy, and lesion areas.
   * When no foliar damage is present on a healthy specimen, damage area is reported truthfully as 0.0 percent.
   * When the segmentation model is unavailable or disabled, the interface displays Unavailable rather than a misleading zero score.

3. Out of Distribution Screening:
   * Non foliar imagery, severe blur, and extreme underexposure or overexposure are detected prior to neural inference.
   * When input quality is inadequate, the system issues an explicit diagnostic warning rather than presenting an authoritative false diagnosis.

4. 9 Stage Explainability Pipeline:
   * Grad CAM is computed directly against the final convolutional layer of EfficientNetV2 S (`features.7` or `conv_head`).
   * Target gradients are evaluated dynamically for the predicted or requested class.
   * Heatmaps are never cached statically or reused across different input images.

## Verified Release Packages

Two distinct release packages have been produced in `dist/`:

1. Production Runtime Package:
   * File Name: `dist/smartcropvision_v2.2.0_production.tar.gz`
   * Size: 275.27 MB
   * SHA 256 Digest: `759e22955d3e253614f87853954741dbfeb50e31012a858aa86049f5e48baadd`
   * Purpose: Lightweight container ready deployable package containing FastAPI backend, static frontend, 6 frozen model checkpoints, IoT firmware, docs, and the automated test suite. Excludes training datasets and historical research logs.

2. Educational Master Package:
   * File Name: `dist/smartcropvision_v2.2.0_educational.tar.gz`
   * Size: 92.49 KB
   * SHA 256 Digest: `c5520a900e7b847198227d842459c4f016ad1f8d4febaa4e9299e4eeac19a1df`
   * Purpose: Standalone educational archive containing the complete 32 stage pedagogical master notebook with safe read only execution controls and architectural documentation.

## External Resource Dependencies & Verification Status

| Component | Operational Status | Verification Evidence |
| :--- | :--- | :--- |
| FastAPI Backend Core | Verified | 71 automated PyTest cases passing cleanly |
| Primary EfficientNetV2 S Classifier | Verified | Dynamic inference verified on authentic sample imagery |
| Gen2 Tri Domain Classifier | Verified | Parity verified against baseline benchmarks |
| YOLO PlantDoc Canopy Detector | Verified | Multi box detection verified without array flattening |
| YOLOv8n Granular Lesion Detector | Verified | Spot detection verified on infected specimens |
| Mobile UNet Foliar Segmenter | Verified | Mask generation and damage calculation verified |
| 9 Stage Explainability Engine | Verified | Dynamic gradient computation verified |
| Frontend Web Client | Verified | 18 Node.js unit tests passing cleanly |
| Decompression Bomb Protection | Verified | 20.25 MP bomb rejection verified |
| AgriRover ESP32 Motor Controller | Requires External Resource | C++ firmware verified statically; requires physical ESP32 board |
| XiaoZhi Voice Assistant Gateway | Requires External Resource | Protocol handler verified; requires physical XiaoZhi endpoint |
| Cloud GPU Cluster Deployment | Requires External Resource | Docker container verified; cloud staging requires remote cloud quota |

Conclusion: SmartCropVision version 2.2.0 is validated, fully functional, and ready for deployment and presentation by Team Innovex.
