# SmartCropVision System Architecture

## 1. Architectural Overview

SmartCropVision is an end to end multi task agricultural vision and edge intelligence platform. The system operates on a client server topology where edge devices capture photographs and stream telemetry, while high capacity computer vision models remain strictly on the centralized server.

```
[ Foliage Camera / Web Client / AgriRover / ESP8266 ]
                     │
                     │  (HTTP Multipart / WebSocket Telemetry)
                     ▼
           [ FastAPI Backend Gateway ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [ Preflight Audit ]     [ Model Registry ]
   * Laplacian blur        * Artifact checksums
   * Dynamic range         * Memory warm up
   * Non plant check       * Concurrency locks
         │
         ▼
 ┌──────────────────────────────────────────────────────────┐
 │               Multi Tier Inference Pipeline               │
 │                                                          │
 │  Tier 1: EfficientNetV2 S (Universal Classification)     │
 │  Tier 2: YOLO PlantDoc (Canopy Specimen Detection)       │
 │  Tier 2b: YOLOv8n Lesions (Granular Spot Localization)   │
 │  Tier 3: Mobile UNet (Sub Pixel Foliar Segmentation)     │
 │  Tabular: Random Forest (Agro Climatic Recommender)      │
 └──────────────────────────────────────────────────────────┘
         │
         ▼
 [ 9 Stage Explainability Pipeline ]
   * Grad CAM Botanical Heatmaps
   * Activation Collages
   * Foliar Damage Index Calculation
         │
         ▼
 [ Unified Diagnostic JSON Response ]
         │
 ┌───────┴───────────────────────────────┐
 │                                       │
 ▼                                       ▼
[ Web Client Dashboard ]    [ AgriRover & XiaoZhi Assistant ]
 * Dual layer canvas         * Precision spray actuation
 * Telemetry cards           * Speech synthesis via MCP
 * Agronomic advisory        * Preserved uncertainty
```

## 2. Server Side Execution Boundary

A core security and resource constraint of SmartCropVision is that production neural network weights execute exclusively on the backend server.
* The browser never downloads PyTorch or YOLO weight files.
* AgriRover firmware never stores convolutional checkpoints.
* ESP8266 microcontrollers only transmit sensor telemetry and receive plain text advisory summaries.
* XiaoZhi voice interfaces interact through the Model Context Protocol (MCP) by querying backend endpoints.

This design guarantees that model weights cannot be exfiltrated through client inspection, reduces edge device hardware costs, and prevents mobile browser memory exhaustion.

## 3. The 3 Tier Vision Pipeline

Unlike conventional agricultural models that force a single network to perform diagnosis, boundary detection, and pixel segmentation simultaneously, SmartCropVision partitions computer vision into three distinct mathematical tasks:

### Tier 1: Universal Plant Pathology Classification
* **Model**: EfficientNetV2 S
* **Input**: 256x256 normalized RGB tensor
* **Output**: Softmax probability distribution across 38 canonical classes
* **Function**: Answers the question What is the botanical identity and pathology of this leaf?
* **Key Evidence**: Test Top 1 Accuracy of 95.13% and Macro F1 of 0.9354 with Expected Calibration Error of 0.0803.

### Tier 2: Spatial Specimen & Lesion Detection
* **Models**: YOLO PlantDoc (Specimen foliage) and YOLOv8n Lesions (Granular spots)
* **Input**: 640x640 letterbox tensor
* **Output**: Normalized bounding coordinates, class labels, and confidence scores
* **Function**: Answers the question Where are the individual leaves located and where are the necrotic spots?
* **Separation of Semantics**: Whole leaf boundaries (emerald layer) and granular necrotic foci (coral layer) are processed independently to prevent misrepresenting leaf outlines as disease spots.

### Tier 3: Sub Pixel Foliar Segmentation
* **Model**: Mobile UNet
* **Input**: 256x256 RGB tensor
* **Output**: 3 class pixel mask (Class 0: Background, Class 1: Healthy Foliage, Class 2: Necrotic Lesions)
* **Function**: Answers the question Exactly which pixels contain diseased tissue?
* **Metric Formulation**: Foliar Damage Percentage = (Lesion Pixels / Leaf Pixels) * 100.0. If segmentation is unavailable, the system outputs Unavailable rather than an invented zero percentage.

## 4. Nine Stage Explainability Engine

To satisfy agricultural trust requirements and prevent black box misdiagnoses, the backend executes nine distinct computational stages for explainability requests:

1. **Stage 1: Specimen Ingestion & Quality Audit** → Decodes image bytes, assesses Laplacian focus variance, and verifies foliage greenness index.
2. **Stage 2: Color Standardization** → Bilinear interpolation to 256x256 with RGB color space validation.
3. **Stage 3: Tensor Formulation** → ImageNet channel normalization producing [1, 3, 256, 256] Float32 tensors.
4. **Stage 4: Intermediate Activation Extraction** → Extracts feature maps from early convolutional layers capturing leaf venation and textures.
5. **Stage 5: Botanical Grad CAM** → Backpropagates gradients from the top predicted logit to the final convolutional feature maps, producing saliency heatmaps.
6. **Stage 6: Softmax Uncertainty Projection** → Calculates Shannon entropy and top 1 versus top 2 prediction margin to isolate ambiguous samples.
7. **Stage 7: Spatial Detection Breakdown** → Renders verified YOLO bounding boxes with explicit non zero validation.
8. **Stage 8: Foliar Damage Index** → Computes necrotic pixel ratio from clean Mobile UNet semantic masks.
9. **Stage 9: Integrated Agronomic Decision Synthesis** → Merges multi modal telemetry with computer vision evidence into grower advisory.

## 5. Concurrency and Thread Safety

Inference invocations on persistent devices (such as Apple Silicon MPS or NVIDIA CUDA) are wrapped in explicit Python threading locks. This guarantees that simultaneous user requests or asynchronous sensor streams cannot cause GPU memory allocation collisions or corrupt intermediate feature maps. Uploaded image payloads are capped at 15 MB with decompression bomb protection limiting maximum decoded pixels to 16,000,000.
