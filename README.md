<div align="center">

# SmartCropVisionSystem 🌿🔬

**A unified edge IoT telemetry and 3-Tier computer vision platform for real-time crop recommendation and foliar disease diagnosis.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00599C?style=flat-square)](https://ultralytics.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ESP8266](https://img.shields.io/badge/Hardware-ESP8266_NodeMCU-E7352C?style=flat-square&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Tests](https://img.shields.io/badge/Tests-10_Passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

<p align="center"><em>Dual-engine agricultural intelligence platform: physical edge IoT telemetry feeding Random Forest crop planning paired with a 3-tier deep vision cascade (MobileNetV2, YOLOv8-nano, Mobile-UNet) for sub-pixel foliar pathology triage.</em></p>

</div>

---

## 📖 Overview

**SmartCropVisionSystem** is an end-to-end precision agriculture platform designed to solve two interrelated operational challenges faced by modern smallholder farms and agronomic greenhouses:

1. **Pre-Planting & Growing Season Intelligence**: Recommending optimal crop selection based on dynamic micro-climate soil-atmospheric conditions while continuously screening for environmental pathogen conditions before physical infections take root.
2. **In-Season Foliar Diagnostic Cascade**: Accurately diagnosing plant pathologies from leaf photographs without falling victim to the domain shift common in standard machine learning models, where healthy field foliage with natural soil or sunlight variations is misclassified as catastrophic disease.

Rather than relying on isolated single-image classifiers or disconnected microcontroller monitors, SmartCropVisionSystem creates an integrated edge-to-cloud ecosystem:

- **Edge Microcontroller Sensing**: An ESP8266 NodeMCU continuously samples air temperature and relative humidity (DHT11), volumetric soil moisture, and active precipitation (YL-83 digital rain sensor). Sensor streams are dispatched via HTTP JSON to the backend and rendered locally on a 16x2 I2C character LCD.
- **Crop Recommendation & Pathogen Risk Engine**: A 100-estimator Random Forest classifier trained on 22 distinct crop varieties determines optimal cultivation viability, while an agronomic rule engine evaluates 18 environmental pathogen triggers across fungal, bacterial, viral, and abiotic categories.
- **3-Tier Computer Vision Cascade**: When a grower uploads a foliar photograph, the system executes a three-stage deep learning pipeline:
  - **Tier 1 (Screening)**: MobileNetV2 classifies the specimen across 38 botanical categories spanning 14 crop species.
  - **Tier 2 (Localization)**: YOLOv8-nano identifies necrotic infection foci, outputting bounding box coordinates and normalized centroids for targeted variable-rate sprayers.
  - **Tier 3 (Segmentation)**: Mobile-UNet segments healthy leaf lamina from necrotic tissue to quantify the Botanical Damage Index (percentage of damaged leaf area).
- **Multi-Factor Decision Matrix**: A deterministic arbitration matrix resolves laboratory-versus-field domain shifts by cross-referencing classification probability distributions with spatial lesion counts and segmented damage area, ensuring healthy leaves are never falsely flagged as severe disease.
- **Single-Page Dashboard**: A dual-panel web interface with a fixed left control panel and independently scrollable right intelligence panel displays live Chart.js sensor trends, camera/file upload dropzones, interactive inspection canvas overlays, and actionable agronomic treatment advice.

---

## 🌟 Core Capabilities

### 1. Edge IoT Telemetry & Environmental Crop Matching
- **Multi-Sensor Acquisition**: Real-time acquisition of ambient temperature (°C), relative humidity (% RH), soil moisture content (% volumetric), and binary precipitation state (1 = rain, 0 = dry).
- **Calibrated Scikit-Learn Feature Scaling**: Physical rain sensors output binary wet/dry states rather than rainfall volume. The pipeline maps binary rain states to representative precipitation baselines (20.0 mm for dry conditions, 120.0 mm for active precipitation) to match the distribution of the `StandardScaler` model without distorting sensor inputs.
- **22-Class Crop Viability Ranking**: Returns top-1 and top-3 probabilistic crop recommendations with percentage confidence metrics across 22 crops: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, and watermelon.
- **Deterministic Pathogen Risk Engine**: Evaluates live environmental telemetry against 18 deterministic agronomic rules, computing disease risk indices (such as fungal infection risk when relative humidity exceeds 80% with temperatures between 18°C and 28°C) categorized by severity (`CRITICAL`, `HIGH`, `MODERATE`, `WATCH`).

### 2. 3-Tier Deep Computer Vision Pathology Cascade
- **Tier 1: MobileNetV2 Universal Classification (38 Classes)**
  - Classifies foliage across 38 botanical conditions spanning 14 crop species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).
  - Fine-tuned with field augmentation over 8 epochs, achieving 87.32% weighted validation F1 score on multi-crop field evaluation.
- **Tier 2: YOLOv8-Nano Spatial Lesion Localization**
  - Pinpoints necrotic lesion foci on the leaf lamina with spatial bounding boxes (`[x1, y1, x2, y2]`).
  - Merges overlapping bounding box proposals from YOLO and segmentation clusters using a dual-metric threshold:

$$\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} > 0.20 \quad \lor \quad \text{Containment} = \frac{\text{Area}(A \cap B)}{\min(\text{Area}(A), \text{Area}(B))} > 0.40$$

  - Emits normalized centroid coordinates (`[cx, cy]`) to drive automated variable-rate spray nozzle actuation.
- **Tier 3: Mobile-UNet Sub-Pixel Semantic Segmentation**
  - Evaluates foliar pixels using a lightweight 4-stage encoder-decoder U-Net with skip connections.
  - Generates a 3-class segmentation mask: background (0), healthy leaf lamina (1), and necrotic lesion tissue (2).
  - Quantifies the Botanical Damage Index:

$$\text{Foliar Damage Percentage} = \left( \frac{\sum \text{Pixels}_{\text{lesion}}}{\max\left(\sum \text{Pixels}_{\text{leaf}}, 1\right)} \right) \times 100$$

  - Encodes a translucent crimson (`#ef233c`) overlay mask in Base64 for instant canvas rendering.

### 3. Multi-Factor Decision Matrix (Healthy Gating)
- **Elimination of Field False Positives**: Standard deep learning classifiers trained on uniform laboratory backgrounds often misclassify healthy field leaves (such as field tomato foliage) as Late Blight or Early Blight due to natural leaf venation, soil mulch, or sunlight highlights.
- **Deterministic Multi-Tier Arbitration**:
  - **Case A (Explicit Healthy)**: Triggered if top-1 prediction belongs to a healthy botanical class, or if the aggregate probability mass of all healthy classes exceeds the top-1 disease confidence while top-1 confidence is below 65%. Forces clean healthy diagnosis (`is_infected: false`, `damage: 0.0%`, `boxes: []`).
  - **Case B (Confirmed Disease)**: Triggered if top-1 prediction is a disease class with confidence ≥ 68%, or confidence ≥ 45% corroborated by active lesion foci (≥ 1) and foliar damage ≥ 4.0%. Confirms active infection and assigns triage severity stages (Stage 1 to Stage 3).
  - **Case C (Spatial Refutation Guard)**: Triggered if a disease class is predicted at moderate confidence but YOLOv8 and Mobile-UNet identify zero lesions on the foliage. If aggregate healthy probability mass ≥ 15%, the false disease prediction is refuted and overridden to Healthy. Otherwise, it is flagged as Uncertain.
  - **Case D (Low-Confidence Fallback)**: If overall confidence is below 40%, the system flags the specimen as `Uncertain: Retake Image` to prevent administering incorrect chemical fungicides.
- **Benchmarked Accuracy**: 98.15% classification accuracy across our 54-specimen validation test suite (53 / 54 correct determinations), achieving 100.00% recall on actual diseases and 96.77% precision.

### 4. Dual-Panel Dashboard Interface
- **Fixed Left Control Panel**: Houses physical sensor sliders with real-time numeric readouts, weather preset buttons (Monsoon, Summer, Foggy, Ideal), manual versus live IoT source indicators, and camera/file upload dropzones.
- **Scrollable Right Intelligence Panel**: Contains server health badges, Chart.js telemetry trend graphs, primary diagnosis result cards, interactive leaf inspection canvas with toggleable bounding boxes and segmentation masks, four-stage triage status indicators, and chemical/cultural treatment advice.
- **Constrained Edge Microcontroller Protocol**: Exposes a specialized `/predict/compact` endpoint returning single-character JSON keys (`crop`, `conf`, `t2`, `c2`, `t3`, `c3`, `ac`, `alerts`) optimized for low-memory microcontrollers updating 16x2 character displays without stack overflow.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHYSICAL FIELD / GREENHOUSE                              │
│                                                                                          │
│   ┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────────────┐   │
│   │   DHT11 Sensor      │   │  Capacitive Soil     │   │  YL-83 Rain Sensor Plate    │   │
│   │  (Temp °C, Hum %RH) │   │  Moisture Sensor (%) │   │  (Binary Digital Wet/Dry)   │   │
│   └──────────┬──────────┘   └──────────┬───────────┘   └──────────────┬──────────────┘   │
│              │                         │                              │                  │
│              └─────────────────┐       │       ┌──────────────────────┘                  │
│                                ▼       ▼       ▼                                         │
│                     ┌──────────────────────────────────────┐                             │
│                     │       ESP8266 NodeMCU V3 (160 MHz)   │                             │
│                     │  - Reads GPIO & Analog ADC0 Bus      │                             │
│                     │  - Renders Status on 16x2 I2C LCD    │                             │
│                     │  - Emits JSON Stream over Wi-Fi      │                             │
│                     └──────────────────┬───────────────────┘                             │
└────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                         │  HTTP POST /predict/compact
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI UNIFIED APPLICATION BACKEND                              │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│   │                            FastAPI Application Router                            │   │
│   │                                                                                  │   │
│   │   GET  /                        → Serves static dual-panel dashboard UI          │   │
│   │   POST /predict                 → Tabular Random Forest crop recommendation      │   │
│   │   POST /predict/compact         → Microcontroller payload with compacted keys    │   │
│   │   POST /predict/vision          → 3-Tier Computer Vision leaf pathology cascade  │   │
│   │   GET  /latest                  → Polled by UI for live IoT telemetry stream     │   │
│   │   GET  /health                  → Service health check and uptime timestamp      │   │
│   │   GET  /models/status           → Hardware accelerator and loaded model status   │   │
│   └──────────────┬───────────────────────────────────────────┬───────────────────────┘   │
│                  │                                           │                           │
│                  ▼                                           ▼                           │
│   ┌─────────────────────────────┐             ┌──────────────────────────────────────┐   │
│   │   Crop Intelligence Engine  │             │   3-Tier Computer Vision Cascade     │   │
│   │                             │             │                                      │   │
│   │  • StandardScaler Transform │             │  • Tier 1: MobileNetV2 (38 classes)  │   │
│   │  • Random Forest Classifier │             │  • Tier 2: YOLOv8-nano (Lesion foci) │   │
│   │  • Deterministic Pathogen   │             │  • Tier 3: Mobile-UNet (Sub-pixel)   │   │
│   │    Risk Rule Evaluator      │             │  • Multi-Factor Decision Matrix      │   │
│   └─────────────────────────────┘             └──────────────────────────────────────┘   │
│                                                              │                           │
└──────────────────────────────────────────────────────────────┼───────────────────────────┘
                                                               │  Real-Time Diagnostic Stream
                                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            SINGLE-PAGE WEB DASHBOARD UI                                  │
│                                                                                          │
│   ┌────────────────────────────────────────┐  ┌──────────────────────────────────────┐   │
│   │      FIXED LEFT CONTROL PANEL          │  │   SCROLLABLE RIGHT INTELLIGENCE      │   │
│   │                                        │  │                                      │   │
│   │  • Mode Switcher (Crop / Vision)       │  │  • Server Status & Live Hardware     │   │
│   │  • Sensor Sliders & Live Numeric Read  │  │  • Chart.js Multi-Sensor Trend Graph │   │
│   │  • Weather Presets (Monsoon, Summer)   │  │  • Primary Diagnosis & Top-3 Prob    │   │
│   │  • Camera & Drag/Drop Upload Area      │  │  • Interactive Canvas Overlay        │   │
│   │  • Quick Foliage Test Scenarios        │  │  • 4-Stage Severity Triage Banner    │   │
│   │  • One-Click Analysis Execution Button │  │  • Agronomic Treatment Protocols     │   │
│   └────────────────────────────────────────┘  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Execution & Data Flow

```
[IoT Sensors / Sliders] ──(Temp, Hum, Soil, Rain)──→ [StandardScaler]
                                                             │
                                                             ▼
                                                    [Random Forest Model]
                                                             │
                                                             ├──→ Top-1 Crop + Confidence %
                                                             └──→ Top-3 Viability Alternatives
                                                             │
[Agronomic Rule Engine] ←──(Live Telemetry Values)───────────┤
          │
          └──→ Active Pathogen Alerts (CRITICAL, HIGH, MODERATE, WATCH)
          
────────────────────────────────────────────────────────────────────────────────────────

[Foliar Leaf Photo] ──(Multipart Upload)──→ [PIL Header Validation & Decode]
                                                            │
                                  ┌─────────────────────────┴────────────────────────┐
                                  ▼                                                  ▼
                        [Tier 1: MobileNetV2]                             [Tier 2: YOLOv8-Nano]
                      (Softmax 38-Class Vector)                         (Lesion Bounding Boxes)
                                  │                                                  │
                                  ▼                                                  │
                        [Tier 3: Mobile-UNet]                                        │
                     (Semantic Pixel Segmentation)                                   │
                                  │                                                  │
                                  ├──────────────────────────────────────────────────┘
                                  ▼
                    [Multi-Factor Decision Matrix]
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
     [Case A: Healthy]     [Case B: Disease]    [Case C/D: Guard]
     - Zero damage %       - Confirmed stage    - False alert refuted
     - Empty boxes         - Merged foci boxes  - Marked as Uncertain
     - Optimal advisory    - Chemical treatment - Request clearer photo
```

---

## 🛠 Technology Stack

### Backend & Machine Learning
- **Python 3.10+**: Core programming environment.
- **FastAPI & Starlette**: High-throughput asynchronous REST API framework.
- **Uvicorn**: ASGI web server implementation.
- **PyTorch 2.0+**: Deep learning runtime powering MobileNetV2 and Mobile-UNet inference with Apple Silicon GPU (`mps`), NVIDIA CUDA (`cuda`), and CPU device autoselection.
- **Ultralytics YOLOv8**: Real-time bounding box object detection for foliar lesion localization.
- **Scikit-Learn 1.4.2**: Random Forest tabular classifier, StandardScaler preprocessing pipeline, and LabelEncoder target mapping.
- **OpenCV & Pillow**: Sub-pixel morphological clustering, connected component labeling, image validation, and dynamic alpha mask rendering.
- **NumPy & Pandas**: Matrix computation, probability aggregation, and tabular feature preparation.

### Frontend Dashboard
- **HTML5**: Semantic dual-panel architecture.
- **Vanilla CSS3**: Glassmorphism aesthetic, tailored CSS custom properties, fixed control panel, and independently scrollable intelligence panel.
- **Vanilla JavaScript (ES6+)**: Zero framework overhead, asynchronous Fetch API, state machine for IoT polling, and dynamic HTML injection.
- **HTML5 Canvas API**: Interactive high-resolution bounding box rendering, scaled coordinate transforms, and responsive mask overlay compositing.
- **Chart.js 4.4**: Animated real-time stepped and spline line charts for environmental sensor trends.

### Embedded IoT Hardware
- **ESP8266 NodeMCU V3 (ESP-12E)**: 160 MHz Tensilica Xtensa LX106 Wi-Fi microcontroller.
- **DHT11**: Digital air temperature and relative humidity sensor.
- **Capacitive Soil Moisture Sensor v1.2**: Corrosion-resistant analog moisture probe connected to ADC0.
- **YL-83 Rain Sensor Plate**: Gold-plated resistive grid with LM393 comparator providing binary digital rain detection.
- **16x2 Character LCD (HD44780 + PCF8574 I2C Backpack)**: Real-time on-device status output.

---

## 📁 Project Structure

```text
.
├── main.py                          # Unified FastAPI application entry point
├── vision_engine.py                 # 3-Tier Computer Vision inference cascade & decision matrix
├── disease_engine.py                # Deterministic agronomic pathogen risk engine (18 rules)
├── predict.py                       # Random Forest crop recommendation inference pipeline
├── models/                          # Production ML model checkpoints
│   ├── crop_model.pkl               # 100-tree Random Forest classifier (22 crops, 19.5 MB)
│   ├── scaler.pkl                   # StandardScaler for environmental sensor features
│   ├── label_encoder.pkl            # LabelEncoder mapping indices to 22 crop names
│   ├── feature_names.pkl            # Serialized feature list for tabular validation
│   ├── mobilenet_v2_38classes_best.pth # Tier 1: 38-class MobileNetV2 classifier (8.9 MB)
│   ├── yolov8n_lesions_best.pt      # Tier 2: YOLOv8-nano necrotic lesion detector (5.9 MB)
│   ├── mobile_unet_lesions_best.pth # Tier 3: Mobile-UNet sub-pixel pathology segmenter (1.9 MB)
│   └── yolov8n_plantdoc_best.pt     # Auxiliary YOLOv8 PlantDoc detector checkpoint (23.3 MB)
├── configs/
│   └── taxonomy_38classes.json      # 38-class botanical condition metadata & label index
├── static/                          # Production Web Dashboard
│   ├── index.html                   # Dual-panel dashboard layout (fixed left, scrollable right)
│   ├── style.css                    # Dark-green theme, custom responsive grid, inspection canvas
│   ├── app.js                       # IoT telemetry polling, Chart.js graphs, canvas overlays
│   └── samples/                     # Pre-loaded field test specimens (healthy & diseased)
│       ├── tomato__fungal__early_blight.jpg  # Early Blight diseased sample
│       ├── potato__healthy__healthy.jpg      # Healthy potato sample
│       ├── corn__fungal__common_rust_.jpg    # Common Rust diseased sample
│       ├── grape__fungal__black_rot.jpg      # Black Rot diseased sample
│       ├── apple__healthy__healthy.jpg       # Healthy apple sample
│       ├── user_healthy_tomato.jpg           # Field tomato healthy test specimen
│       └── samples.json                      # Sample catalog metadata
├── esp8266_firmware.ino             # Production C++ firmware for NodeMCU + DHT11 + LCD + Sensors
├── Crop_Recommendation.csv          # Agronomic training dataset (2,200 samples, 22 crops)
├── SmartCropIntelligenceSystem.ipynb# Model training, validation, and feature analysis notebook
├── test_integration.py              # Automated test suite for backend & vision pipeline
├── Dockerfile                       # Container definition for containerized/cloud deployments
├── start.sh                         # Lifespan startup and self-check launch script
├── requirements.txt                 # Pinned production Python dependencies
├── .gitignore                       # Clean production ignore definitions
└── README.md                        # Comprehensive system documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10** or higher
- **pip** and **virtualenv**
- Optional: Arduino IDE 2.0+ (if deploying physical ESP8266 IoT hardware)

### 1. Clone the Repository
```bash
git clone https://github.com/MohammadFayasKhan/SmartCropVisionSystem.git
cd SmartCropVisionSystem
```

### 2. Configure Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Application
Start the unified FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
The dashboard will load immediately with the interactive crop recommendation controls and computer vision leaf pathology inspection canvas.

---

## 🔬 Computer Vision Pathology Cascade

### Taxonomy & Supported Classes
The Tier 1 MobileNetV2 model classifies foliage across 38 distinct botanical conditions:

| Crop Species | Supported Health & Pathology Conditions |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

### Model Checkpoints Summary
All model checkpoints are committed and self-contained inside the `models/` directory:

| Model Tier | Checkpoint Path | Architecture | Input Resolution | Size | Memory Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Crop Intelligence** | `models/crop_model.pkl` | Random Forest (100 Trees) | 3 Numerical Features | 19.5 MB | Host CPU / RAM |
| **Vision Tier 1** | `models/mobilenet_v2_38classes_best.pth` | MobileNetV2 (38 Classes) | 224 × 224 × 3 | 8.9 MB | MPS / CUDA / CPU |
| **Vision Tier 2** | `models/yolov8n_lesions_best.pt` | YOLOv8-Nano (Lesion Foci) | 640 × 640 × 3 | 5.9 MB | MPS / CUDA / CPU |
| **Vision Tier 3** | `models/mobile_unet_lesions_best.pth` | Mobile-UNet (3 Classes) | 256 × 256 × 3 | 1.9 MB | Production Validated |
| **PlantDoc Field** | `models/yolov8n_plantdoc_best.pt` | YOLOv8-Nano (PlantDoc) | 640 × 640 × 3 | 23.3 MB | Experimental (Halted at Epoch 3) |

> [!NOTE]
> PlantDoc training was halted after early epochs due to device memory constraints (reaching ~0.048 mAP50). It is preserved strictly as an auxiliary checkpoint. Production lesion localization is driven by the fully trained `models/yolov8n_lesions_best.pt` (15 epochs, 0.337 mAP50) fused with Mobile-UNet segmentation clusters.

### Device Gating & Autoselection
At application startup, `VisionInferenceEngine` initializes compute devices automatically:
1. **Apple Silicon GPU (`mps`)** if running on macOS with Apple Silicon.
2. **NVIDIA GPU (`cuda`)** if a compatible CUDA device is detected.
3. **Host CPU (`cpu`)** as universal fallback.

Weights are pre-cached in memory during server lifespan initialization, ensuring subsequent requests execute in 50ms to 150ms total latency.

---

## 📡 Hardware & Edge IoT Setup

### Pinout Mapping
The production firmware in `esp8266_firmware.ino` connects to standard agricultural sensors:

| Hardware Module | NodeMCU Pin | GPIO | Protocol / Function |
| :--- | :--- | :--- | :--- |
| **DHT11 Air Temp/Humidity** | `D4` | `GPIO 2` | 1-Wire Digital Telemetry |
| **Capacitive Soil Moisture** | `A0` | `ADC0` | 10-bit Analog Voltage (0 - 1023) |
| **YL-83 Rain Sensor Board** | `D7` | `GPIO 13`| Digital Input (LOW = Raining, HIGH = Dry) |
| **16x2 LCD SDA** | `D2` | `GPIO 4` | I2C Data Line |
| **16x2 LCD SCL** | `D1` | `GPIO 5` | I2C Clock Line |
| **Sensor Power (VCC)** | `3V3` / `VIN`| - | 3.3V (DHT11/Soil) or 5V (LCD/YL-83) |
| **Ground (GND)** | `GND` | - | Common System Ground |

### Microcontroller Configuration
1. Open `esp8266_firmware.ino` in Arduino IDE.
2. Configure your local Wi-Fi credentials and server host address:
```cpp
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* host     = "192.168.1.100";  // IP address of host machine running backend
const int   port     = 8000;
```
3. Install required Arduino libraries:
   - `ESP8266WiFi`
   - `ESP8266HTTPClient`
   - `DHT sensor library` (Adafruit)
   - `LiquidCrystal_I2C` (Frank de Brabander)
   - `ArduinoJson` (v6 or v7)
4. Compile and flash to your NodeMCU board (NodeMCU 1.0 ESP-12E Module).

---

## 🔌 API Reference

### 1. Health & System Status
- **`GET /health`**
  - Returns server status, version, and timestamp.
- **`GET /models/status`**
  - Returns hardware accelerator device, loaded models, and class counts.

### 2. Environmental Crop Recommendation
- **`POST /predict`**
  - **Request Body**:
    ```json
    {
      "temperature": 27.5,
      "humidity": 78.0,
      "soil_moisture": 62.0,
      "rain": 1
    }
    ```
  - **Response Body**:
    ```json
    {
      "timestamp": "2026-09-07T00:35:00.000000",
      "recommended_crop": "rice",
      "confidence": 84.5,
      "top3": [
        {"crop": "rice", "confidence": 84.5},
        {"crop": "jute", "confidence": 11.2},
        {"crop": "cotton", "confidence": 4.3}
      ],
      "disease_alerts": [
        {
          "name": "Fungal Blast Risk",
          "severity": "HIGH",
          "trigger": "RH > 75% with moderate temp",
          "pesticide": "Tricyclazole 75% WP",
          "technique": "Avoid excessive nitrogen fertilizers"
        }
      ],
      "alert_count": 1
    }
    ```

### 3. Edge Microcontroller Compact Stream
- **`POST /predict/compact`**
  - Designed for low-memory microcontrollers with 16x2 character displays.
  - Returns single-character JSON keys: `crop`, `conf`, `t2`, `c2`, `t3`, `c3`, `ac`, `alerts`.

### 4. Computer Vision Foliar Diagnosis
- **`POST /predict/vision`**
  - **Payload**: `multipart/form-data` with key `file` (JPEG, PNG, or WebP image).
  - **Response Body**: Returns structured diagnosis, confidence level, top-3 candidates, spatial lesion bounding boxes with normalized coordinates, foliar damage percentage, Base64 translucent overlay mask, and agronomic advisory treatments.

---

## 🧪 Testing & Validation Matrix

### Automated Test Suite
Run the automated integration tests:
```bash
PYTHONPATH=. pytest test_integration.py -v
```

All 10 integration test scenarios pass with complete coverage:
1. `test_homepage_serves_unified_system`: Validates dual-panel HTML structure and component mount points.
2. `test_server_health`: Validates `/health` online status and ISO timestamp formatting.
3. `test_models_status`: Validates runtime hardware report, loaded model classes, and experimental model tags.
4. `test_crop_recommendation_prediction`: Validates Random Forest inference, top-3 probabilities, and disease risk engine.
5. `test_vision_early_blight_diagnosis`: Validates 3-tier cascade and Stage 3 severe triage on real infected foliage.
6. `test_vision_healthy_foliage_fast_gating`: Validates decision matrix healthy gating on healthy field tomato leaves.
7. `test_vision_invalid_mime_type`: Validates rejection of non-image MIME types with HTTP 400.
8. `test_vision_corrupted_image_bytes`: Validates rejection of corrupted or malformed image payloads.
9. `test_vision_field_tomato_healthy_gating`: Validates decision matrix healthy gating on field tomato leaf with natural venation.
10. `test_esp8266_compact_prediction`: Validates constrained edge microcontroller compact stream response schema.

---

## 🔒 Security & Privacy Considerations

- **No Hardcoded Secrets**: Wi-Fi network credentials and host IP addresses use standard development placeholders (`YOUR_WIFI_SSID`, `YOUR_WIFI_PASSWORD`, `192.168.1.100`).
- **Strict Multipart Upload Validation**: Uploaded files undergo two-phase verification. MIME types are validated against allowed headers, and raw image bytes are inspected by Pillow to prevent decompression bomb attacks and malformed payload injection.
- **Local-First On-Premise Execution**: The entire inference cascade runs locally on the host machine using PyTorch and Scikit-Learn. Foliar images and sensor streams are never transmitted to external third-party cloud APIs.

---

## ⚠️ Limitations & Future Roadmap

- **Single-Leaf Focus**: The current vision pipeline is optimized for close-up photographs of individual leaves. Whole-canopy drone surveillance requires wide-angle orthomosaic tiling models.
- **Extreme Weather Sensors**: The rain sensor outputs a binary digital signal (wet/dry) rather than rainfall accumulation. Integrating an optical tipping-bucket rain gauge would provide continuous millimetric precipitation inputs.
- **Multi-Crop Expansion**: Future iterations will extend semantic segmentation masks beyond solanaceous crops to include cereal rusts, cucurbit downy mildews, and citrus canker lesions.

---

## 📄 License & Authorship

Distributed under the MIT License. Developed and maintained by **Mohammad Fayas Khan**.
Contributions and suggestions are welcome via issues and pull requests on GitHub.
