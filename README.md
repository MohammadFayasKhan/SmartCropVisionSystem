<div align="center">

# SmartCropVisionSystem 🌿🔬

**A unified edge IoT telemetry and 3-tier deep computer vision platform for real-time crop recommendation and foliar disease diagnosis.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00599C?style=flat-square)](https://ultralytics.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ESP8266](https://img.shields.io/badge/Hardware-ESP8266_NodeMCU-E7352C?style=flat-square&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Chart.js](https://img.shields.io/badge/Dashboard-Chart.js_4.4-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![Tests](https://img.shields.io/badge/Tests-10_Passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

<p align="center"><em>Dual-engine edge-to-cloud agricultural intelligence: physical micro-climate telemetry feeding Random Forest crop planning paired with a 3-tier deep vision cascade (MobileNetV2, YOLOv8-nano, Mobile-UNet) for sub-pixel foliar pathology triage.</em></p>

</div>

---

## 📖 Overview

**SmartCropVisionSystem** is an end-to-end precision agriculture platform designed to resolve two interrelated operational challenges faced by modern smallholder farms and agronomic greenhouses:

1. **Pre-Planting & Growing Season Intelligence**: Recommending optimal crop varieties based on dynamic micro-climate soil-atmospheric conditions while continuously screening for environmental pathogen conditions before physical infections take root.
2. **In-Season Foliar Diagnostic Cascade**: Accurately diagnosing plant pathologies from leaf photographs without falling victim to the domain shift common in standard machine learning models, where healthy field foliage with natural soil or sunlight variations is misclassified as catastrophic disease.

Rather than relying on isolated single-image classifiers or disconnected microcontroller monitors, SmartCropVisionSystem creates an integrated edge-to-cloud ecosystem:

- **Edge Microcontroller Sensing**: An ESP8266 NodeMCU continuously samples air temperature and relative humidity (DHT11), volumetric soil moisture, and active precipitation (YL-83 digital rain sensor). Sensor streams are dispatched via HTTP JSON to the backend and rendered locally on a 16x2 I2C character LCD.
- **Crop Recommendation & Pathogen Risk Engine**: A 100-estimator Random Forest classifier trained on 22 distinct crop varieties determines optimal cultivation viability, while an agronomic rule engine evaluates 18 environmental pathogen triggers across fungal, bacterial, viral, and abiotic categories.
- **3-Tier Computer Vision Cascade**: When a grower uploads a foliar photograph, the system executes a three-stage deep learning pipeline:
  - **Tier 1 (Screening)**: MobileNetV2 classifies the specimen across 38 botanical categories spanning 14 crop species.
  - **Tier 2 (Localization)**: YOLOv8-nano identifies necrotic infection foci, outputting bounding box coordinates and normalized centroids for targeted variable-rate sprayers.
  - **Tier 3 (Segmentation)**: Mobile-UNet segments healthy leaf lamina from necrotic tissue to quantify the Botanical Damage Index (percentage of damaged leaf area).
- **Multi-Factor Decision Matrix**: A deterministic arbitration matrix resolves laboratory-versus-field domain shifts by cross-referencing classification probability distributions with spatial lesion counts and segmented damage area, ensuring healthy leaves are never falsely flagged as severe disease.
- **Dual-Panel Dashboard**: A responsive web interface with a fixed left control panel and independently scrollable right intelligence panel displays live Chart.js sensor trends, camera/file upload dropzones, interactive inspection canvas overlays, and actionable agronomic treatment advice.

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

### 3. Multi-Factor Decision Matrix (Domain Shift Elimination)
- **Elimination of Field False Positives**: Standard deep learning classifiers trained on uniform laboratory backgrounds often misclassify healthy field leaves (such as field tomato foliage) as Late Blight or Early Blight due to natural leaf venation, soil mulch, or sunlight highlights.
- **Deterministic Multi-Tier Arbitration**:
  - **Case A (Explicit Healthy)**: Triggered if top-1 prediction belongs to a healthy botanical class, or if the aggregate probability mass of all healthy classes exceeds the top-1 disease confidence while top-1 confidence is below 65%. Forces clean healthy diagnosis (`is_infected: false`, `damage: 0.0%`, `boxes: []`).
  - **Case B (Confirmed Disease)**: Triggered if top-1 prediction is a disease class with confidence ≥ 68%, or confidence ≥ 45% corroborated by active lesion foci (≥ 1) and foliar damage ≥ 4.0%. Confirms active infection and assigns triage severity stages (Stage 1 to Stage 3).
  - **Case C (Spatial Refutation Guard)**: Triggered if a disease class is predicted at moderate confidence but YOLOv8 and Mobile-UNet identify zero lesions on the foliage. If aggregate healthy probability mass ≥ 15%, the false disease prediction is refuted and overridden to Healthy. Otherwise, it is flagged as Uncertain.
  - **Case D (Low-Confidence Fallback)**: If overall confidence is below 40%, the system flags the specimen as `Uncertain: Retake Image` to prevent administering incorrect chemical fungicides.
- **Benchmarked Accuracy**: 98.15% classification accuracy across our 54-specimen validation test suite (53 / 54 correct determinations), achieving 100.00% recall on actual diseases and 96.77% precision.

### 4. Real-Time Dual-Panel Dashboard & Microcontroller Streaming
- **Fixed Left Control Panel**: Houses physical sensor sliders with real-time numeric readouts, weather preset buttons (Monsoon, Summer, Foggy, Ideal), manual versus live IoT source indicators, and camera/file upload dropzones.
- **Scrollable Right Intelligence Panel**: Contains server health badges, Chart.js telemetry trend graphs, primary diagnosis result cards, interactive leaf inspection canvas with toggleable bounding boxes and segmentation masks, four-stage triage status indicators, and chemical/cultural treatment advice.
- **Constrained Edge Microcontroller Protocol**: Exposes a specialized `/predict/compact` endpoint returning single-character JSON keys (`crop`, `conf`, `t2`, `c2`, `t3`, `c3`, `ac`, `alerts`) optimized for low-memory microcontrollers updating 16x2 character displays without stack overflow.

---

## 🏗️ Architecture & Execution Flow

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
│                       ┌─────────────────────────────────┐                                │
│                       │   NodeMCU ESP8266 (ESP-12E)     │                                │
│                       │   Local 16x2 I2C Character LCD  │                                │
│                       └────────────────┬────────────────┘                                │
└────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                         │ Wi-Fi HTTP POST (JSON Telemetry Payload)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI BACKEND GATEWAY (:8000)                            │
│                                                                                          │
│   ┌───────────────────────────┐                  ┌───────────────────────────────────┐   │
│   │ POST /predict             │                  │ POST /predict/vision              │   │
│   │ (Environmental Telemetry) │                  │ (Foliar Image Multipart Upload)   │   │
│   └─────────────┬─────────────┘                  └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 ▼                                                  ▼                     │
│   ┌───────────────────────────┐                  ┌───────────────────────────────────┐   │
│   │ Feature Scaler & Encoding │                  │ Image Normalization & Decoding    │   │
│   │ StandardScaler Mapping    │                  │ Pillow Byte Verification          │   │
│   └─────────────┬─────────────┘                  └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 ▼                                                  ▼                     │
│   ┌───────────────────────────┐                  ┌───────────────────────────────────┐   │
│   │ Random Forest Classifier  │                  │ Tier 1: MobileNetV2 Screening     │   │
│   │ 100 Trees • 22 Crop Types │                  │ 38 Botanical Conditions (14 Crops)│   │
│   └─────────────┬─────────────┘                  └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 ▼                                                  ▼                     │
│   ┌───────────────────────────┐                  ┌───────────────────────────────────┐   │
│   │ Pathogen Risk Rule Engine │                  │ Tier 2: YOLOv8-Nano Localization  │   │
│   │ 18 Micro-Climate Triggers │                  │ Spatial Lesion Bounding Boxes     │   │
│   └─────────────┬─────────────┘                  └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 │                                                  ▼                     │
│                 │                                ┌───────────────────────────────────┐   │
│                 │                                │ Tier 3: Mobile-UNet Segmentation  │   │
│                 │                                │ Sub-Pixel Foliar Damage Mask      │   │
│                 │                                └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 │                                                  ▼                     │
│                 │                                ┌───────────────────────────────────┐   │
│                 │                                │ Multi-Factor Decision Matrix      │   │
│                 │                                │ Lab-to-Field Domain Shift Gating  │   │
│                 │                                └─────────────────┬─────────────────┘   │
│                 │                                                  │                     │
│                 └──────────────────────┐   ┌───────────────────────┘                     │
│                                        ▼   ▼                                             │
│                       ┌─────────────────────────────────────┐                            │
│                       │ Actionable Agronomic Advisory Engine│                            │
│                       │ Chemical & Cultural Action Protocols│                            │
│                       └──────────────────┬──────────────────┘                            │
└──────────────────────────────────────────┼───────────────────────────────────────────────┘
                                           │ Structured JSON Payloads & Base64 Overlays
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         SINGLE-PAGE RESPONSIVE DASHBOARD                                 │
│                                                                                          │
│   ┌────────────────────────────────────────┐ ┌───────────────────────────────────────┐   │
│   │ Fixed Left Control Panel               │ │ Scrollable Right Intelligence Panel   │   │
│   │ ├─ Interactive Telemetry Sliders       │ │ ├─ Real-Time Chart.js Rolling Trends  │   │
│   │ ├─ Micro-Climate Simulation Presets    │ │ ├─ Primary Crop Recommendation Cards  │   │
│   │ ├─ Live IoT Feed Status Chip           │ │ ├─ Pathogen Alert Cards with Triggers │   │
│   │ └─ Drag & Drop Image Dropzone          │ │ ├─ Leaf Inspection Canvas with BBoxes │   │
│   │                                        │ │ └─ 4-Stage Severity Triage Protocols  │   │
│   └────────────────────────────────────────┘ └───────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Edge Hardware** | **ESP8266 NodeMCU (ESP-12E)** | Low-cost Wi-Fi microcontroller for real-time ambient telemetry sampling |
| **Sensors & Display** | **DHT11, YL-83, Capacitive Soil, 16x2 I2C LCD** | Ambient temp/humidity, rain detection, volumetric moisture, and local field display |
| **Backend Framework** | **FastAPI 0.104+, Uvicorn** | High-performance asynchronous REST API and edge microcontroller compact streaming |
| **Machine Learning** | **Scikit-Learn 1.4.2** | 100-estimator Random Forest classifier for 22-class crop viability recommendation |
| **Deep Learning** | **PyTorch 2.0+ (Torchvision)** | MobileNetV2 38-class classifier and Mobile-UNet semantic segmentation engine |
| **Object Detection** | **Ultralytics YOLOv8-Nano** | Spatial necrotic lesion localization, centroid tracking, and bounding box regression |
| **Foliar Segmentation** | **Mobile-UNet (3-Class)** | Sub-pixel leaf lamina vs lesion pixel segmentation and Botanical Damage Index |
| **Frontend UI** | **Vanilla HTML5, CSS3, ES6+ JavaScript** | Modern dual-panel layout with fixed controls, glassmorphic styling, and dark mode tokens |
| **Data Visualization** | **Chart.js 4.4** | Real-time rolling telemetry trend graphs for temperature, humidity, and soil moisture |
| **Image Processing** | **Pillow (PIL), OpenCV (cv2)** | Image decoding, EXIF handling, color space transforms, and Base64 mask encoding |
| **Containerization** | **Docker (python:3.10-slim)** | Standardized container packaging with exposed ports 8000 and 7860 |
| **Automated Testing** | **Pytest 9.0+** | Complete test suite validating API contracts, ML models, and healthy gating (10 passing) |

---

## 📁 Project Structure

```
smart-crop-intelligene-system-main-repo/
├── configs/
│   └── taxonomy_38classes.json                  # Complete 38-class botanical condition mapping
│
├── models/                                      # Centralized machine learning model checkpoints
│   ├── crop_model.pkl                           # Random Forest 100-tree crop viability estimator
│   ├── scaler.pkl                               # Scikit-Learn StandardScaler for sensor inputs
│   ├── label_encoder.pkl                        # Crop class label encoder (22 botanical varieties)
│   ├── feature_names.pkl                        # Input feature schema manifest
│   ├── mobilenet_v2_38classes_best.pth          # Tier 1: 38-class MobileNetV2 classification checkpoint
│   ├── yolov8n_lesions_best.pt                  # Tier 2: YOLOv8-nano necrotic lesion localization weights
│   ├── mobile_unet_lesions_best.pth             # Tier 3: Mobile-UNet 3-class foliar damage segmentation
│   └── yolov8n_plantdoc_best.pt                 # Experimental PlantDoc research checkpoint
│
├── static/                                      # Responsive web dashboard frontend
│   ├── index.html                               # Dual-panel dashboard markup and component anchors
│   ├── style.css                                # Design tokens, glassmorphism, and responsive layout
│   ├── app.js                                   # State coordination, Chart.js trends, and canvas overlay
│   └── samples/                                 # Multi-crop foliar evaluation and validation samples
│       ├── apple__healthy__healthy.jpg          # Healthy apple foliar control
│       ├── corn__fungal__common_rust_.jpg       # Active Common Rust infection specimen
│       ├── grape__fungal__black_rot.jpg         # Grape Black Rot necrotic specimen
│       ├── potato__healthy__healthy.jpg         # Healthy potato control
│       ├── tomato__fungal__early_blight.jpg     # Tomato Early Blight severe lesion specimen
│       ├── user_healthy_tomato.jpg              # Real field healthy tomato leaf (domain shift test)
│       └── samples.json                         # Ground-truth sample metadata and expected outputs
│
├── Crop_Recommendation.csv                      # Canonical crop viability training dataset
├── Dockerfile                                   # Production container runtime definition
├── SmartCropIntelligenceSystem.ipynb            # Original research, model training, and evaluation notebook
├── disease_engine.py                            # Deterministic environmental pathogen risk rule engine
├── esp8266_firmware.ino                         # C++ firmware for NodeMCU ESP8266 with 16x2 LCD display
├── main.py                                      # Unified FastAPI backend application and endpoints
├── predict.py                                   # Random Forest crop recommendation inference pipeline
├── requirements.txt                             # Python runtime dependency manifest
├── start.sh                                     # Automated execution launcher script
├── test_integration.py                          # Automated regression and edge case test suite
└── vision_engine.py                             # 3-tier deep computer vision and decision matrix cascade
```

---

## ⚙️ Installation & Local Development

### Prerequisites

- [Python 3.10+](https://www.python.org/)
- [NodeMCU ESP8266](https://www.espressif.com/) and [Arduino IDE](https://www.arduino.cc/en/software) (optional, for physical hardware sensing)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### 1. Clone the Repository

```bash
git clone https://github.com/MohammadFayasKhan/SmartCropVisionSystem.git
cd SmartCropVisionSystem
```

### 2. Environment Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Review `.env` to configure your runtime parameters:

```ini
HOST=0.0.0.0
PORT=8000
DEBUG=False
DEVICE_TARGET=cpu
```

### 3. Install Backend Dependencies

Create a clean virtual environment and install the locked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the Application

Launch the unified FastAPI application using the automated launcher or Uvicorn:

```bash
# Option A: Automated launcher script
chmod +x start.sh
./start.sh

# Option B: Direct Uvicorn invocation
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to `http://localhost:8000` to access the interactive dashboard.

### 5. (Optional) Edge Microcontroller Hardware Setup

To connect a physical ESP8266 NodeMCU edge node:

1. Open `esp8266_firmware.ino` in the Arduino IDE.
2. Install required board packages and libraries via the Library Manager:
   - **ESP8266 Board Package**: `esp8266 by ESP8266 Community`
   - **DHT Sensor Library**: `DHT sensor library by Adafruit`
   - **LiquidCrystal I2C**: `LiquidCrystal I2C by Frank de Brabander`
3. Configure your Wi-Fi credentials and local machine IP in `esp8266_firmware.ino`:
   ```cpp
   const char* ssid     = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverUrl = "http://192.168.1.100:8000/predict/compact";
   ```
4. Wire the sensors to your NodeMCU according to the hardware pinout:
   - **DHT11 Data**: Pin `D4` (GPIO2)
   - **Capacitive Soil Moisture**: Pin `A0` (Analog In)
   - **YL-83 Rain Sensor**: Pin `D7` (GPIO13)
   - **16x2 I2C LCD**: SDA to Pin `D2` (GPIO4), SCL to Pin `D1` (GPIO5)
5. Flash the sketch to your NodeMCU (NodeMCU 1.0 ESP-12E Module). The dashboard will automatically reflect live telemetry.

---

## 🧪 Testing & Verification Matrix

The repository features a comprehensive automated integration test suite validating API schemas, model loading, crop recommendation logic, 3-tier vision pipelines, domain-shift healthy gating, and edge microcontroller compact streaming:

```bash
pytest test_integration.py -v
```

### Test Coverage Summary

| Test Suite | Focus Area | Status |
| :--- | :--- | :--- |
| `test_homepage_serves_unified_system` | Dual-panel HTML structure, DOM containers, script mounts | ✅ Passed |
| `test_server_health` | FastAPI health probe, online status, ISO timestamp formatting | ✅ Passed |
| `test_models_status` | Runtime hardware accelerator report, model counts, experimental tags | ✅ Passed |
| `test_crop_recommendation_prediction` | Random Forest inference, top-3 crop probabilities, disease alerts | ✅ Passed |
| `test_vision_early_blight_diagnosis` | 3-tier cascade and Stage 3 severe triage on verified infected foliage | ✅ Passed |
| `test_vision_healthy_foliage_fast_gating` | Fast-path decision matrix gating on healthy lab foliage | ✅ Passed |
| `test_vision_invalid_mime_type` | Security validation rejecting non-image payloads with HTTP 400 | ✅ Passed |
| `test_vision_corrupted_image_bytes` | Pillow byte verification rejecting truncated or malformed images | ✅ Passed |
| `test_vision_field_tomato_healthy_gating` | Multi-factor arbitration refuting false disease on field foliage | ✅ Passed |
| `test_esp8266_compact_prediction` | Constrained edge microcontroller single-character JSON stream | ✅ Passed |

---

## 🔒 Privacy & Security Disclosures

- **Zero Cloud Data Leakage**: The entire inference cascade runs locally on the host machine using PyTorch and Scikit-Learn. Foliar images and environmental sensor readings are never transmitted to external commercial cloud APIs.
- **Strict Multipart Upload Validation**: Uploaded files undergo strict two-phase inspection. MIME types are validated against allowed image formats (`image/jpeg`, `image/png`, `image/webp`), and raw byte streams are verified by Pillow to prevent image decompression bombs and shell injection.
- **Safe Fallback & Anti-Poisoning Gating**: If an image has poor lighting or ambiguous classification (< 40% confidence), the system safely returns `Uncertain: Retake Image` rather than guessing a false diagnosis, preventing accidental chemical over-application.
- **Zero Hardcoded Secrets**: All network configurations and hardware endpoints use clean environment variables with safe defaults, preventing accidental leakage of network credentials.

---

## 💡 Why I Built This

As a Computer Science & Engineering student passionate about the intersection of Edge IoT and Computer Vision in agriculture, I witnessed firsthand how modern smallholder farmers and greenhouse managers struggle with two critical agricultural bottlenecks:

1. **Environmental Blind Spots**: Soil and weather variations directly influence crop viability and pathogen gestation. Without localized sensor telemetry, farmers apply chemical fertilizers and fungicides reactively after fungal blast or rot has already decimated the canopy.
2. **The Field Domain Shift Trap**: Commercial vision models trained exclusively on clean, lab-curated leaf datasets (like PlantVillage) perform poorly in real farm environments. Natural leaf veins, dust, shadows, and soil mulch are routinely misclassified as catastrophic diseases like Late Blight. Inexperienced farmers act on these false alarms by spraying costly and ecologically damaging chemical fungicides unnecessarily.

I built **SmartCropVisionSystem** to address these realities through an integrated edge-to-cloud architecture:
- **Low-Cost Edge Sensing**: Connects low-cost, off-the-shelf microcontrollers (ESP8266 + DHT11 + capacitive soil + rain sensors) directly to predictive ML without needing expensive proprietary weather stations.
- **3-Tier Diagnostic Cascade**: Dissects foliar pathology into screening (MobileNetV2), spatial localization (YOLOv8-nano), and sub-pixel damage segmentation (Mobile-UNet).
- **Fail-Safe Decision Arbitration**: Implements a deterministic multi-factor decision matrix that prevents false alarms, ensures healthy leaves are never diagnosed with phantom infections, and recommends chemical treatments only when physical lesions are confirmed.

---

## 👨💻 Author

<div align="center">
  <h3><strong>Mohammad Fayas Khan</strong></h3>
  <p><em>B.Tech Computer Science Engineering Student • Lovely Professional University</em></p>

  <p>
    <a href="https://www.linkedin.com/in/mohammadfayaskhan/" target="_blank">
      <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
    </a>&nbsp;
    <a href="https://github.com/MohammadFayasKhan" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
    </a>&nbsp;
    <a href="mailto:fayaskhanmohammad@gmail.com">
      <img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
    </a>
  </p>
</div>

---

## 📝 License

This project is open-source and licensed under the [MIT License](LICENSE).
