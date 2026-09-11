# SmartCropVision Final Demonstration Checklist

Target Audience: Team Innovex Presenters and SIH Evaluators
Authoritative Version: 2.2.0
Release Tag: v2.2.0
Document Purpose: Step by step operational guide for a flawless live technical demonstration

## 1. Pre Presentation Environmental Setup

Complete these verification checks at least 15 minutes before presenting to the evaluation committee.

### Step 1: Start the Backend Service
From the repository root directory, launch the production server:
```bash
python3 backend/run.py
```
* Expected Console Output: Uvicorn running on `http://127.0.0.1:8000`
* Dynamic Device Detected: Confirm whether `MPS`, `CUDA`, or `CPU` is surfaced

### Step 2: Confirm System Health and Model Readiness
In a separate terminal or browser tab, verify that all models are active:
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models/status
```
* Verify that `status` equals `healthy`
* Verify that all 8 models in the registry report `loaded: true`

### Step 3: Launch User Interface
Open your preferred web browser and navigate to:
```
http://127.0.0.1:8000/
```
* Verify header status badge displays `Online` with authentic device telemetry
* Ensure dropzone is receptive and initial sample buttons are clickable

## 2. Recommended Live Presentation Sequence

Follow this structured workflow to showcase the system capabilities logically:

### Stage 1: Problem Introduction and Architecture Overview
* Talking Point: Foliar crop diseases reduce smallholder yields by over 30 percent annually.
* Architectural Solution: SmartCropVision couples edge IoT rovers with a robust multi stage server vision pipeline.
* Emphasize: Production model checkpoints are hosted server side. Mobile browsers and microcontrollers never download heavy model weights.

### Stage 2: Negative Control Demo (Healthy Leaf)
1. Select a healthy sample leaf from the test image library or click sample healthy preview.
2. Click `Analyze Crop Health`.
3. Highlight to Evaluators:
   * Model identifies specimen condition accurately as `Healthy`.
   * Spatial detection produces exactly 0 bounding boxes.
   * Explain: The system never fabricates synthetic boxes or hallucinates lesions when foliage is healthy.
   * Segmentation damage index reports 0.0 percent.

### Stage 3: Infected Foliar Canopy Diagnosis (Multi Target Detection)
1. Click the strawberry test sample (`cv/test_images/Strawberry-Leaves.jpg`) or drag and drop it into the dropzone.
2. Click `Analyze Crop Health`.
3. Walkthrough the Results:
   * Primary Classification: Surfaces `Strawberry___Leaf_scorch` with calibrated confidence.
   * Spatial Detection Visualization: Delineates 3 independent canopy specimen boxes. Demonstrate that hovering or inspecting coordinates preserves all 3 detections without array flattening.
   * Subpixel Foliar Segmentation: Toggle the mask overlay to show exact foliar tissue delineation and computed leaf damage percentage.
   * 9 Stage Explainability Pipeline: Review the dynamic Grad CAM activation map showing peak attention focused on necrotic margins rather than background artifacts.
   * Agronomic Advisory: Point out actionable organic and chemical management strategies tailored to the diagnosed disease.

### Stage 4: Tomato Septoria Complex Foliage Test
1. Load `cv/test_images/tomato-badleaves.jpg`.
2. Observe 7 distinct foliage bounding boxes detected across dense canopy leaves.
3. Highlight that coordinate transformations account for aspect ratio, padding, and fullscreen zoom without box distortion.

### Stage 5: Agro Climatic Crop Recommendation
1. Switch to the `Crop Planning` panel in the navigation bar.
2. Enter regional soil and climate parameters:
   * Nitrogen: 90, Phosphorus: 42, Potassium: 43
   * Temperature: 21.0 C, Humidity: 82.0%
   * Soil pH: 6.5, Rainfall: 202.0 mm
3. Click `Recommend Optimal Crop`.
4. Observe instantaneous machine learning recommendation for optimal crop selection (`Rice`) with primary nutritional suitability metrics.

### Stage 6: IoT Rover Telemetry and Control
1. Switch to the `AgriRover` control section.
2. If physical AgriRover ESP32 is connected to the local network:
   * Demonstrate directional motor commands: Forward, Backward, Left, Right.
   * Showcase real time environmental sensor telemetry streaming over WebSocket.
3. Fallback Procedure if Hardware is Absent:
   * Explain to evaluators: AgriRover uses ESP32 C++ firmware communicating over WebSocket.
   * The UI displays `Rover Offline (Requires Physical Hardware)` without crashing or presenting false telemetry.
   * Reiterate that motor control semantics remain isolated from neural inference pipelines.

### Stage 7: XiaoZhi Voice Interaction Gateway
1. Introduce the XiaoZhi multilingual voice assistant integration.
2. If XiaoZhi hardware endpoint is active:
   * Demonstrate natural language voice queries regarding disease severity and spray schedules.
   * Note how XiaoZhi accesses the current session identifier without requesting checkpoint weights.
3. Fallback Procedure if Voice Node is Absent:
   * The web application continues to operate seamlessly.
   * Point out the structured REST and WebSocket protocol endpoints documented in `docs/XIAOZHI_INTEGRATION.md`.

## 3. Evaluator Question Defense Guide

Prepare for these common technical questions from SIH judges:

* Question 1: Why does the detector output canopy boxes instead of individual lesion dots?
  ⤷ Answer: Our primary YOLO detector is trained on PlantDoc imagery, where ground truth labels delineate leaf specimens. For granular necrotic spot localization, we provide a specialized YOLOv8n lesion detector. We never falsely label canopy boxes as lesion spots.

* Question 2: Is your Grad CAM precomputed or static?
  ⤷ Answer: Grad CAM is computed dynamically on the server during each inference call by performing backpropagation against the final convolutional layer of EfficientNetV2 S. Every uploaded image produces a unique activation map.

* Question 3: How do you handle out of distribution or non leaf images?
  ⤷ Answer: Our inference pipeline includes an image quality and out of distribution screening module that assesses green foliar presence, blur variance, and illumination before running the neural network.

* Question 4: How can we trust your reported 95 percent accuracy?
  ⤷ Answer: All reported metrics represent holdout evaluation set benchmarks documented in our release manifest. Furthermore, our models incorporate post training temperature scaling to calibrate confidence scores.

## 4. Emergency Troubleshooting Protocol

* Issue: Backend fails to bind port 8000.
  ⤷ Remediation: Check if an existing process is using the port with `lsof -i :8000` and terminate it, or export `PORT=8001` before starting.

* Issue: Browser shows `Connecting to Backend...` continuously.
  ⤷ Remediation: Check that `python3 backend/run.py` is running and verify that CORS allows `http://localhost:8000`.

* Issue: GPU acceleration is unavailable.
  ⤷ Remediation: The backend automatically falls back to CPU execution without manual intervention.
