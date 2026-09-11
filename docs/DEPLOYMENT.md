# SmartCropVision Deployment & Operations Guide

## 1. Quick Start Local Development

### 1.1 Prerequisites
* Python 3.10, 3.11, or 3.12
* PyTorch 2.0 or higher
* Node.js (optional, for frontend testing)

### 1.2 Installation
```bash
# 1. Clone or extract repository
cd SmartCropVision

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install production dependencies
pip install -r requirements.txt
```

### 1.3 Launching the Backend
```bash
# Start FastAPI backend server
python3 backend/run.py
```
The server will initialize the model registry, warm up GPU or CPU kernels, and bind to `http://0.0.0.0:8000`.

### 1.4 Opening the Web Client
The frontend is a vanilla single page application requiring zero compilation steps:
```bash
# Option A: Open directly in your browser
open frontend/index.html

# Option B: Serve through a local HTTP server
python3 -m http.server 3000 --directory frontend
```
Navigate to `http://localhost:3000` to interact with the diagnostic dashboard.

---

## 2. Docker Container Deployment

The provided `Dockerfile` builds a production container running as an unprivileged system user.

### 2.1 Building the Container
```bash
docker build -t smartcropvision:v2.2 .
```

### 2.2 Running with Docker Compose
```bash
docker compose up -d
```
The container exposes port 8000 and includes automatic healthcheck polling at `/health/live`.

---

## 3. Air Gapped & Kaggle Deployment

For deployments on air gapped edge servers or Kaggle kernels where external internet connectivity is restricted, SmartCropVision bundles offline wheel packages in `packages/`:

```bash
# Install required libraries 100% offline
pip install --no-index --find-links packages/ \
    ultralytics \
    albumentations \
    albucore \
    nvidia_ml_py \
    ultralytics_thop
```

---

## 4. Operational Verification Checklist

### 4.1 Confirm Models Remain Server Side
To verify that large weight files are never transmitted to user browsers:
1. Open your browser Developer Tools (`F12`) and select the Network tab.
2. Upload a leaf sample and click Analyze Leaf Specimen.
3. Verify that the only outbound network request is a POST to `/api/v1/vision/diagnose` transmitting the image bytes.
4. Verify that zero requests attempt to fetch `.pt`, `.pth`, or `.pkl` weight files.

### 4.2 Confirm Loaded Model Versions
Query the live registry endpoint:
```bash
curl -s http://localhost:8000/models/status | python3 -m json.tool
```
Confirm that `models_ready` matches expected production checkpoints and that the active compute device is truthfully identified (MPS, CUDA, or CPU).

---

## 5. Rollback Procedure

If a deployed checkpoint exhibits an unexpected anomaly:
1. Identify the previous stable release tag from `RELEASE_MANIFEST.json`.
2. Verify the SHA 256 signature of the backup checkpoint in `cv/models/`.
3. Update `SERVER_TIER1_MODEL_PATH` in `.env` to point to the verified checkpoint.
4. Restart the backend service. The startup preflight will validate the signature before declaring readiness.
