"""
main.py
-------
FastAPI backend for the Smart Plant Intelligence System.

Endpoints:
    GET  /              → serves the dashboard (index.html)
    POST /predict       → runs the full ML pipeline and returns JSON
    POST /predict/compact → compact response for ESP8266 LCD display
    GET  /latest        → most recent IoT reading (polled by dashboard)
    GET  /history       → last N sensor readings + predictions
    GET  /health        → server health check

Deployment:
    Local:              uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    Hugging Face Spaces: Docker exposes port 7860 (see Dockerfile)
    ESP8266 posts to:   https://<user>-<space>.hf.space/predict/compact

Rain sensor note:
    The YL-83 rain sensor outputs a binary digital signal only.
    The 'rain' field in all requests is 1 (raining) or 0 (dry).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
import os

from predict import predict_from_sensors
from vision_engine import vision_engine

# ── app lifespan for pre-caching models ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Pre-caching Computer Vision models at startup...")
    try:
        vision_engine.load_models()
    except Exception as e:
        print(f"[!] Warning: Vision models failed to preload: {e}")
    yield
    print("[*] Shutting down Smart Plant Intelligence API service.")

# ── app setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Plant Intelligence System",
    description="Unified IoT-driven crop recommendation and computer vision plant disease detection API",
    version="2.2.0",
    lifespan=lifespan
)

# ── CORS : required so ESP8266 and external browsers can call the API ──
# When deployed on HF Spaces, the Space URL is the only origin, but
# the ESP8266 has no browser Origin header, so allow_origins=["*"] is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory reading history (last 50 readings)
history: deque = deque(maxlen=50)

# Latest reading from a physical IoT device : None until first reading arrives
latest_iot: dict = None


# : request model :::::::::::::::::::::::::::::::::::::::::::::::::::::::
class SensorReading(BaseModel):
    temperature   : float = Field(..., ge=-10, le=60,  description="°C from DHT11")
    humidity      : float = Field(..., ge=0,   le=100, description="% from DHT11")
    soil_moisture : float = Field(..., ge=0,   le=100, description="% from soil sensor")
    rain          : int   = Field(..., ge=0,   le=1,   description="1=rain detected, 0=no rain (YL-83 binary)")


# ── routes ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(index_path)


@app.get("/health")
def health_check():
    return {
        "status"   : "online",
        "system"   : "Smart Plant Intelligence System",
        "version"  : "2.2.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
def predict(reading: SensorReading):
    """
    Full ML pipeline on incoming sensor readings.
    Returns crop recommendation, top 3, disease alerts and risk flags.
    """
    try:
        result = predict_from_sensors(
            temperature   = reading.temperature,
            humidity      = reading.humidity,
            soil_moisture = reading.soil_moisture,
            rain          = reading.rain,
        )

        alerts = [
            {
                "id"       : d.get("id", "unknown"),
                "name"     : d["name"],
                "type"     : d["type"],
                "severity" : d["severity"],
                "trigger"  : d["trigger"],
                "symptoms" : d["symptoms"],
                "pesticide": d["pesticide"],
                "technique": d["technique"],
            }
            for d in result["disease_alerts"]
        ]

        response = {
            "timestamp"       : datetime.now().isoformat(),
            "recommended_crop": result["recommended_crop"],
            "confidence"      : result["confidence"],
            "top3"            : result["top3"],
            "disease_alerts"  : alerts,
            "alert_count"     : len(alerts),
            "features"        : result["features_used"],
            "status"          : "success",
        }

        history.append({
            "timestamp"       : response["timestamp"],
            "source"          : "dashboard",
            "temperature"     : reading.temperature,
            "humidity"        : reading.humidity,
            "soil_moisture"   : reading.soil_moisture,
            "rain"            : reading.rain,
            "recommended_crop": result["recommended_crop"],
            "confidence"      : result["confidence"],
            "alert_count"     : len(alerts),
        })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/compact")
def predict_compact(reading: SensorReading):
    """
    Compact prediction endpoint for ESP8266 / microcontrollers.
    Keeps response under 512 bytes to fit in the ESP8266 JSON buffer.
    """
    try:
        result = predict_from_sensors(
            temperature   = reading.temperature,
            humidity      = reading.humidity,
            soil_moisture = reading.soil_moisture,
            rain          = reading.rain,
        )

        top3   = result["top3"]
        alerts = result["disease_alerts"]

        iot_entry = {
            "timestamp"       : datetime.now().isoformat(),
            "source"          : "iot",
            "temperature"     : reading.temperature,
            "humidity"        : reading.humidity,
            "soil_moisture"   : reading.soil_moisture,
            "rain"            : reading.rain,
            "recommended_crop": result["recommended_crop"],
            "confidence"      : result["confidence"],
            "alert_count"     : len(alerts),
        }
        history.append(iot_entry)

        global latest_iot
        latest_iot = iot_entry

        return {
            "ok"  : 1,
            "crop": result["recommended_crop"][:14],
            "conf": result["confidence"],
            "ac"  : len(alerts),
            "t2"  : top3[1]["crop"][:12] if len(top3) > 1 else "",
            "c2"  : top3[1]["confidence"] if len(top3) > 1 else 0,
            "t3"  : top3[2]["crop"][:12] if len(top3) > 2 else "",
            "c3"  : top3[2]["confidence"] if len(top3) > 2 else 0,
            "alerts": [
                {"n": d["name"][:18], "s": d["severity"]}
                for d in alerts[:4]
            ],
        }

    except Exception as e:
        return {"ok": 0, "err": str(e)[:40]}


@app.get("/latest")
def get_latest():
    """
    Returns the most recent reading sent by the ESP8266.
    Dashboard polls this every 10 seconds for live sensor feed.
    Returns 404 if no IoT device has reported yet.
    """
    if latest_iot is None:
        raise HTTPException(
            status_code=404,
            detail="No IoT reading received yet. Is the ESP8266 running?"
        )
    return latest_iot


@app.get("/history")
def get_history(limit: int = 20):
    """Return the last N sensor readings and their predictions."""
    items = list(history)[-limit:]
    items.reverse()
    return {"count": len(items), "readings": items}


@app.post("/predict/vision")
async def predict_vision(file: UploadFile = File(...)):
    """
    Computer Vision Plant Health Inference Endpoint.
    Receives leaf photograph and executes the 3-Tier cascade:
    Tier 1: MobileNetV2 universal crop disease screening
    Tier 2: YOLOv8-nano spatial necrotic lesion localization
    Tier 3: Mobile-UNet sub-pixel pathology segmentation
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file received.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG, PNG, and WebP are supported."
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    try:
        result = vision_engine.diagnose(image_bytes)
        result["filename"] = file.filename
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision inference error: {str(e)}")


@app.get("/models/status")
def models_status():
    """Returns runtime status of ML inference engines."""
    if not vision_engine.is_loaded:
        vision_engine.load_models()

    return {
        "status": "online",
        "crop_recommendation": {
            "model": "RandomForestClassifier",
            "classes": 22
        },
        "computer_vision": {
            "is_loaded": vision_engine.is_loaded,
            "device": vision_engine.device_name,
            "classes": len(vision_engine.taxonomy),
            "tiers": [
                "Tier 1: MobileNetV2 Universal Classifier",
                "Tier 2: YOLOv8-nano Spatial Lesion Detector",
                "Tier 3: Mobile-UNet Sub-Pixel Segmenter"
            ]
        }
    }
