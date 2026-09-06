"""
test_integration.py : Comprehensive Automated Integration Test Suite
Smart Plant Intelligence System
Validates unified Crop Recommendation and 3-Tier Computer Vision inference.
"""

import io
import os
import pytest
from fastapi.testclient import TestClient
from main import app, vision_engine

client = TestClient(app)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")


def test_homepage_serves_unified_system():
    """Verify index.html serves both Crop Recommendation and Plant Vision sections."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "Smart Plant Intelligence System" in html
    assert 'id="cropSection"' in html
    assert 'id="visionSection"' in html
    assert 'id="btnModeCrop"' in html
    assert 'id="btnModeVision"' in html
    assert 'id="cvDropzone"' in html
    assert 'id="visionInspectionCanvas"' in html


def test_server_health():
    """Verify /health endpoint returns online status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "timestamp" in data
    assert data["system"] == "Smart Plant Intelligence System"


def test_models_status():
    """Verify /models/status endpoint returns all loaded models and hardware device."""
    response = client.get("/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["crop_recommendation"]["classes"] == 22
    assert data["computer_vision"]["is_loaded"] is True
    assert data["computer_vision"]["classes"] == 38
    assert len(data["computer_vision"]["tiers"]) == 3
    assert "device" in data["computer_vision"]


def test_crop_recommendation_prediction():
    """Verify original Crop Recommendation capability functions correctly."""
    payload = {
        "temperature": 28.5,
        "humidity": 72.0,
        "soil_moisture": 65.0,
        "rain": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_crop" in data
    assert "confidence" in data
    assert "top3" in data
    assert len(data["top3"]) == 3
    assert "disease_alerts" in data
    assert "features" in data
    assert data["features"]["temperature"] == 28.5
    assert data["features"]["rain"] == 1
    assert "fungal_risk" in data["features"]


def test_vision_early_blight_diagnosis():
    """Verify real inference on Early Blight tomato foliage."""
    sample_path = os.path.join(SAMPLE_DIR, "tomato__fungal__early_blight.jpg")
    assert os.path.exists(sample_path), f"Sample missing: {sample_path}"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/vision",
        files={"file": ("tomato_early_blight.jpg", file_bytes, "image/jpeg")},
        data={"use_yolo": "true", "use_seg": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    diag = data["diagnosis"]
    assert diag["crop"].lower() == "tomato"
    assert "Early_blight" in diag["predicted_class"]
    assert diag["is_infected"] is True
    assert diag["confidence_pct"] >= 80.0
    assert len(diag["top3_predictions"]) == 3

    telemetry = data["spatial_telemetry"]
    assert len(telemetry["bounding_boxes"]) > 0
    box = telemetry["bounding_boxes"][0]
    assert "bbox_xyxy" in box
    assert len(box["bbox_xyxy"]) == 4

    advisory = data["advisory"]
    assert "treatment" in advisory
    assert "immediate" in advisory
    assert "cultural" in advisory
    assert len(advisory["treatment"]) > 0

    lat = data["latency_ms"]
    assert lat["total_ms"] > 0
    assert "device" in lat


def test_vision_healthy_foliage_fast_gating():
    """Verify real inference and fast gating on healthy leaf."""
    sample_path = os.path.join(SAMPLE_DIR, "potato__healthy__healthy.jpg")
    assert os.path.exists(sample_path), f"Sample missing: {sample_path}"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/vision",
        files={"file": ("potato_healthy.jpg", file_bytes, "image/jpeg")},
        data={"use_yolo": "true", "use_seg": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    diag = data["diagnosis"]
    assert diag["is_infected"] is False
    assert "STAGE 0" in diag["triage_stage"]
    assert diag["foliar_damage_pct"] == 0.0


def test_vision_invalid_mime_type():
    """Verify backend validates file mime-type and returns HTTP 400."""
    fake_txt = b"This is a plain text file, not a plant leaf image.\n" * 100
    response = client.post(
        "/predict/vision",
        files={"file": ("notes.txt", fake_txt, "text/plain")},
        data={"use_yolo": "true", "use_seg": "true"}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_vision_corrupted_image_bytes():
    """Verify backend validates corrupted image bytes and returns HTTP 400."""
    corrupted_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_payload_garbage" * 200
    response = client.post(
        "/predict/vision",
        files={"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")},
        data={"use_yolo": "true", "use_seg": "true"}
    )
    assert response.status_code == 400
    assert "Corrupt or invalid image file" in response.json()["detail"]


def test_vision_field_tomato_healthy_gating():
    """Verify field tomato leaf with natural variation is correctly recognized as Healthy."""
    sample_path = os.path.join(SAMPLE_DIR, "user_healthy_tomato.jpg")
    assert os.path.exists(sample_path), f"Sample missing: {sample_path}"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/vision",
        files={"file": ("user_healthy_tomato.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    diag = data["diagnosis"]
    assert diag["is_infected"] is False
    assert "healthy" in diag["condition_type"].lower()
    assert diag["foliar_damage_pct"] == 0.0
    assert diag["lesion_foci_count"] == 0
    assert "STAGE 0" in diag["triage_stage"]
    assert len(data["spatial_telemetry"]["bounding_boxes"]) == 0


def test_esp8266_compact_prediction():
    """Verify constrained edge microcontroller /predict/compact endpoint response schema."""
    payload = {
        "temperature": 26.0,
        "humidity": 65.0,
        "soil_moisture": 55.0,
        "rain": 0
    }
    response = client.post("/predict/compact", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "crop" in data
    assert "conf" in data
    assert "t2" in data
    assert "c2" in data
    assert "t3" in data
    assert "c3" in data
    assert "ac" in data
    assert "alerts" in data
    assert isinstance(data["conf"], (int, float))

