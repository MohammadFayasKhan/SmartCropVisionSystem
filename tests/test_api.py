"""
Automated Test Suite for Smart Plant Intelligence FastAPI Backend.
Verifies health checks, model status auditing, image validation gates,
and end-to-end 3-tier inference on field leaf samples.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)

def get_test_sample_images():
    """Finds authentic test sample images from cv/test_images, frontend/samples, or test fixtures."""
    candidates = list(Path("cv/test_images").glob("*.jpg"))
    if not candidates:
        candidates = list(Path("frontend/samples").glob("*.jpg"))
    if not candidates:
        candidates = list(Path("cv/datasets/processed/segmentation/images/val").glob("*.jpg"))
    return candidates

def test_root_index():
    """Verify root index redirects and provides navigation URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert "health_url" in data

def test_health_endpoint():
    """Verify system health liveness endpoint returns healthy status."""
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data

def test_models_status_endpoint():
    """Verify platform models status endpoint returns production models."""
    response = client.get(f"{settings.API_V1_STR}/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["total_models"] >= 6
    assert data["models_ready"] >= 4
    assert len(data["models"]) >= 6
    model_names = [m["name"] for m in data["models"]]
    assert any("EfficientNet" in n for n in model_names)
    assert any("PlantDoc" in n for n in model_names)

def test_validation_rejects_empty_file():
    """Verify validation gate rejects empty or zero-byte uploads."""
    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "IMAGE_VALIDATION_FAILED"

def test_validation_rejects_invalid_mime_and_magic_bytes():
    """Verify validation gate rejects non-image text files."""
    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": ("document.txt", b"This is plain text and not a plant photograph.", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "IMAGE_VALIDATION_FAILED"

def test_end_to_end_leaf_diagnosis():
    """Verify end-to-end inference on an authentic validation leaf image."""
    val_images = get_test_sample_images()
    assert len(val_images) > 0, "Validation images directory is empty!"
    
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "diagnosis" in data
    assert "crop" in data["diagnosis"]
    assert "confidence_pct" in data["diagnosis"]
    assert "triage_stage" in data["diagnosis"]
    assert "spatial_telemetry" in data
    assert "advisory" in data
    assert "performance_benchmark" in data
    assert data["performance_benchmark"]["total_pipeline_ms"] > 0

def test_crop_presets_endpoint():
    """Verify curated regional presets return properly formatted list."""
    response = client.get(f"{settings.API_V1_STR}/crops/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 4
    for preset in presets:
        assert "id" in preset
        assert "title" in preset
        assert "temperature" in preset
        assert "humidity" in preset
        assert "soil_moisture" in preset
        assert "rain" in preset

def test_crop_classes_endpoint():
    """Verify all 22 crop taxonomy classes are returned."""
    response = client.get(f"{settings.API_V1_STR}/crops/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 22
    assert "rice" in classes
    assert "apple" in classes

def test_crop_recommendation_inference():
    """Verify crop recommendation pipeline executes properly for tropical wetland conditions."""
    payload = {
        "temperature": 28.5,
        "humidity": 84.0,
        "soil_moisture": 80.0,
        "rain": 1,
        "rainfall_mm": 160.0
    }
    response = client.post(f"{settings.API_V1_STR}/crops/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "recommended_crop" in data
    assert data["confidence_pct"] > 0.0
    assert len(data["top3_candidates"]) == 3
    assert data["top3_candidates"][0]["rank"] == 1
    assert "features_used" in data
    assert data["features_used"]["rain"] == 1
    assert data["latency_ms"] > 0.0
    assert len(data["agronomic_summary"]) > 10

def test_crop_recommendation_validation_rejection():
    """Verify crop recommendation rejects physically impossible sensor values."""
    payload = {
        "temperature": 150.0,  # exceeds maximum bound of 60.0°C
        "humidity": 50.0,
        "soil_moisture": 40.0,
        "rain": 0
    }
    response = client.post(f"{settings.API_V1_STR}/crops/recommend", json=payload)
    assert response.status_code == 422  # Pydantic validation unprocessable entity

def test_vision_diagnose_server_grade():
    """Verify server-grade EfficientNet-B2 executes when model_tier='server'."""
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "server"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["diagnosis"]["model_tier"] == "server"
    assert "EfficientNetV2-S" in data["diagnosis"]["model_architecture"]

def test_vision_diagnose_edge_model():
    """Verify edge MobileNetV2 executes when model_tier='edge'."""
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "edge"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Production enforces server-grade EfficientNetV2-S classifier across all requests
    assert "EfficientNetV2-S" in data["diagnosis"]["model_architecture"]

def test_vision_diagnose_ensemble_mode():
    """Verify ensemble request routes to authoritative production classifier."""
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "ensemble"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "EfficientNetV2-S" in data["diagnosis"]["model_architecture"]

def test_root_predict_vision_server_grade():
    """Verify root /predict/vision endpoint supports server-grade model_tier."""
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/vision",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "server"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["diagnosis"]["model_tier"] == "server"
    assert "EfficientNetV2-S" in data["diagnosis"]["model_architecture"]


def test_multimodal_diagnosis_and_traceability():
    """Verify diagnosis endpoint handles paired multimodal context and enforces modality traceability."""
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    # Case 1: Purely visual inference (no multimodal context)
    res_visual = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "server"}
    )
    assert res_visual.status_code == 200
    data_vis = res_visual.json()
    assert data_vis["modalities_used"] == ["image"]
    assert data_vis["multimodal_context"] is None

    # Case 2: Multimodal inference with paired agronomic and environmental context
    res_multi = client.post(
        f"{settings.API_V1_STR}/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={
            "model_tier": "server",
            "crop_context": "Tomato",
            "temperature_c": 27.5,
            "humidity_pct": 80.0,
            "symptom_notes": "Small circular lesions on lower canopy foliage"
        }
    )
    assert res_multi.status_code == 200
    data_multi = res_multi.json()
    assert "image" in data_multi["modalities_used"]
    assert "environmental_context" in data_multi["modalities_used"]
    assert data_multi["multimodal_context"]["crop_context"] == "Tomato"
    assert data_multi["multimodal_context"]["temperature_c"] == 27.5
    assert data_multi["multimodal_context"]["humidity_pct"] == 80.0


