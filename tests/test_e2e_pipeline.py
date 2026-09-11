"""
End-to-End Integration Smoke Test for SmartCropVision.
Validates the full request-response lifecycle from real leaf photograph upload
to structured contract consumption identical to the frontend client.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

SAMPLE_APPLE_SCAB = Path("frontend/samples/apple__fungal__apple_scab.jpg")
SAMPLE_CORN_RUST = Path("frontend/samples/corn__fungal__common_rust_.jpg")


def test_e2e_real_sample_fast_inference():
    """E2E smoke test verifying fast production path with a real leaf specimen."""
    assert SAMPLE_APPLE_SCAB.exists(), f"Sample image missing at {SAMPLE_APPLE_SCAB}"

    with open(SAMPLE_APPLE_SCAB, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/predict/vision",
        files={"file": ("apple__fungal__apple_scab.jpg", img_bytes, "image/jpeg")},
        data={"model_tier": "server", "include_explainability": "false"}
    )
    assert res.status_code == 200
    data = res.json()

    # Core response contract
    assert data["status"] == "success"
    assert "request_id" in data
    assert data["filename"] == "apple__fungal__apple_scab.jpg"

    # Diagnosis contract
    diag = data["diagnosis"]
    assert diag["crop"] == "Apple"
    assert diag["predicted_class"] == "Apple___Apple_scab"
    assert "confidence_pct" in diag
    assert "confidence_level" in diag
    assert "is_low_confidence" in diag
    assert "uncertainty_score" in diag
    assert "entropy" in diag
    assert "top3_predictions" in diag
    assert len(diag["top3_predictions"]) == 3
    assert diag["segmentation_status"] in ["available", "unavailable", "healthy_not_applicable"]

    # Spatial telemetry contract
    spatial = data["spatial_telemetry"]
    assert "detection_engine" in spatial
    assert isinstance(spatial["bounding_boxes"], list)
    # Zero fake boxes: verified bounding box list
    for box in spatial["bounding_boxes"]:
        assert "bbox_xyxy" in box
        assert len(box["bbox_xyxy"]) == 4
        assert "confidence" in box
        assert "centroid_norm" in box

    # Advisory contract
    adv = data["advisory"]
    assert "immediate_action" in adv
    assert "treatment_protocol" in adv
    assert "cultural_practices" in adv

    # Latency contract
    latency = data["latency_ms"]
    assert latency["tier1_ms"] > 0
    assert latency["total_ms"] > 0
    assert "device" in latency

    # Fast path: explainability should be None
    assert data["explainability"] is None


def test_e2e_real_sample_explainability_inference():
    """E2E smoke test verifying on-demand 9-stage explainability suite."""
    assert SAMPLE_APPLE_SCAB.exists()

    with open(SAMPLE_APPLE_SCAB, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/predict/vision/explain",
        files={"file": ("apple__fungal__apple_scab.jpg", img_bytes, "image/jpeg")},
        data={"model_tier": "server"}
    )
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    explain = data["explainability"]
    assert explain is not None
    assert "stages" in explain
    assert len(explain["stages"]) == 9

    for stage in explain["stages"]:
        assert "stage_number" in stage
        assert "title" in stage
        assert "technical_name" in stage
        assert "explanation" in stage
        assert "metrics" in stage


def test_e2e_frontend_status_consumption():
    """E2E test verifying frontend health and model counters receive expected telemetry."""
    res_status = client.get("/models/status")
    assert res_status.status_code == 200
    status_data = res_status.json()

    assert status_data["status"] == "ready"
    assert status_data["total_models"] >= 5
    assert status_data["models_ready"] >= 4
    assert status_data["taxonomy_classes"] == 38
    assert "device" in status_data
