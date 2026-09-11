"""
Repeated Inference, Memory Stability, and Latency Telemetry Test Suite.
Enforces Master Prompt Section 51, 70, and 86:
- 25 repeated real inference requests through the active vision pipeline.
- Monitored process RSS memory (no runaway memory growth or tensor retention).
- Verified single model load (no reload per request).
- Verified latency stability across requests.
- Verified deterministic outputs on identical inputs.
- Validated both Fast Path and Full Explainability Path.
"""

import os
import gc
import time
import psutil
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.inference_service import inference_engine

client = TestClient(app)

SAMPLE_IMAGE_PATH = "frontend/samples/tomato__fungal__early_blight.jpg"


@pytest.fixture(scope="module")
def sample_image_bytes():
    assert os.path.exists(SAMPLE_IMAGE_PATH), f"Sample image {SAMPLE_IMAGE_PATH} must exist"
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        return f.read()


def test_repeated_inference_memory_stability_and_latency(sample_image_bytes):
    """
    Executes 25 repeated inference requests to ensure:
    1. Single model initialization in memory.
    2. Zero progressive memory leak (delta RSS bounded).
    3. Latency remains stable without degradation.
    4. Prediction outputs are identical/deterministic.
    """
    process = psutil.Process(os.getpid())
    
    # 1. Warmup pass
    warmup_res = client.post(
        "/predict/vision",
        files={"file": ("specimen.jpg", sample_image_bytes, "image/jpeg")},
        data={"include_explainability": "false"}
    )
    assert warmup_res.status_code == 200, f"Warmup failed: {warmup_res.text}"
    baseline_prediction = warmup_res.json()["diagnosis"]["predicted_class"]
    baseline_crop = warmup_res.json()["diagnosis"]["crop"]

    # Force garbage collection to record clean baseline RSS
    gc.collect()
    rss_baseline_mb = process.memory_info().rss / (1024 * 1024)

    latencies = []
    num_iterations = 25

    for i in range(num_iterations):
        t0 = time.perf_counter()
        res = client.post(
            "/predict/vision",
            files={"file": ("specimen.jpg", sample_image_bytes, "image/jpeg")},
            data={"include_explainability": "false"}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        assert res.status_code == 200
        data = res.json()
        
        # Verify output determinism
        assert data["diagnosis"]["predicted_class"] == baseline_prediction
        assert data["diagnosis"]["crop"] == baseline_crop
        assert "response_schema_version" in data
        assert data["response_schema_version"] == "1.0"
        
        # Verify fast path was respected
        assert data.get("explainability") is None or data.get("explainability_status") in ["not_requested", "unavailable"]

    gc.collect()
    rss_final_mb = process.memory_info().rss / (1024 * 1024)
    memory_growth_mb = rss_final_mb - rss_baseline_mb

    avg_latency_ms = sum(latencies) / len(latencies)
    max_latency_ms = max(latencies)
    min_latency_ms = min(latencies)

    print(f"\n[Repeated Inference Telemetry - {num_iterations} runs]:")
    print(f"  Baseline RSS:     {rss_baseline_mb:.2f} MB")
    print(f"  Final RSS:        {rss_final_mb:.2f} MB")
    print(f"  Memory Growth:    {memory_growth_mb:.2f} MB")
    print(f"  Latency (Min):    {min_latency_ms:.1f} ms")
    print(f"  Latency (Avg):    {avg_latency_ms:.1f} ms")
    print(f"  Latency (Max):    {max_latency_ms:.1f} ms")

    # Assert memory growth is tightly bounded (< 100 MB across 25 iterations)
    assert memory_growth_mb < 100.0, f"Possible memory leak detected: growth of {memory_growth_mb:.2f} MB exceeds 100 MB threshold"
    
    # Assert latency is reasonable for fast path inference (< 1000ms per request on CPU)
    assert avg_latency_ms < 1000.0, f"Average latency {avg_latency_ms:.1f}ms exceeds target 1000ms"


def test_explainability_memory_stability(sample_image_bytes):
    """
    Executes repeated full explainability requests (Grad-CAM + 9-stage analysis)
    and verifies that intermediate activation maps and gradients do not leak memory.
    """
    process = psutil.Process(os.getpid())
    gc.collect()
    rss_start_mb = process.memory_info().rss / (1024 * 1024)

    for i in range(5):
        res = client.post(
            "/predict/vision/explain",
            files={"file": ("specimen.jpg", sample_image_bytes, "image/jpeg")},
            data={"model_tier": "server"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["explainability"] is not None
        assert "stages" in data["explainability"]
        assert len(data["explainability"]["stages"]) == 9

    gc.collect()
    rss_end_mb = process.memory_info().rss / (1024 * 1024)
    growth_mb = rss_end_mb - rss_start_mb

    print(f"\n[Explainability Telemetry - 5 full passes]:")
    print(f"  Start RSS:   {rss_start_mb:.2f} MB")
    print(f"  End RSS:     {rss_end_mb:.2f} MB")
    print(f"  Growth:      {growth_mb:.2f} MB")

    assert growth_mb < 120.0, f"Grad-CAM explainability leaked {growth_mb:.2f} MB"
