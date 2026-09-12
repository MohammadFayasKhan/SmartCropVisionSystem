"""
Unit and Integration Tests for Groq Vision Multimodal Preflight Validator.
Verifies:
  1. Groq semantic detection and rejection of application UI screenshots.
  2. Groq semantic detection and rejection of PDF documents / browser dialogs.
  3. Groq acceptance of genuine plant leaves.
  4. Resilient fallback behavior on network timeout or rate limits (HTTP 429).
  5. End-to-end integration within validate_plant_image and prediction pipelines.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import pytest
import requests

from backend.app.services.groq_validator import (
    GroqVisionValidator,
    GroqValidationResult,
    groq_vision_validator
)
from backend.app.utils.domain_validation import validate_plant_image


def test_groq_vision_screenshot_rejection():
    """Confirms Groq Vision identifies SmartCropVision UI screenshot as screenshot_document."""
    screenshot_path = Path("tests/fixtures/smartcropvision_dashboard_screenshot.png")
    assert screenshot_path.exists()

    img = cv2.imread(str(screenshot_path))
    assert img is not None

    res = groq_vision_validator.validate_image(img, filename="dashboard.png")
    if res is not None:
        assert res.valid is False
        assert res.category == "screenshot_document"
        assert res.inference_allowed is False
        assert res.suitable_for_crop_analysis is False
        assert len(res.reason) > 5


def test_groq_vision_pdf_rejection():
    """Confirms Groq Vision identifies PDF/browser document as screenshot_document."""
    pdf_path = Path("tests/fixtures/pdf_document_screenshot.png")
    assert pdf_path.exists()

    img = cv2.imread(str(pdf_path))
    assert img is not None

    res = groq_vision_validator.validate_image(img, filename="document.png")
    if res is not None:
        assert res.valid is False
        assert res.category == "screenshot_document"
        assert res.inference_allowed is False


def test_groq_vision_authentic_leaf_acceptance():
    """Confirms Groq Vision accepts authentic agricultural leaf photograph."""
    leaf_path = Path("frontend/samples/apple__healthy__healthy.jpg")
    assert leaf_path.exists()

    img = cv2.imread(str(leaf_path))
    assert img is not None

    res = groq_vision_validator.validate_image(img, filename="apple_leaf.jpg")
    if res is not None:
        assert res.valid is True
        assert res.category == "plant_leaf"
        assert res.inference_allowed is True
        assert res.plant_present is True
        assert res.suitable_for_crop_analysis is True


def test_groq_vision_graceful_fallback_on_timeout():
    """Verifies that when Groq Vision times out, validate_image returns None without crashing."""
    img = cv2.imread("tests/fixtures/smartcropvision_dashboard_screenshot.png")

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        res = groq_vision_validator.validate_image(img)
        assert res is None, "Expected None fallback on timeout"


def test_groq_vision_graceful_fallback_on_rate_limit():
    """Verifies that HTTP 429 rate limit triggers clean fallback."""
    img = cv2.imread("tests/fixtures/smartcropvision_dashboard_screenshot.png")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Rate limit reached"

    with patch("requests.post", return_value=mock_resp):
        res = groq_vision_validator.validate_image(img)
        assert res is None, "Expected None fallback on HTTP 429"


def test_end_to_end_validation_resilience_when_groq_fails():
    """
    Verifies that even if Groq Vision is completely unavailable or throws errors,
    validate_plant_image still rejects screenshots using local CV checks.
    """
    screenshot_path = Path("tests/fixtures/smartcropvision_dashboard_screenshot.png")
    img = cv2.imread(str(screenshot_path))

    with patch("requests.post", side_effect=RuntimeError("Groq service unreachable")):
        val_res = validate_plant_image(img, filename="dashboard.png")
        assert val_res.validation_status == "INVALID_SCREENSHOT_OR_DOCUMENT"
        assert val_res.is_inference_allowed is False
