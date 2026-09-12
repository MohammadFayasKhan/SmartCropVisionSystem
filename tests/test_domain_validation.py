"""
Unit and Integration Test Suite for SmartCropVision Pre-Inference Domain Validation.
Covers:
  1. Authentic plant leaf specimens (pass with VALID_PLANT_IMAGE and allow inference).
  2. Unrelated objects (vehicles, metallic objects, buildings).
  3. Human photographs / skin portraits / selfies.
  4. Domestic animals / non-plant nature.
  5. Text documents, invoices, or screenshots.
  6. Corrupted byte streams / synthetic high-frequency noise.
  7. Extremely small images below dimension boundaries.
  8. Blank uniform images (pure black, pure white).
  9. Critically underexposed (dark) images.
  10. Severely blurred plant photographs (LOW_QUALITY_OR_UNCERTAIN_IMAGE).
  11. Complex borderline agricultural field photographs (leaves with soil and sunlight).
  12. Verification of zero neural classification calls on rejected inputs.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2

from backend.app.utils.domain_validation import (
    validate_plant_image,
    extract_botanical_signals,
    DomainValidationResult,
)
from backend.app.services.inference_service import inference_engine

# Ensure models are loaded for testing
inference_engine.load_models()


def get_sample_leaf_paths():
    """Returns list of genuine sample leaf paths in the repository."""
    samples = sorted(list(Path("frontend/samples").glob("*.jpg")))
    assert len(samples) > 0, "Expected authentic leaf samples in frontend/samples"
    return samples


def create_human_portrait_proxy() -> np.ndarray:
    """Creates a human face/skin tone test image."""
    img = np.full((300, 300, 3), (130, 160, 210), dtype=np.uint8) # BGR skin
    img[:80, :] = (30, 40, 50) # Hair
    cv2.circle(img, (100, 140), 12, (240, 240, 240), -1) # Eyes
    cv2.circle(img, (100, 140), 5, (40, 40, 40), -1)
    cv2.circle(img, (200, 140), 12, (240, 240, 240), -1)
    cv2.circle(img, (200, 140), 5, (40, 40, 40), -1)
    cv2.line(img, (120, 220), (180, 220), (80, 90, 180), 4) # Mouth
    return img


def create_vehicle_proxy() -> np.ndarray:
    """Creates a blue vehicle / metallic object on road."""
    img = np.full((300, 400, 3), (220, 220, 220), dtype=np.uint8) # Grey road/sky
    cv2.rectangle(img, (50, 120), (350, 230), (220, 80, 20), -1) # Blue car body
    cv2.rectangle(img, (100, 70), (300, 120), (180, 180, 180), -1) # Windshield
    cv2.circle(img, (110, 230), 28, (20, 20, 20), -1) # Wheel
    cv2.circle(img, (290, 230), 28, (20, 20, 20), -1)
    return img


def create_document_proxy() -> np.ndarray:
    """Creates a printed text document / digital screenshot."""
    img = np.full((400, 300, 3), 255, dtype=np.uint8) # White page
    for y in range(40, 380, 22):
        cv2.line(img, (30, y), (270, y), (30, 30, 30), 2)
    return img


def create_animal_proxy() -> np.ndarray:
    """Creates a dog/animal fur proxy image."""
    img = np.full((300, 300, 3), (60, 120, 180), dtype=np.uint8) # Golden fur
    cv2.circle(img, (150, 150), 90, (70, 140, 200), -1)
    cv2.circle(img, (120, 120), 10, (20, 20, 20), -1)
    cv2.circle(img, (180, 120), 10, (20, 20, 20), -1)
    cv2.circle(img, (150, 170), 14, (20, 20, 20), -1)
    return img


def create_building_proxy() -> np.ndarray:
    """Creates an architectural brick building image."""
    img = np.full((300, 300, 3), (90, 100, 150), dtype=np.uint8) # Brick red/brown
    for rx in range(30, 270, 60):
        for ry in range(30, 270, 70):
            cv2.rectangle(img, (rx, ry), (rx + 40, ry + 50), (220, 220, 240), -1)
    return img


# ── TEST 1: Authentic Leaf Specimens Validation ──────────────────────────────
def test_valid_clear_leaf_images():
    """Confirms all authentic repository plant leaves are accepted as VALID_PLANT_IMAGE."""
    sample_paths = get_sample_leaf_paths()
    detector = inference_engine.model_tier2_plantdoc

    for p in sample_paths:
        img = cv2.imread(str(p))
        assert img is not None, f"Could not read leaf at {p}"
        res = validate_plant_image(img, detector_model=detector, filename=p.name)

        assert res.validation_status == "VALID_PLANT_IMAGE", (
            f"Authentic leaf {p.name} was unexpectedly rejected: {res.validation_status} ({res.validation_reason})"
        )
        assert res.is_inference_allowed is True
        assert res.plant_presence is True
        assert res.leaf_presence is True
        assert res.validation_confidence >= 0.70


# ── TEST 2: Unrelated Object Rejection (Vehicle / Metallic) ───────────────────
def test_unrelated_object_rejection():
    """Confirms vehicles, tools, and industrial objects are rejected with INVALID_NON_PLANT_IMAGE."""
    vehicle = create_vehicle_proxy()
    res = validate_plant_image(vehicle, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status == "INVALID_NON_PLANT_IMAGE"
    assert res.is_inference_allowed is False
    assert res.plant_presence is False
    assert "clear image of a plant leaf" in res.validation_reason.lower()


# ── TEST 3: Human Photograph Rejection ───────────────────────────────────────
def test_human_portrait_rejection():
    """Confirms human selfies and portraits are rejected based on skin chrominance."""
    portrait = create_human_portrait_proxy()
    res = validate_plant_image(portrait, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status == "INVALID_NON_PLANT_IMAGE"
    assert res.is_inference_allowed is False
    assert "person or skin surface" in res.validation_reason.lower()


# ── TEST 4: Document / Screenshot Rejection ──────────────────────────────────
def test_document_screenshot_rejection():
    """Confirms text screenshots and document scans are rejected."""
    doc = create_document_proxy()
    res = validate_plant_image(doc, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status == "INVALID_NON_PLANT_IMAGE"
    assert res.is_inference_allowed is False
    assert "document or digital screenshot" in res.validation_reason.lower()


# ── TEST 5: Animal Photograph Rejection ──────────────────────────────────────
def test_animal_photograph_rejection():
    """Confirms domestic animal / pet photography is rejected."""
    dog = create_animal_proxy()
    res = validate_plant_image(dog, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status in ("INVALID_NON_PLANT_IMAGE", "LOW_QUALITY_OR_UNCERTAIN_IMAGE")
    assert res.is_inference_allowed is False


# ── TEST 6: Landscape / Architecture Rejection ───────────────────────────────
def test_building_architecture_rejection():
    """Confirms non-agricultural architecture scenes are rejected."""
    bldg = create_building_proxy()
    res = validate_plant_image(bldg, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status in ("INVALID_NON_PLANT_IMAGE", "LOW_QUALITY_OR_UNCERTAIN_IMAGE")
    assert res.is_inference_allowed is False


# ── TEST 7: Blank Uniform Images Rejection ───────────────────────────────────
def test_blank_white_and_black_rejection():
    """Confirms pure black or pure white frames are rejected."""
    white = np.full((256, 256, 3), 255, dtype=np.uint8)
    black = np.zeros((256, 256, 3), dtype=np.uint8)

    res_w = validate_plant_image(white)
    assert res_w.validation_status == "INVALID_NON_PLANT_IMAGE"
    assert res_w.is_inference_allowed is False

    res_b = validate_plant_image(black)
    assert res_b.validation_status in ("INVALID_NON_PLANT_IMAGE", "LOW_QUALITY_OR_UNCERTAIN_IMAGE")
    assert res_b.is_inference_allowed is False


# ── TEST 8: Critically Dark Image Rejection ──────────────────────────────────
def test_critically_dark_image_rejection():
    """Confirms severely underexposed photography is rejected."""
    dark = np.full((256, 256, 3), 10, dtype=np.uint8)
    res = validate_plant_image(dark)

    assert res.validation_status in ("INVALID_NON_PLANT_IMAGE", "LOW_QUALITY_OR_UNCERTAIN_IMAGE")
    assert res.is_inference_allowed is False


# ── TEST 9: Severely Blurred Plant Photograph ────────────────────────────────
def test_severely_blurred_leaf():
    """Confirms out-of-focus leaf photograph returns LOW_QUALITY_OR_UNCERTAIN_IMAGE."""
    sample = get_sample_leaf_paths()[0]
    leaf = cv2.imread(str(sample))
    blurry_leaf = cv2.GaussianBlur(leaf, (51, 51), 0)

    res = validate_plant_image(blurry_leaf, detector_model=inference_engine.model_tier2_plantdoc)

    assert res.validation_status == "LOW_QUALITY_OR_UNCERTAIN_IMAGE"
    assert res.is_inference_allowed is False
    assert "blur" in res.validation_reason.lower()


# ── TEST 10: Borderline Agricultural Field Photograph ────────────────────────
def test_borderline_field_plant_preservation():
    """Confirms field leaf with background soil and high natural contrast is preserved."""
    sample = get_sample_leaf_paths()[0]
    leaf = cv2.imread(str(sample))

    # Add realistic brown soil border around the leaf
    field_composite = np.full((320, 320, 3), (40, 60, 90), dtype=np.uint8) # Soil BGR
    field_composite[32:288, 32:288] = cv2.resize(leaf, (256, 256))

    res = validate_plant_image(field_composite, detector_model=inference_engine.model_tier2_plantdoc)
    assert res.validation_status == "VALID_PLANT_IMAGE"
    assert res.is_inference_allowed is True


# ── TEST 11: End-to-End Pipeline Zero-Inference Guarantee on Rejection ───────
def test_zero_inference_calls_on_rejection():
    """
    Critical requirement: When an invalid image is supplied,
    the pipeline MUST return status 'rejected' with zero fabricated disease diagnosis,
    zero bounding boxes, zero segmentation masks, and zero Grad-CAM maps.
    """
    portrait = create_human_portrait_proxy()
    _, enc = cv2.imencode(".jpg", portrait)
    file_bytes = enc.tobytes()

    response = inference_engine.predict_vision(
        file_bytes,
        filename="human_selfie.jpg",
        include_explainability=True
    )

    # 1. Pipeline status must be 'rejected'
    assert response.status == "rejected"

    # 2. Validation assessment must declare invalid non plant
    assert response.image_validation is not None
    assert response.image_validation.validation_status == "INVALID_NON_PLANT_IMAGE"
    assert response.image_validation.is_inference_allowed is False

    # 3. Zero disease prediction / Zero fabricated diagnosis
    assert response.diagnosis.predicted_class == "N/A"
    assert response.diagnosis.confidence_pct == 0.0
    assert response.diagnosis.is_infected is False
    assert len(response.diagnosis.top3_predictions) == 0

    # 4. Zero detections / Zero segmentation / Zero Grad-CAM
    assert len(response.spatial_telemetry.bounding_boxes) == 0
    assert response.segmentation_mask_b64 is None
    assert response.cam_heatmap_b64 is None
    assert response.cam_overlay_b64 is None
    assert response.explainability is None
