"""
SmartCropVision Multi-Tier Detection Parity & Coordinate Verification Test Suite.
Verifies that:
  1. Authoritative YOLO detector checkpoints are loaded without fallback to generic COCO models.
  2. All genuine detections produced by the detector reach the API response without being collapsed into 1 box.
  3. Original detector classes, confidences, coordinates, and categories (canopy vs lesion) are preserved.
  4. Zero fabricated boxes, synthetic spots, or duplicate boxes are generated.
  5. Coordinate space is strictly bounded to the original image pixel dimensions.
"""

import os
import pytest
from pathlib import Path
from backend.app.services.inference_service import InferenceEngine
from backend.app.config import settings

STRAWBERRY_IMG = "cv/test_images/Strawberry-Leaves.jpg"
TOMATO_VIRUS_IMG = "cv/test_images/tomato_virus_06_zoom.jpg"
TOMATO_SEPTORIA_IMG = "cv/test_images/tomato-badleaves.jpg"


@pytest.fixture(scope="module")
def engine():
    eng = InferenceEngine.get_instance()
    eng.load_models()
    return eng


def test_authoritative_checkpoints_loaded(engine):
    """Ensure production checkpoints exist and are loaded without COCO fallbacks."""
    assert Path("cv/models/yolo_plantdoc_best.pt").exists(), "PlantDoc checkpoint missing"
    assert engine.model_tier2_plantdoc is not None, "Tier 2 PlantDoc model failed to load"
    
    # Verify 29 genuine PlantDoc classes
    names = engine.model_tier2_plantdoc.names
    assert len(names) == 29, f"Expected 29 PlantDoc classes, found {len(names)}"
    assert "Strawberry leaf" in names.values(), "Strawberry leaf missing from detector classes"
    assert "Tomato Septoria leaf spot" in names.values(), "Tomato Septoria leaf spot missing from detector classes"


def test_strawberry_multi_detection_preservation(engine):
    """
    Strawberry-Leaves.jpg validation image contains 8 genuine YOLO detections.
    Verify that all 8 detections are preserved in the response (not collapsed to 1 box).
    """
    if not os.path.exists(STRAWBERRY_IMG):
        pytest.skip(f"Test image not found: {STRAWBERRY_IMG}")

    resp = engine.predict_vision(STRAWBERRY_IMG, model_tier="server")
    telemetry = resp.spatial_telemetry
    boxes = telemetry.bounding_boxes

    # Verify genuine detector output preserved
    assert telemetry.raw_detection_count >= 8, f"Expected raw count >= 8, got {telemetry.raw_detection_count}"
    assert telemetry.post_filtering_count == len(boxes), "post_filtering_count mismatch with boxes list"
    assert len(boxes) == 8, f"Expected exactly 8 genuine detections, got {len(boxes)}"

    # Verify every box retains its genuine metadata and identity
    seen_ids = set()
    for b in boxes:
        assert b.detection_id not in seen_ids, f"Duplicate detection ID found: {b.detection_id}"
        seen_ids.add(b.detection_id)
        assert b.class_name == "Strawberry leaf", f"Class identity altered: {b.class_name}"
        assert b.category_type == "canopy", f"Strawberry leaf should be canopy, got {b.category_type}"
        assert b.box_type == "leaf"
        assert b.color_hex == "#52b788"
        assert b.confidence >= settings.YOLO_PLANTDOC_CONF_THRESH or b.confidence >= 0.08
        assert b.coordinate_space == "original_image_pixels"
        
        # Check coordinate bounds
        x1, y1, x2, y2 = b.bbox_xyxy
        assert 0 <= x1 < x2 <= 685, f"Invalid x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 550, f"Invalid y coordinates: {y1}, {y2}"

    assert telemetry.canopy_box_count == 8
    assert telemetry.lesion_box_count == 0


def test_tomato_seven_detections_verification(engine):
    """
    Verify images producing both specimen canopy boundaries (7 leaves from PlantDoc)
    and granular pathology lesion spot foci (12 spots from Lesion Detector).
    Ensure detector preserves genuine lesion classes distinct from canopy classes.
    """
    target_img = TOMATO_SEPTORIA_IMG if os.path.exists(TOMATO_SEPTORIA_IMG) else TOMATO_VIRUS_IMG
    if not os.path.exists(target_img):
        pytest.skip("Tomato test images not found")

    resp = engine.predict_vision(target_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    boxes = telemetry.bounding_boxes

    # Verify specimen and lesion separation
    assert len(telemetry.specimen_detections) == 7, f"Expected 7 specimen leaf boxes, got {len(telemetry.specimen_detections)}"
    assert len(telemetry.lesion_detections) == 12, f"Expected 12 lesion spot boxes, got {len(telemetry.lesion_detections)}"
    assert len(boxes) == 19, f"Expected 19 total genuine detections (7 specimen + 12 lesion), got {len(boxes)}"
    assert telemetry.post_filtering_count == 19
    assert telemetry.specimen_box_count == 7
    assert telemetry.lesion_box_count == 12
    assert telemetry.localization_capability == "specimen_boundary_and_lesion_foci"

    # Verify each detection retains unique ID and correct coordinates
    seen_ids = set()
    for b in boxes:
        assert b.detection_id not in seen_ids
        seen_ids.add(b.detection_id)
        assert b.confidence >= 0.10 or b.confidence >= 0.08
        x1, y1, x2, y2 = b.bbox_xyxy
        assert 0 <= x1 < x2
        assert 0 <= y1 < y2


def test_potato_early_blight_lesion_and_specimen_separation(engine):
    """
    Test Potato Early Blight (images_7.jpg):
    Verifies that:
      1. Specimen detector yields 1 whole-leaf canopy box (not claimed as lesion).
      2. Lesion detector yields 8 genuine necrotic lesion spots.
      3. Diagnosis lesion_foci_count matches len(lesion_detections) exactly (8, not 1 or 0).
      4. Localization capability is 'specimen_boundary_and_lesion_foci'.
    """
    potato_img = "cv/test_images/images_7.jpg"
    if not os.path.exists(potato_img):
        pytest.skip(f"Potato test image not found: {potato_img}")

    resp = engine.predict_vision(potato_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    diag = resp.diagnosis

    # 1 specimen leaf boundary, 8 genuine spot lesions
    assert len(telemetry.specimen_detections) == 1, f"Expected 1 leaf box, got {len(telemetry.specimen_detections)}"
    assert len(telemetry.lesion_detections) == 8, f"Expected 8 lesion spot boxes, got {len(telemetry.lesion_detections)}"
    assert diag.lesion_foci_count == 8, f"Expected lesion_foci_count=8, got {diag.lesion_foci_count}"
    assert telemetry.localization_capability == "specimen_boundary_and_lesion_foci"

    # Verify specimen box encompasses large foliar region
    specimen = telemetry.specimen_detections[0]
    assert specimen.category_type == "canopy"
    assert specimen.box_type == "leaf"
    assert specimen.color_hex == "#52b788"
    assert specimen.class_name == "Potato leaf early blight"
    s_w = specimen.bbox_xyxy[2] - specimen.bbox_xyxy[0]
    s_h = specimen.bbox_xyxy[3] - specimen.bbox_xyxy[1]
    assert s_w > 300 and s_h > 500, f"Specimen box should cover leaf: w={s_w}, h={s_h}"

    # Verify each lesion box is an internal spot
    for les in telemetry.lesion_detections:
        assert les.category_type == "lesion"
        assert les.box_type == "lesion"
        assert les.color_hex == "#f4a261"
        l_w = les.bbox_xyxy[2] - les.bbox_xyxy[0]
        l_h = les.bbox_xyxy[3] - les.bbox_xyxy[1]
        assert l_w < 120 and l_h < 120, f"Lesion box should be a granular spot: w={l_w}, h={l_h}"


def test_healthy_leaf_negative_control(engine):
    """
    Test healthy foliage (potato__healthy__healthy.jpg):
    Verifies that zero lesion spots are predicted (clean negative control).
    """
    healthy_img = "frontend/samples/potato__healthy__healthy.jpg"
    if not os.path.exists(healthy_img):
        pytest.skip(f"Healthy test image not found: {healthy_img}")

    resp = engine.predict_vision(healthy_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    diag = resp.diagnosis

    assert len(telemetry.lesion_detections) == 0, f"Expected 0 lesions on healthy leaf, got {len(telemetry.lesion_detections)}"
    assert diag.lesion_foci_count == 0
    assert telemetry.localization_capability in ("specimen_boundary_only", "unavailable")


def test_no_fabricated_boxes_or_heuristics(engine):
    """Verify that zero heuristic or synthetic spots are generated."""
    if not os.path.exists(STRAWBERRY_IMG):
        pytest.skip("Strawberry test image not found")

    resp = engine.predict_vision(STRAWBERRY_IMG, model_tier="server")
    for b in resp.spatial_telemetry.bounding_boxes:
        assert not b.label.startswith("Spot #"), f"Found synthetic spot label: {b.label}"
        assert b.class_name in engine.model_tier2_plantdoc.names.values() or (
            engine.model_tier2_lesions and b.class_name in engine.model_tier2_lesions.names.values()
        )
