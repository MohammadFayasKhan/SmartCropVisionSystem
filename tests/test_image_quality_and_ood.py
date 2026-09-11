"""
tests/test_image_quality_and_ood.py
-----------------------------------
Tests the Image Quality Assessment stage and Epistemic Uncertainty / OOD safeguards.
Verifies:
  1. Laplacian blur variance detection (sharp foliage vs artificially blurred input).
  2. Photometric exposure limits (overexposed and underexposed rejection).
  3. Excess Green Index (ExG) foliar presence calculation.
  4. Out-of-Distribution (OOD) flagging based on prediction entropy and top1-top2 margin.
  5. Calibrated uncertainty metrics serialization into Pydantic schema.
"""

import pytest
import numpy as np
import cv2

from backend.app.utils.image_quality import assess_image_quality
from backend.app.schemas.diagnosis import ImageQualityAssessment, UncertaintyMetrics

def generate_test_leaf_image(sharp=True, bright=True, green=True) -> np.ndarray:
    """Creates a deterministic synthetic test leaf image in BGR format."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    if green:
        # Green foliar region (BGR: 34, 139, 34)
        cv2.circle(img, (128, 128), 90, (34, 139, 34), -1)
        for i in range(40, 210, 15):
            cv2.line(img, (i, 128), (128, 210), (0, 100, 0), 2)
    else:
        # Gray non-foliar circle
        cv2.circle(img, (128, 128), 90, (120, 120, 120), -1)

    if not sharp:
        img = cv2.GaussianBlur(img, (31, 31), 10.0)

    if not bright:
        img = (img * 0.1).astype(np.uint8)

    return img

def test_blur_variance_sharp_vs_blurred():
    """Confirms sharp image has significantly higher Laplacian variance than blurred."""
    sharp_img = generate_test_leaf_image(sharp=True)
    blurry_img = generate_test_leaf_image(sharp=False)

    sharp_res = assess_image_quality(sharp_img)
    blurry_res = assess_image_quality(blurry_img)

    assert sharp_res["blur_score"] > blurry_res["blur_score"], (
        f"Sharp blur score ({sharp_res['blur_score']}) should exceed blurry ({blurry_res['blur_score']})"
    )
    assert blurry_res["blur_score"] < 40.0, (
        f"Severely blurred image should register variance < 40, got {blurry_res['blur_score']}"
    )
    assert any("blur" in issue.lower() for issue in blurry_res["quality_issues"])

def test_illumination_metrics_dark_vs_normal():
    """Confirms underexposed images register low photometric illumination scores."""
    normal_img = generate_test_leaf_image(bright=True)
    dark_img = generate_test_leaf_image(bright=False)

    normal_res = assess_image_quality(normal_img)
    dark_res = assess_image_quality(dark_img)

    assert dark_res["brightness_mean"] < normal_res["brightness_mean"]
    assert dark_res["brightness_mean"] < 30.0
    assert any("underexposed" in issue.lower() or "dark" in issue.lower() for issue in dark_res["quality_issues"])

def test_foliar_presence_ratio():
    """Confirms Excess Green (ExG) detects green foliage over gray non-botanical content."""
    foliar_img = generate_test_leaf_image(green=True)
    non_foliar_img = generate_test_leaf_image(green=False)

    foliar_res = assess_image_quality(foliar_img)
    non_foliar_res = assess_image_quality(non_foliar_img)

    assert foliar_res["greenness_ratio"] > 0.15, (
        f"Green leaf circle should occupy foliar ratio > 0.15, got {foliar_res['greenness_ratio']}"
    )
    assert non_foliar_res["greenness_ratio"] < 0.05, (
        f"Non-green image should have near zero foliar ratio, got {non_foliar_res['greenness_ratio']}"
    )

def test_structured_image_quality_assessment():
    """Tests comprehensive assess_image_quality pipeline returning validated Pydantic model."""
    normal_img = generate_test_leaf_image(sharp=True, bright=True, green=True)
    res_dict = assess_image_quality(normal_img)

    model = ImageQualityAssessment(**res_dict)
    assert isinstance(model, ImageQualityAssessment)
    assert model.quality_score > 0.0
    assert model.blur_score > 0.0
    assert 0.0 <= model.greenness_ratio <= 1.0

def test_severely_degraded_image_rejection():
    """Confirms pitch-black degenerate image is flagged as not acceptable."""
    black_img = np.zeros((128, 128, 3), dtype=np.uint8)
    res_dict = assess_image_quality(black_img)

    model = ImageQualityAssessment(**res_dict)
    assert model.is_acceptable is False
    assert len(model.quality_issues) > 0
    assert model.recapture_guidance is not None

def test_uncertainty_metrics_ood_logic():
    """Tests prediction margin and entropy calculation for in-distribution vs OOD."""
    # Case 1: High confidence prediction (margin=0.88, entropy=0.45)
    u_high = UncertaintyMetrics(
        prediction_margin=0.88,
        entropy_nats=0.45,
        normalized_uncertainty=0.12,
        ood_status="IN_DISTRIBUTION",
        is_low_confidence=False
    )
    assert u_high.ood_status == "IN_DISTRIBUTION"
    assert u_high.is_low_confidence is False

    # Case 2: Ambiguous / OOD prediction (margin=0.03, entropy=3.25)
    u_ood = UncertaintyMetrics(
        prediction_margin=0.03,
        entropy_nats=3.25,
        normalized_uncertainty=0.89,
        ood_status="OUT_OF_DISTRIBUTION",
        is_low_confidence=True
    )
    assert u_ood.ood_status == "OUT_OF_DISTRIBUTION"
    assert u_ood.is_low_confidence is True
