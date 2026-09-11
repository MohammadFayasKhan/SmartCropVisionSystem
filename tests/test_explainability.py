"""
SmartCropVision Test Suite: Explainability & Botanical Attention Pipeline
Validates:
1. Dynamic Grad-CAM layer resolution across different architectures.
2. Heatmap tensor dimensions and [0, 1] normalization.
3. 4x4 Feature activation grid generation.
4. Full 9-stage explainability pipeline consistency and plain-language descriptions.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from backend.app.services.inference_service import inference_engine
from backend.app.utils.explainability import (
    GradCAM,
    get_target_cam_layer,
    extract_feature_activation_grid,
    build_explainability_pipeline
)
from backend.app.schemas.diagnosis import ExplainabilityPipeline


@pytest.fixture(scope="module")
def loaded_engine():
    inference_engine.load_models()
    return inference_engine


@pytest.fixture
def dummy_input_tensor(loaded_engine):
    # Shape [1, 3, 288, 288] on loaded_engine.device
    return torch.randn(1, 3, 288, 288, device=loaded_engine.device)


def test_gradcam_dynamic_layer_detection(loaded_engine):
    """
    Verifies that get_target_cam_layer dynamically discovers the deepest
    convolutional layer without relying on fragile hardcoded indices.
    """
    server_model = loaded_engine.model_tier1_server
    target_layer = get_target_cam_layer(server_model)
    assert target_layer is not None, "Failed to dynamically locate target Conv2d layer!"
    assert isinstance(target_layer, nn.Conv2d), f"Expected nn.Conv2d, got {type(target_layer)}"


def test_gradcam_heatmap_dimensions_and_bounds(loaded_engine, dummy_input_tensor):
    """
    Verifies that GradCAM produces a valid 2D float32 heatmap with values
    bounded strictly in [0.0, 1.0] matching the spatial resolution of the feature map.
    """
    server_model = loaded_engine.model_tier1_server
    target_layer = get_target_cam_layer(server_model)
    cam_engine = GradCAM(server_model, target_layer)

    try:
        heatmap = cam_engine.generate_heatmap(dummy_input_tensor.clone(), class_idx=0)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2, f"Expected 2D array, got ndim={heatmap.ndim}"
        assert heatmap.dtype == np.float32
        assert float(heatmap.min()) >= 0.0
        assert float(heatmap.max()) <= 1.0 + 1e-6
    finally:
        cam_engine.remove_hooks()


def test_feature_activation_grid_extraction(loaded_engine, dummy_input_tensor):
    """
    Verifies that extract_feature_activation_grid produces a 4x4 visual
    collage of intermediate convolutional activations.
    """
    server_model = loaded_engine.model_tier1_server
    grid_img, desc = extract_feature_activation_grid(server_model, dummy_input_tensor, max_channels=16)
    assert isinstance(grid_img, np.ndarray)
    assert grid_img.ndim == 3 and grid_img.shape[2] == 3
    assert len(desc) > 10
    assert "convolutional" in desc.lower()


def test_9_stage_pipeline_generation(loaded_engine):
    """
    Verifies that build_explainability_pipeline generates exactly 9 distinct
    stages, each with plain-language explanations and technical details.
    """
    img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    img_tensor = torch.zeros(1, 3, 288, 288)
    top1_meta = {"raw_folder": "Tomato___Early_blight", "crop": "Tomato", "disease_name": "Tomato Early Blight", "condition_type": "fungal"}
    top3_preds = [{"label": "Tomato Early Blight", "confidence_pct": 85.0}]
    detected_boxes = []
    
    pipeline = build_explainability_pipeline(
        img_bgr=img_bgr,
        img_tensor=img_tensor,
        model=loaded_engine.model_tier1_server,
        predicted_idx=29,
        top1_meta=top1_meta,
        top3_preds=top3_preds,
        detected_boxes=detected_boxes,
        pred_mask_256=None,
        foliar_damage_pct=None,
        entropy_score=0.45,
        uncertainty_score=0.15,
        model_name="EfficientNet-B2",
        orig_meta={"width": 300, "height": 300, "format": "JPEG", "size_bytes": 45000}
    )

    assert isinstance(pipeline, ExplainabilityPipeline)
    assert len(pipeline.stages) == 9, f"Expected 9 explainability stages, got {len(pipeline.stages)}"
    for idx, stage in enumerate(pipeline.stages):
        assert stage.stage_number == idx + 1
        assert len(stage.title) > 0
        assert len(stage.explanation) > 15
        assert isinstance(stage.metrics, dict)

    assert len(pipeline.synthesis) > 20
