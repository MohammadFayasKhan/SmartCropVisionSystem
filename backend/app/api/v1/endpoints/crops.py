"""
FastAPI Router for Crop Recommendation and Agro-Climatic Intelligence.
Provides endpoints for predictive crop matching, environmental disease alerting,
and regional agricultural parameter presets.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status
import logging

from backend.app.schemas.crop import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    CropPreset
)
from backend.app.services.crop_service import crop_service

logger = logging.getLogger("smartplant.api.crops")

router = APIRouter(prefix="/crops", tags=["Crop Intelligence"])


@router.post(
    "/recommend",
    response_model=CropRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Agro-Climatic Crop Recommendation",
    description="Accepts ambient temperature, relative humidity, soil moisture, and rain parameters to calculate top-3 crop recommendations and microclimate disease risk alerts."
)
async def recommend_crop(request: CropRecommendationRequest):
    """
    Executes inference via the trained 22-class Random Forest model and
    evaluates microclimate disease pathogen thresholds.
    """
    try:
        if not crop_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Crop Recommendation model is not initialized or failed to load."
            )
            
        return crop_service.recommend(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during crop recommendation inference: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during crop recommendation: {str(e)}"
        )


@router.get(
    "/presets",
    response_model=List[CropPreset],
    status_code=status.HTTP_200_OK,
    summary="List Curated Regional Agricultural Presets",
    description="Returns pre-calibrated environmental presets representing diverse agro-climatic zones for quick testing and exploration."
)
async def get_crop_presets():
    """Returns list of curated regional agro-climatic presets."""
    return crop_service.get_presets()


@router.get(
    "/classes",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List Supported Crop Classes",
    description="Returns all 22 crop classes recognized by the agro-climatic recommendation model."
)
async def get_crop_classes():
    """Returns the list of 22 supported crop taxonomy classes."""
    return crop_service.classes
