"""
Pydantic Schemas for Crop Recommendation and Agro-Climatic Intelligence.
Defines typed contracts for environmental sensor inputs, crop suitability predictions,
and microclimate disease risk alerting.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CropRecommendationRequest(BaseModel):
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Ambient temperature in degrees Celsius (°C)")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative atmospheric humidity percentage (%)")
    soil_moisture: float = Field(..., ge=0.0, le=100.0, description="Volumetric soil moisture content percentage (%)")
    rain: int = Field(0, ge=0, le=1, description="Binary precipitation sensor status (1=rain detected, 0=dry)")
    rainfall_mm: Optional[float] = Field(None, ge=0.0, le=500.0, description="Optional continuous rainfall measurement in mm")

class CropCandidate(BaseModel):
    crop: str = Field(..., description="Crop common name")
    confidence_pct: float = Field(..., description="Calculated probability percentage")
    rank: int = Field(..., description="Recommendation ranking (1 to 3)")
    suitability_level: str = Field(..., description="Suitability rating: Optimal, Viable, or Marginal")

class AgroClimaticAlert(BaseModel):
    id: str = Field(..., description="Disease condition identifier")
    name: str = Field(..., description="Pathogen or disease condition name")
    type: str = Field(..., description="Pathology classification (Fungal, Bacterial, etc.)")
    severity: str = Field(..., description="Alert severity: CRITICAL, HIGH, MODERATE, WATCH")
    trigger: str = Field(..., description="Environmental threshold condition that triggered the alert")
    symptoms: str = Field(..., description="Typical foliar or fruit symptoms")
    pesticide: str = Field(..., description="Recommended chemical or biological control agent")
    technique: str = Field(..., description="Cultural management and preventative canopy practices")

class EnvironmentalFeaturesUsed(BaseModel):
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    soil_moisture: float = Field(..., description="Soil moisture in %")
    rain: int = Field(..., description="Binary rain state (0 or 1)")
    rainfall_mm: float = Field(..., description="Effective rainfall applied to scaler in mm")
    fungal_risk: int = Field(..., description="1 if high humidity and heat create fungal spore pressure")
    drought_risk: int = Field(..., description="1 if low moisture and no rain indicate water stress")
    waterlog_risk: int = Field(..., description="1 if excessive moisture or active precipitation present")

class CropPreset(BaseModel):
    id: str
    title: str
    region_description: str
    temperature: float
    humidity: float
    soil_moisture: float
    rain: int
    rainfall_mm: float
    expected_crop: str

class CropRecommendationResponse(BaseModel):
    status: str = Field("success", description="Response status")
    recommended_crop: str = Field(..., description="Primary recommended agricultural crop")
    confidence_pct: float = Field(..., description="Recommendation model confidence score (0 to 100)")
    top3_candidates: List[CropCandidate] = Field(..., description="Top-3 ranked crop options")
    disease_alerts: List[AgroClimaticAlert] = Field(default_factory=list, description="Active microclimate disease warnings")
    features_used: EnvironmentalFeaturesUsed = Field(..., description="Sanitized agro-climatic parameters and risk flags")
    agronomic_summary: str = Field(..., description="Human-readable decision explanation grounded in supplied parameters")
    latency_ms: float = Field(..., description="Model inference latency in milliseconds")
