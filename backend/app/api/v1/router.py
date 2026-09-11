"""
API v1 Router Aggregator.
Combines health check, models status, and vision diagnosis endpoints.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, diagnosis, crops

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Status"])
api_router.include_router(diagnosis.router, tags=["Plant Intelligence Vision"])
api_router.include_router(crops.router, tags=["Crop Recommendation Intelligence"])
