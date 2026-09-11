"""
Image-Based Plant Intelligence Inference Endpoint.
Accepts multipart leaf photograph uploads, coordinates server-side validation,
triggers hierarchical ML vision cascade, and returns typed agronomic diagnosis.
"""

from typing import Optional
import uuid
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from backend.app.services.inference_service import inference_engine
from backend.app.utils.image_processing import ImageValidationError
from backend.app.schemas.diagnosis import DiagnosisResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/vision/diagnose",
    response_model=DiagnosisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format, corrupted stream, or size violation"},
        503: {"model": ErrorResponse, "description": "Model weights or inference engine currently unavailable"},
        500: {"model": ErrorResponse, "description": "Internal model execution or inference failure"}
    },
    summary="Diagnose Plant Leaf Image",
    description="Upload a plant leaf photograph to execute the 3-Tier SmartCropVision pipeline."
)
async def diagnose_leaf_image(
    request: Request,
    file: UploadFile = File(..., description="Plant leaf photograph (JPEG, PNG, or WebP up to 15 MB)"),
    model_tier: Optional[str] = Form("server", description="Vision model tier: 'server', 'edge', or 'ensemble'"),
    include_explainability: bool = Form(False, description="Whether to compute the full 9-stage explainability suite"),
    crop_context: Optional[str] = Form(None, description="Optional crop hint (e.g., Tomato, Corn, Grape)"),
    temperature_c: Optional[float] = Form(None, description="Optional ambient temperature in Celsius"),
    humidity_pct: Optional[float] = Form(None, description="Optional relative humidity percentage"),
    symptom_notes: Optional[str] = Form(None, description="Optional grower symptom description")
) -> DiagnosisResponse:
    """
    Authoritative plant diagnostic pipeline:
    1. Validates upload stream, magic bytes, dimensions, and MIME format.
    2. Runs Tier 1 classification: Server-grade EfficientNet-B2 (default), Edge MobileNetV2, or Ensemble.
    3. Runs Tier 2 YOLOv8-nano lesion localization if infected.
    4. Runs Tier 3 Mobile-UNet sub-pixel segmentation if infected.
    5. Optionally computes 9-stage explainability (Grad-CAM and feature activation maps).
    6. Incorporates optional paired multimodal context with explicit modality traceability.
    7. Formulates grounded agronomic action advisory and returns payload.
    """
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    filename = file.filename or "uploaded_leaf.jpg"
    content_type = file.content_type

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "error_code": "FILE_READ_FAILED",
                "message": f"Failed to read uploaded file payload: {str(e)}",
                "recovery_hint": "Please try selecting the image file again.",
                "request_id": request_id
            }
        )
    finally:
        await file.close()

    # Build optional multimodal context dict if provided
    multimodal_context = None
    if any([crop_context, temperature_c is not None, humidity_pct is not None, symptom_notes]):
        multimodal_context = {
            k: v for k, v in {
                "crop_context": crop_context,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "symptom_notes": symptom_notes
            }.items() if v is not None and v != ""
        }

    try:
        response = inference_engine.run_inference(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            model_tier=model_tier or "server",
            include_explainability=include_explainability,
            request_id=request_id,
            multimodal_context=multimodal_context
        )
        return response
    except ImageValidationError as ive:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "IMAGE_VALIDATION_FAILED",
                "message": ive.message,
                "recovery_hint": ive.recovery_hint,
                "request_id": request_id
            }
        )
    except RuntimeError as rte:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "message": str(rte),
                "recovery_hint": "Please verify model checkpoint files are mounted in the model directory.",
                "request_id": request_id
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_code": "INFERENCE_PIPELINE_ERROR",
                "message": "An unexpected error occurred during neural inference execution.",
                "recovery_hint": "Please verify image integrity and retry. System logs contain the trace.",
                "request_id": request_id
            }
        )


@router.post(
    "/vision/explain",
    response_model=DiagnosisResponse,
    summary="Diagnose Plant Leaf Image with Full Explainability Suite",
    description="Upload a plant leaf photograph to execute the 3-Tier cascade and generate all 9 explainability stages."
)
async def explain_leaf_image(
    request: Request,
    file: UploadFile = File(..., description="Plant leaf photograph"),
    model_tier: Optional[str] = Form("server", description="Vision model tier"),
    crop_context: Optional[str] = Form(None, description="Optional crop hint"),
    temperature_c: Optional[float] = Form(None, description="Optional temperature in Celsius"),
    humidity_pct: Optional[float] = Form(None, description="Optional humidity percentage"),
    symptom_notes: Optional[str] = Form(None, description="Optional symptom notes")
) -> DiagnosisResponse:
    """Convenience endpoint that explicitly sets include_explainability=True."""
    return await diagnose_leaf_image(
        request=request,
        file=file,
        model_tier=model_tier,
        include_explainability=True,
        crop_context=crop_context,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        symptom_notes=symptom_notes
    )
