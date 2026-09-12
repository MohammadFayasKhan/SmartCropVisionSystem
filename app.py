"""
SmartCropVision Cloud Production Entrypoint for Hugging Face Spaces.
Unified deployment entrypoint that binds the FastAPI service and static frontend
dashboard to the target host and port (defaulting to 7860 on Hugging Face).
"""
import os
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SmartCropVision unified production server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
