"""
FastAPI Backend for Domain-Adaptive Speech Emotion Recognition using CORAL.

Provides REST API endpoints for real-time speech emotion inference,
health checks, and domain adaptation using pre-trained model and CORAL weights.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.model import EmotionPredictor, MODEL_PATH, CORAL_PATH

# Initialize FastAPI application
app = FastAPI(
    title="Domain-Adaptive Speech Emotion Recognition API",
    description="Speech Emotion Recognition with CORAL Domain Adaptation across RAVDESS and CREMA-D datasets.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor instance
predictor: Optional[EmotionPredictor] = None


def get_predictor() -> EmotionPredictor:
    """
    Retrieves or initializes the singleton EmotionPredictor instance.
    """
    global predictor
    if predictor is None:
        if not MODEL_PATH.exists() or not CORAL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model or CORAL reference weights not found. Please train the model first.",
            )
        try:
            predictor = EmotionPredictor(model_file=MODEL_PATH, coral_file=CORAL_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load trained model artifacts: {e}",
            )
    return predictor


@app.on_event("startup")
def startup_event():
    """
    Load model and CORAL reference parameters during application startup.
    """
    try:
        get_predictor()
        print("EmotionPredictor loaded successfully on API startup.")
    except Exception as e:
        print(f"Warning: Could not pre-load model on startup ({e}). Will load on first request.")


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """
    Root endpoint confirming the API service status.
    """
    return {
        "message": "Domain-Adaptive Speech Emotion Recognition API is running",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", tags=["General"])
def health_check() -> Dict[str, Union[str, bool, list]]:
    """
    Health check endpoint reporting system status and model readiness.
    """
    try:
        pred = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": pred.model is not None,
            "coral_loaded": pred.coral.is_fitted,
            "supported_emotions": pred.classes,
        }
    except HTTPException:
        return {
            "status": "degraded",
            "model_loaded": False,
            "coral_loaded": False,
            "supported_emotions": [],
        }


def validate_wav_header(content: bytes) -> bool:
    """
    Verifies that the uploaded byte buffer has a valid RIFF/WAVE header.
    """
    if len(content) < 12:
        return False
    # RIFF magic bytes at offset 0, WAVE format at offset 8
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


@app.post("/predict", tags=["Inference"])
async def predict_audio_emotion(
    file: UploadFile = File(..., description="Audio WAV file to analyze"),
    use_coral: bool = Query(True, description="Whether to apply CORAL domain adaptation"),
):
    """
    Accepts an uploaded WAV file, validates its RIFF/WAVE header,
    extracts 80-dim MFCC features, applies CORAL domain adaptation (optional, default True),
    and returns predicted emotion class and probability distribution.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded.",
        )

    # 1. Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {e}",
        )

    if not content or len(content) < 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or too small to be a valid audio recording.",
        )

    # 2. Validate WAV/RIFF binary format
    if not validate_wav_header(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio format. Only valid RIFF/WAVE (.wav) audio files are accepted.",
        )

    pred = get_predictor()
    temp_file_path: Optional[str] = None

    # 3. Temporarily save file for librosa processing and perform inference
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            temp_file_path = tmp.name

        # Run prediction
        result = pred.predict(temp_file_path, use_coral=use_coral)

        return {
            "status": "success",
            "filename": file.filename,
            "predicted_emotion": result["emotion"],
            "confidence": round(float(result["confidence"]), 4),
            "probabilities": {k: round(float(v), 4) for k, v in result["probabilities"].items()},
            "coral_applied": result["coral_applied"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emotion prediction failed: {str(e)}",
        )
    finally:
        # 4. Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
