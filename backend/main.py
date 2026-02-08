"""
FastAPI backend for voice emotion detection
Production-ready inference API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import tempfile
from typing import Dict
from audio_processor import extract_features

# Initialize FastAPI app
app = FastAPI(
    title="Voice Emotion Detection API",
    description="Predict emotions from voice audio files",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response model
class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    valid: bool
    error: str = None


# Global model variables (loaded once at startup)
model = None
label_encoder = None
scaler = None


@app.on_event("startup")
async def load_models():
    """Load all model components at startup"""
    global model, label_encoder, scaler

    try:
        model_dir = "models"

        print("Loading models...")
        model = joblib.load(os.path.join(model_dir, "emotion_classifier.joblib"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

        print("✅ Models loaded successfully!")
        print(f"   Emotion classes: {label_encoder.classes_}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Voice Emotion Detection API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/emotions"]
    }


@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "emotions": label_encoder.classes_.tolist(),
        "count": len(label_encoder.classes_)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file

    Args:
        file: Audio file (.wav format)

    Returns:
        PredictionResponse with emotion, confidence, and probabilities
    """
    # Validate models are loaded
    if model is None or label_encoder is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Validate file format
    if not file.filename.endswith('.wav'):
        return PredictionResponse(
            emotion="",
            confidence=0.0,
            all_probabilities={},
            valid=False,
            error="Invalid file format. Please upload a .wav file"
        )

    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # Write uploaded file to temp location
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Extract features
        features = extract_features(tmp_file_path)

        # Clean up temp file
        os.unlink(tmp_file_path)

        # Validate audio
        if features is None:
            return PredictionResponse(
                emotion="",
                confidence=0.0,
                all_probabilities={},
                valid=False,
                error="Invalid audio: File is too quiet, corrupted, or not a valid audio format"
            )

        # Reshape and scale features
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Decode emotion
        emotion = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        # Create probability dictionary
        all_probs = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return PredictionResponse(
            emotion=emotion,
            confidence=confidence,
            all_probabilities=all_probs,
            valid=True
        )

    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        return PredictionResponse(
            emotion="",
            confidence=0.0,
            all_probabilities={},
            valid=False,
            error=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)