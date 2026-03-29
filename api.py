import logging
import os
import sys
from contextlib import asynccontextmanager

import cv2
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configures import MODEL_FILE, SCALER_FILE
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Module-level flag; set to True by the lifespan startup handler
models_loaded: bool = False

THRESHOLD = 0.61
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]

# Module-level model references populated during startup
_extractor: FeatureExtractor | None = None
_scaler = None
_svm = None


class ClassificationResult(BaseModel):
    label: str
    confidence: float


class Classifier:
    def classify(self, frame_bytes: bytes) -> ClassificationResult:
        # Decode bytes to numpy array
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return ClassificationResult(label="unknown", confidence=0.0)

        # Convert to RGB if fewer than 3 channels
        if len(frame.shape) < 3 or frame.shape[2] < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        margin_x = w // 4
        margin_y = h // 4
        crop = frame[margin_y: h - margin_y, margin_x: w - margin_x]

        if crop.size == 0:
            return ClassificationResult(label="unknown", confidence=0.0)

        features = _extractor.extract_features(crop)
        features = _scaler.transform(features.reshape(1, -1))
        probs = _svm.predict_proba(features)[0]
        max_prob = float(probs.max())

        if max_prob < THRESHOLD:
            label = "unknown"
        else:
            label = CLASS_NAMES[int(probs.argmax())]

        return ClassificationResult(label=label, confidence=max_prob)


classifier = Classifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_loaded, _extractor, _scaler, _svm
    try:
        logger.info("Loading FeatureExtractor...")
        _extractor = FeatureExtractor()
        logger.info("Loading scaler from %s...", SCALER_FILE)
        _scaler = joblib.load(SCALER_FILE)
        logger.info("Loading SVM model from %s...", MODEL_FILE)
        _svm = joblib.load(MODEL_FILE)
        models_loaded = True
        logger.info("All models loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        sys.exit(1)
    yield


def _get_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "")
    if not raw.strip():
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    if models_loaded:
        return {"status": "ok"}
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"status": "loading"})
