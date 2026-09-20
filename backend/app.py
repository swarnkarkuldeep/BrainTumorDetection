#!/usr/bin/env python
# coding: utf-8
"""
backend/app.py — FastAPI service exposing the brain tumor detector over HTTP.

Run with:
    uvicorn backend.app:app --reload --port 8000
(from the repo root, so `inference.py` and the model files resolve correctly)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# repo root (parent of backend/) holds inference.py and the trained model files
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import inference  # noqa: E402  (must come after sys.path fix-up)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO_ROOT / inference.DEFAULT_MODEL_PATH))
LABELS_PATH = os.environ.get("LABELS_PATH", str(REPO_ROOT / inference.DEFAULT_LABELS_PATH))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:80"
    ).split(",")
    if origin.strip()
]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

model_state = {"model": None, "class_names": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, class_names = inference.load_model_and_labels(MODEL_PATH, LABELS_PATH)
        model_state["model"] = model
        model_state["class_names"] = class_names
        log.info("Model loaded from %s (classes: %s)", MODEL_PATH, class_names)
    except Exception as exc:  # noqa: BLE001 - want to serve a 503 instead of crashing
        model_state["load_error"] = str(exc)
        log.error("Failed to load model: %s", exc)
    yield


app = FastAPI(title="Brain Tumor Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ProbabilitiesResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    classes: list[str] | None = None


def require_model():
    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {model_state['load_error']}",
        )
    return model_state["model"], model_state["class_names"]


@app.get("/health", response_model=HealthResponse)
def health():
    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {model_state['load_error']}",
        )
    return HealthResponse(status="ok", classes=model_state["class_names"])


@app.post("/predict", response_model=ProbabilitiesResponse)
async def predict(file: UploadFile = File(...)):
    model, class_names = require_model()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.",
        )
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_array = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image.")

    image_batch = inference.preprocess_image_array(image_array)
    result = inference.predict(model, image_batch, class_names)
    return ProbabilitiesResponse(**result)
