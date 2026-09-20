#!/usr/bin/env python
# coding: utf-8
"""
inference.py — shared preprocessing/prediction logic for the brain tumor
detector. Used by predict.py (CLI) and the backend API, so there's one
place that knows how to load the model and turn an image into a prediction.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224
DEFAULT_MODEL_PATH = "brain_tumor_detector.keras"
DEFAULT_LABELS_PATH = "label_classes.json"


def load_labels(labels_path: Path) -> list:
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found at {labels_path}. It's created automatically by "
            "train.py — make sure it's sitting next to your model file."
        )
    with open(labels_path) as f:
        return json.load(f)


def load_model_and_labels(model_path: Path = DEFAULT_MODEL_PATH,
                           labels_path: Path = DEFAULT_LABELS_PATH):
    """Load the trained model and its class names together."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train.py first, or pass "
            "the correct path."
        )
    model = load_model(model_path)
    class_names = load_labels(labels_path)
    return model, class_names


def preprocess_image(image_path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Load an image from disk and turn it into a (1, H, W, 3) float32 batch."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}. Is it a valid image file?")
    return preprocess_image_array(image, img_size=img_size)


def preprocess_image_array(image: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Same preprocessing as preprocess_image, but starting from an
    already-decoded BGR image array (e.g. from cv2.imdecode on an upload)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)  # add batch dimension


def predict(model, image_batch: np.ndarray, class_names: list) -> dict:
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "all_probabilities": {
            class_names[i]: float(probabilities[i]) for i in range(len(class_names))
        },
    }
