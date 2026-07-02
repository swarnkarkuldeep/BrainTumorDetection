#!/usr/bin/env python
# coding: utf-8
"""
predict.py — run the trained brain tumor detector on a single MRI image.

"""

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224
MODEL_PATH = "brain_tumor_detector.keras"
LABELS_PATH = "label_classes.json"


def load_labels(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found at {labels_path}. It's created automatically by "
            "train.py — make sure it's sitting next to your model file."
        )
    with open(labels_path) as f:
        return json.load(f)


def preprocess_image(image_path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}. Is it a valid image file?")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)  # add batch dimension


def predict(model, image_batch: np.ndarray, class_names):
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "all_probabilities": {
            class_names[i]: float(probabilities[i]) for i in range(len(class_names))
        },
    }


def main():
    image_path_str = input("Enter the path to the MRI image: ").strip().strip('"').strip("'")
    image_path = Path(image_path_str)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    class_names = load_labels(Path(LABELS_PATH))

    image_batch = preprocess_image(image_path, img_size=IMG_SIZE)
    result = predict(model, image_batch, class_names)

    print("\n--- Prediction ---")
    print(f"Image:      {image_path.name}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nAll class probabilities:")
    for class_name, prob in result["all_probabilities"].items():
        print(f"  {class_name:<10s}: {prob * 100:6.2f}%")

    if result["confidence"] < 0.6:
        print(
            "\nNote: confidence is low. Treat this prediction with caution — it's not a "
            "substitute for radiologist review."
        )


if __name__ == "__main__":
    main()