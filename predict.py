#!/usr/bin/env python
# coding: utf-8
"""
predict.py — run the trained brain tumor detector on a single MRI image.

Usage:
    python predict.py --image path/to/scan.jpg
    python predict.py                       # prompts for a path interactively
"""

import argparse
from pathlib import Path

from inference import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    load_model_and_labels,
    predict,
    preprocess_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tumor / no tumor on an MRI image.")
    parser.add_argument("--image", type=str, default=None, help="Path to the MRI image.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path to the trained .keras model file.")
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS_PATH,
                         help="Path to label_classes.json.")
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    image_path_str = args.image
    if not image_path_str:
        image_path_str = input("Enter the path to the MRI image: ").strip().strip('"').strip("'")
    image_path = Path(image_path_str)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nLoading model from {args.model} ...")
    model, class_names = load_model_and_labels(args.model, args.labels)

    image_batch = preprocess_image(image_path)
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
