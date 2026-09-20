import json

import cv2
import numpy as np
import pytest

import inference


class FakeModel:
    """Stands in for a Keras model so prediction logic can be tested without
    loading real weights."""

    def __init__(self, probabilities):
        self._probabilities = np.array([probabilities], dtype="float32")

    def predict(self, batch, verbose=0):
        return self._probabilities


def _random_bgr_image(height=64, width=48):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestLoadLabels:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inference.load_labels(tmp_path / "does_not_exist.json")

    def test_valid_file_returns_list(self, tmp_path):
        labels_path = tmp_path / "label_classes.json"
        labels_path.write_text(json.dumps(["no", "yes"]))

        assert inference.load_labels(labels_path) == ["no", "yes"]


class TestLoadModelAndLabels:
    def test_missing_model_raises(self, tmp_path):
        labels_path = tmp_path / "label_classes.json"
        labels_path.write_text(json.dumps(["no", "yes"]))

        with pytest.raises(FileNotFoundError):
            inference.load_model_and_labels(tmp_path / "missing.keras", labels_path)


class TestPreprocessImage:
    def test_array_output_shape_and_range(self):
        image = _random_bgr_image()

        batch = inference.preprocess_image_array(image)

        assert batch.shape == (1, inference.IMG_SIZE, inference.IMG_SIZE, 3)
        assert batch.dtype == np.float32
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_from_disk_matches_array_path(self, tmp_path):
        image = _random_bgr_image()
        image_path = tmp_path / "scan.jpg"
        cv2.imwrite(str(image_path), image)

        batch = inference.preprocess_image(image_path)

        assert batch.shape == (1, inference.IMG_SIZE, inference.IMG_SIZE, 3)
        assert batch.dtype == np.float32

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            inference.preprocess_image(tmp_path / "does_not_exist.jpg")


class TestPredict:
    def test_returns_expected_structure(self):
        model = FakeModel([0.2, 0.8])
        dummy_batch = np.zeros((1, inference.IMG_SIZE, inference.IMG_SIZE, 3), dtype="float32")

        result = inference.predict(model, dummy_batch, ["no", "yes"])

        assert result["predicted_class"] == "yes"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["all_probabilities"] == {
            "no": pytest.approx(0.2),
            "yes": pytest.approx(0.8),
        }

    def test_picks_highest_probability_class(self):
        model = FakeModel([0.9, 0.1])
        dummy_batch = np.zeros((1, inference.IMG_SIZE, inference.IMG_SIZE, 3), dtype="float32")

        result = inference.predict(model, dummy_batch, ["no", "yes"])

        assert result["predicted_class"] == "no"
