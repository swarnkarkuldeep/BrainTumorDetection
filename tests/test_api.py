from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import MAX_UPLOAD_BYTES, app

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_EXISTS = (REPO_ROOT / "brain_tumor_detector.keras").exists() and (
    REPO_ROOT / "label_classes.json"
).exists()

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="brain_tumor_detector.keras / label_classes.json not present — run train.py first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _fake_jpeg_bytes():
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_health_returns_ok_and_classes(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["classes"]) == {"no", "yes"}


def test_predict_returns_valid_response_shape(client):
    files = {"file": ("scan.jpg", _fake_jpeg_bytes(), "image/jpeg")}

    response = client.post("/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {"no", "yes"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_probabilities"]) == {"no", "yes"}
    assert sum(body["all_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_rejects_wrong_content_type(client):
    files = {"file": ("notes.txt", b"just some text", "text/plain")}

    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_requires_file_field(client):
    response = client.post("/predict")

    assert response.status_code == 422


def test_predict_rejects_corrupt_image_bytes(client):
    files = {"file": ("scan.jpg", b"not actually an image", "image/jpeg")}

    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def test_predict_rejects_oversized_file(client):
    oversized = b"0" * (MAX_UPLOAD_BYTES + 1)
    files = {"file": ("scan.jpg", oversized, "image/jpeg")}

    response = client.post("/predict", files=files)

    assert response.status_code == 413
