"""
Tests for /health and /predict, including invalid input.
Run from the Coding_Practice folder:  pytest -v
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SAMPLE = SAMPLES / "factory_worker.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the lifespan, so the model gets loaded
        yield c


def upload(path: Path, content_type="image/jpeg", name=None):
    return {"file": (name or path.name, path.read_bytes(), content_type)}


def make_png(size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


# Health ----------------------------------------------------------------------
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert set(body["classes"].values()) == {"helmet", "boots", "person", "vest"}


# Predict: happy path ---------------------------------------------------------
@pytest.mark.parametrize("image", sorted(SAMPLES.glob("*.jpg")), ids=lambda p: p.name)
def test_predict_samples(client, image):
    res = client.post("/predict", files=upload(image))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["total"] == len(body["detections"])
    assert sum(body["counts"].values()) == body["total"]
    w, h = body["image_size"]["width"], body["image_size"]["height"]
    for det in body["detections"]:
        assert set(det) >= {"class", "confidence", "bbox"}
        assert det["class"] in {"helmet", "boots", "person", "vest"}
        assert 0 <= det["confidence"] <= 1
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_predict_finds_something(client):
    body = client.post("/predict", files=upload(SAMPLE), params={"conf": 0.25}).json()
    assert body["total"] >= 1


def test_high_confidence_returns_fewer(client):
    low = client.post("/predict", files=upload(SAMPLES / "group_handover.jpg"), params={"conf": 0.1}).json()
    high = client.post("/predict", files=upload(SAMPLES / "group_handover.jpg"), params={"conf": 0.9}).json()
    assert high["total"] <= low["total"]


def test_blank_image_has_no_detections(client):
    res = client.post("/predict", files={"file": ("blank.png", make_png(), "image/png")})
    assert res.status_code == 200
    assert res.json() == {**res.json(), "detections": [], "total": 0}


def test_octet_stream_content_type_is_accepted(client):
    res = client.post("/predict", files=upload(SAMPLE, content_type="application/octet-stream"))
    assert res.status_code == 200


# Predict: processed image ----------------------------------------------------
def test_predict_image_returns_jpeg(client):
    res = client.post("/predict/image", files=upload(SAMPLE))
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert int(res.headers["x-total-detections"]) >= 1
    Image.open(io.BytesIO(res.content)).verify()


# Predict: invalid input ------------------------------------------------------
def test_no_file(client):
    res = client.post("/predict")
    assert res.status_code == 422
    assert "No image uploaded" in res.json()["error"]


def test_wrong_extension(client):
    res = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_fake_image(client):
    res = client.post("/predict", files={"file": ("fake.jpg", b"this is not an image", "image/jpeg")})
    assert res.status_code == 400
    assert "not a valid image" in res.json()["error"]


def test_empty_file(client):
    res = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert res.status_code == 400
    assert "empty" in res.json()["error"]


def test_too_small_image(client):
    res = client.post("/predict", files={"file": ("tiny.png", make_png((10, 10)), "image/png")})
    assert res.status_code == 400


def test_too_large_file(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_SIZE", 1000)
    res = client.post("/predict", files=upload(SAMPLE))
    assert res.status_code == 413


def test_bad_confidence(client):
    res = client.post("/predict", files=upload(SAMPLE), params={"conf": 2})
    assert res.status_code == 422


def test_temp_files_are_cleaned_up(client, tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client.post("/predict", files=upload(SAMPLE))
    assert list(tmp_path.iterdir()) == []
