"""
API tests covering every required case:
valid image, unsupported file type, empty file, image with no detections,
different confidence thresholds, plus the processed-image endpoint and other errors.

Run from the Mini_Project folder:  pytest -v
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TEST_FILES = SAMPLES / "test_files"
WORKER = SAMPLES / "factory_worker.jpg"
GROUP = SAMPLES / "group_handover.jpg"
VEST = SAMPLES / "rooftop_red_vest.jpg"
NO_OBJECTS = SAMPLES / "no_objects.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup, so the model is loaded once
        yield c


def send(client, path, endpoint="/predict", ctype="image/jpeg", **params):
    files = {"file": (path.name, path.read_bytes(), ctype)}
    return client.post(endpoint, files=files, params=params)


# ---------------------------------------------------------------- health
def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "best.pt"
    assert set(body["classes"].values()) == {"helmet", "boots", "person", "vest"}


# ---------------------------------------------------------------- 1. valid image
def test_valid_image(client):
    res = send(client, VEST)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["count"] == len(body["detections"]) >= 3
    assert sum(body["counts_by_class"].values()) == body["count"]
    for d in body["detections"]:
        b = d["bbox"]
        assert 0 <= b["x1"] < b["x2"] <= body["image"]["width"]
        assert 0 <= b["y1"] < b["y2"] <= body["image"]["height"]
        assert b["width"] == b["x2"] - b["x1"]
        assert 0.25 <= d["confidence"] <= 1


@pytest.mark.parametrize("path", [WORKER, GROUP, VEST], ids=lambda p: p.name)
def test_all_samples(client, path):
    assert send(client, path).status_code == 200


def test_png_upload(client):
    buf = io.BytesIO()
    Image.open(VEST).save(buf, format="PNG")
    res = client.post("/predict", files={"file": ("vest.png", buf.getvalue(), "image/png")})
    assert res.status_code == 200 and res.json()["count"] >= 1


# ---------------------------------------------------------------- 2. unsupported type
def test_unsupported_file_type(client):
    res = send(client, TEST_FILES / "not_an_image.txt", ctype="text/plain")
    assert res.status_code == 415
    assert "Unsupported file type '.txt'" in res.json()["error"]


def test_unsupported_content_type(client):
    res = send(client, WORKER, ctype="application/pdf")
    assert res.status_code == 415


def test_fake_image(client):
    res = send(client, TEST_FILES / "fake_image.jpg")
    assert res.status_code == 400
    assert "could not be read as an image" in res.json()["error"]


# ---------------------------------------------------------------- 3. empty file
def test_empty_file(client):
    res = send(client, TEST_FILES / "empty.jpg")
    assert res.status_code == 400
    assert res.json()["error"] == "The uploaded file is empty"


# ---------------------------------------------------------------- 4. no detections
def test_image_with_no_objects(client):
    res = send(client, NO_OBJECTS)
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 0 and body["detections"] == [] and body["counts_by_class"] == {}
    assert body["message"].startswith("No objects detected")


# ---------------------------------------------------------------- 5. confidence thresholds
def test_different_confidence_thresholds(client):
    counts = [send(client, GROUP, conf=c).json()["count"] for c in (0.1, 0.25, 0.5, 0.9)]
    assert counts == sorted(counts, reverse=True), counts  # higher threshold -> fewer boxes
    assert counts[0] > counts[-1]


def test_threshold_is_respected(client):
    body = send(client, GROUP, conf=0.5).json()
    assert body["confidence_threshold"] == 0.5
    assert all(d["confidence"] >= 0.5 for d in body["detections"])


@pytest.mark.parametrize("conf", [0, 1.5, "abc"])
def test_invalid_confidence(client, conf):
    res = send(client, WORKER, conf=conf)
    assert res.status_code == 422
    assert any("conf" in d for d in res.json()["details"])


# ---------------------------------------------------------------- class filter
def test_class_filter(client):
    body = send(client, VEST, classes="boots").json()
    assert body["count"] >= 1
    assert set(body["counts_by_class"]) == {"boots"}


def test_unknown_class(client):
    res = send(client, VEST, classes="car")
    assert res.status_code == 400
    assert "Unknown class" in res.json()["error"]


# ---------------------------------------------------------------- processed image
def test_predict_image(client):
    res = send(client, VEST, endpoint="/predict/image")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert int(res.headers["x-detection-count"]) >= 3
    img = Image.open(io.BytesIO(res.content))
    assert img.size == Image.open(VEST).size


def test_predict_image_errors_are_json(client):
    res = send(client, TEST_FILES / "empty.jpg", endpoint="/predict/image")
    assert res.status_code == 400
    assert res.headers["content-type"] == "application/json"


# ---------------------------------------------------------------- other errors
def test_missing_file(client):
    res = client.post("/predict")
    assert res.status_code == 422
    assert "no image was uploaded" in res.json()["details"][0]


def test_file_too_large(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_SIZE", 1024)
    res = send(client, WORKER)
    assert res.status_code == 413
