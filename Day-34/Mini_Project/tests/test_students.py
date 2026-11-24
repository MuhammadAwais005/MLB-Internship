"""
Tests for every endpoint and error case.
Run from the Mini_Project folder:  pytest -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.students import reset_db

client = TestClient(app)

NEW_STUDENT = {"name": "sara ahmed", "age": 20, "email": "Sara@Example.com", "course": "NLP", "gpa": 3.2}


@pytest.fixture(autouse=True)
def fresh_db():
    reset_db()  # every test starts with the 3 sample students
    yield


# General ---------------------------------------------------------------------
def test_root_and_health():
    assert client.get("/").status_code == 200
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["total_students"] == 3


# Add -------------------------------------------------------------------------
def test_add_student():
    res = client.post("/students", json=NEW_STUDENT)
    assert res.status_code == 201
    body = res.json()
    assert body["id"] == 4
    assert body["name"] == "Sara Ahmed"
    assert body["email"] == "sara@example.com"


def test_add_duplicate_email():
    res = client.post("/students", json={**NEW_STUDENT, "email": "awais@example.com"})
    assert res.status_code == 409
    assert res.json()["success"] is False
    assert "already exists" in res.json()["error"]


@pytest.mark.parametrize(
    "payload, field",
    [
        ({**NEW_STUDENT, "name": "A"}, "name"),
        ({**NEW_STUDENT, "name": "Ali123"}, "name"),
        ({**NEW_STUDENT, "age": 10}, "age"),
        ({**NEW_STUDENT, "age": "twenty"}, "age"),
        ({**NEW_STUDENT, "email": "not-an-email"}, "email"),
        ({**NEW_STUDENT, "gpa": 4.5}, "gpa"),
        ({k: v for k, v in NEW_STUDENT.items() if k != "course"}, "course"),
        ({**NEW_STUDENT, "phone": "123"}, "phone"),
    ],
)
def test_add_validation_errors(payload, field):
    res = client.post("/students", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "Validation failed"
    assert any(d["field"] == field for d in body["details"])


# View ------------------------------------------------------------------------
def test_view_all():
    res = client.get("/students")
    assert res.status_code == 200
    assert res.json()["count"] == 3


def test_view_one():
    res = client.get("/students/2")
    assert res.status_code == 200
    assert res.json()["name"] == "Ayesha Khan"


def test_view_one_not_found_and_invalid():
    res = client.get("/students/99")
    assert res.status_code == 404
    assert res.json()["error"] == "Student with id 99 not found"
    assert client.get("/students/0").status_code == 422
    assert client.get("/students/abc").status_code == 422


# Search ----------------------------------------------------------------------
def test_search_by_name_case_insensitive():
    res = client.get("/students/search", params={"name": "AYESHA"})
    assert res.status_code == 200
    assert res.json()["count"] == 1


def test_search_by_course_and_age():
    res = client.get("/students/search", params={"course": "data", "min_age": 20, "max_age": 22})
    assert res.json()["count"] == 1
    res = client.get("/students/search", params={"min_age": 22})
    assert res.json()["count"] == 2


def test_search_no_results():
    res = client.get("/students/search", params={"name": "nobody"})
    assert res.status_code == 200
    assert res.json() == {"count": 0, "students": []}


def test_search_errors():
    assert client.get("/students/search").status_code == 400
    assert client.get("/students/search", params={"min_age": 30, "max_age": 20}).status_code == 400


# Update ----------------------------------------------------------------------
def test_patch_partial_update():
    res = client.patch("/students/1", json={"course": "Deep Learning", "gpa": 3.9})
    assert res.status_code == 200
    s = res.json()["student"]
    assert s["course"] == "Deep Learning" and s["gpa"] == 3.9
    assert s["name"] == "Muhammad Awais"  # unchanged


def test_put_full_update():
    res = client.put("/students/3", json=NEW_STUDENT)
    assert res.status_code == 200
    assert res.json()["student"]["name"] == "Sara Ahmed"
    assert res.json()["student"]["id"] == 3


def test_update_errors():
    assert client.patch("/students/99", json={"age": 30}).status_code == 404
    assert client.patch("/students/1", json={}).status_code == 422
    assert client.patch("/students/1", json={"age": 100}).status_code == 422
    assert client.patch("/students/1", json={"name": None}).status_code == 400
    assert client.patch("/students/1", json={"email": "ayesha@example.com"}).status_code == 409
    # keeping your own email is fine
    assert client.patch("/students/1", json={"email": "awais@example.com"}).status_code == 200
    assert client.put("/students/1", json={"name": "Only Name"}).status_code == 422


# Delete ----------------------------------------------------------------------
def test_delete_student():
    res = client.delete("/students/2")
    assert res.status_code == 200
    assert "deleted successfully" in res.json()["message"]
    assert client.get("/students/2").status_code == 404
    assert client.get("/students").json()["count"] == 2


def test_delete_not_found():
    res = client.delete("/students/99")
    assert res.status_code == 404
