"""
Automated tests for every endpoint.
Run:  pytest -v
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Welcome" in res.json()["message"]


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["total_students"] >= 3


def test_get_students():
    res = client.get("/students")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
    assert len(res.json()) >= 3


def test_create_student():
    payload = {"name": "sara ahmed", "age": 20, "email": "Sara@Example.com", "course": "NLP", "gpa": 3.2}
    res = client.post("/students", json=payload)
    assert res.status_code == 201
    body = res.json()
    assert body["id"] > 0
    assert body["name"] == "Sara Ahmed"          # title-cased by validator
    assert body["email"] == "sara@example.com"   # lower-cased by validator

    # the new student can be fetched by id
    res2 = client.get(f"/students/{body['id']}")
    assert res2.status_code == 200
    assert res2.json() == body


def test_create_duplicate_email():
    payload = {"name": "Copy Cat", "age": 25, "email": "awais@example.com", "course": "AI"}
    res = client.post("/students", json=payload)
    assert res.status_code == 409


def test_get_student_by_id():
    res = client.get("/students/1")
    assert res.status_code == 200
    assert res.json()["id"] == 1


def test_get_student_not_found():
    res = client.get("/students/9999")
    assert res.status_code == 404


def test_get_student_invalid_id():
    assert client.get("/students/0").status_code == 422
    assert client.get("/students/abc").status_code == 422


def test_validation_errors():
    bad_payloads = [
        {"name": "A", "age": 20, "email": "a@b.com", "course": "AI"},             # name too short
        {"name": "Ali", "age": 10, "email": "a@b.com", "course": "AI"},           # age too low
        {"name": "Ali", "age": 20, "email": "not-an-email", "course": "AI"},      # bad email
        {"name": "Ali", "age": 20, "email": "a@b.com", "course": "AI", "gpa": 5}, # gpa > 4
        {"name": "Ali123", "age": 20, "email": "a@b.com", "course": "AI"},        # digits in name
        {"name": "Ali", "age": 20, "email": "a@b.com"},                           # missing course
        {"name": "Ali", "age": 20, "email": "a@b.com", "course": "AI", "x": 1},   # unknown field
    ]
    for payload in bad_payloads:
        res = client.post("/students", json=payload)
        assert res.status_code == 422, payload
