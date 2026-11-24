# Day-34 – Coding Practice: Student API with FastAPI

A simple REST API built with **FastAPI**, with request validation done by **Pydantic**.

## Endpoints

| Method | Endpoint          | Description              | Success code |
|--------|-------------------|--------------------------|--------------|
| GET    | `/`               | Welcome message          | 200          |
| GET    | `/health`         | API health check         | 200          |
| GET    | `/students`       | List all students        | 200          |
| POST   | `/students`       | Add a new student        | 201          |
| GET    | `/students/{id}`  | Get one student by id    | 200          |

Error responses: `404` when the student doesn't exist, `409` for a duplicate email, and `422` for invalid input.

## Validation rules (Pydantic)

| Field    | Rule                                                         |
|----------|--------------------------------------------------------------|
| `name`   | required, 2–50 chars, letters/spaces only (auto Title-Cased) |
| `age`    | required, integer between 16 and 60                          |
| `email`  | required, valid email format, must be unique (auto lowercased) |
| `course` | required, 2–50 chars                                         |
| `gpa`    | optional, 0.0–4.0                                            |
| extra    | unknown fields are rejected                                  |
| `{id}`   | path id must be an integer greater than 0                    |

## Setup & run

```bash
cd Day-34/Coding_Practice
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## Testing

### 1. Swagger UI
Open **http://127.0.0.1:8000/docs**, expand an endpoint, click **Try it out** and then **Execute**.
(ReDoc is also available at `/redoc`.)

### 2. Postman / Thunder Client
Import `Student_API.postman_collection.json`:
- **Postman:** File → Import → choose the file.
- **Thunder Client (VS Code):** Collections → menu (≡) → Import → choose the file.

The collection has 9 requests covering every endpoint plus the error cases (422, 404, 409).
`base_url` is set to `http://127.0.0.1:8000`.

### 3. Automated tests
```bash
pytest -v
```

## Sample request

```http
POST /students
Content-Type: application/json

{
  "name": "Ali Raza",
  "age": 22,
  "email": "ali.raza@example.com",
  "course": "Machine Learning",
  "gpa": 3.4
}
```

Response `201 Created`:

```json
{
  "name": "Ali Raza",
  "age": 22,
  "email": "ali.raza@example.com",
  "course": "Machine Learning",
  "gpa": 3.4,
  "id": 4
}
```

> Data is stored in memory, so it resets every time the server restarts.
