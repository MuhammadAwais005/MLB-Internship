# Day-34 – Mini Project: Student Management REST API

A small REST API built with **FastAPI** that can add, view, search, update and delete students.
Incoming data is validated with **Pydantic**, and every error comes back in the same clear format.
Data is kept in an **in-memory dictionary** for now; a database will be added later.

A **Streamlit** front end (`streamlit_app.py`) is included so the project can be deployed on Streamlit Cloud.

## Project structure

```
Mini_Project/
│
├── app/
│   ├── main.py              # creates the FastAPI app, error handlers, / and /health
│   ├── routes/
│   │   └── students.py      # all student endpoints + in-memory storage
│   └── schemas/
│       └── student.py       # Pydantic models (request/response validation)
│
├── tests/
│   └── test_students.py     # 23 automated tests
├── streamlit_app.py         # Streamlit UI (starts the API in the background)
├── Student_Management_API.postman_collection.json
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Endpoint                 | What it does                               | Success |
|--------|--------------------------|--------------------------------------------|---------|
| GET    | `/`                      | Welcome message + list of endpoints        | 200     |
| GET    | `/health`                | Health check                               | 200     |
| POST   | `/students`              | Add a student                              | 201     |
| GET    | `/students`              | View all students                          | 200     |
| GET    | `/students/search`       | Search by `name`, `course`, `email`, `min_age`, `max_age` | 200 |
| GET    | `/students/{id}`         | View one student                           | 200     |
| PUT    | `/students/{id}`         | Replace all of a student's information     | 200     |
| PATCH  | `/students/{id}`         | Update only the fields you send            | 200     |
| DELETE | `/students/{id}`         | Delete a student                           | 200     |

## Validation rules

| Field    | Rule                                                              |
|----------|-------------------------------------------------------------------|
| `name`   | required, 2–50 characters, letters/spaces only (saved in Title Case) |
| `age`    | required, whole number from 16 to 60                              |
| `email`  | required, valid format, must be unique (saved in lowercase)       |
| `course` | required, 2–50 characters                                         |
| `gpa`    | optional, 0.0–4.0                                                 |
| other    | unknown fields are rejected; PATCH needs at least one field       |
| `{id}`   | must be a whole number greater than 0                             |

## Error format

Every error uses the same shape:

```json
{
  "success": false,
  "status_code": 422,
  "error": "Validation failed",
  "details": [
    {"field": "age", "message": "Input should be greater than or equal to 16"},
    {"field": "email", "message": "Invalid email address"}
  ]
}
```

| Code | When                                                  |
|------|-------------------------------------------------------|
| 400  | Search with no parameters, `min_age > max_age`, or setting a required field to null |
| 404  | Student id doesn't exist                              |
| 409  | Email already belongs to another student              |
| 422  | Invalid or missing data                               |

## Run locally

```bash
cd Day-34/Mini_Project
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

**API only:**
```bash
uvicorn app.main:app --reload
```
Then open Swagger UI at http://127.0.0.1:8000/docs.

**Streamlit UI** (starts the API for you):
```bash
streamlit run streamlit_app.py
```

**Tests:**
```bash
pytest -v
```

## Testing with Postman / Thunder Client

Import `Student_Management_API.postman_collection.json`:
- **Postman:** File → Import.
- **Thunder Client:** Collections → ≡ → Import.

It contains 17 requests, grouped into General, Add, View, Search, Update and Delete, including the error cases.

## Deploying on Streamlit Cloud

1. Push the repo to GitHub.
2. On share.streamlit.io, create a new app from this repo.
3. Set **Main file path** to `Day-34/Mini_Project/streamlit_app.py`.
4. Deploy. Streamlit installs `Day-34/Mini_Project/requirements.txt` automatically.

`streamlit_app.py` starts the FastAPI server inside the Streamlit app and calls it over HTTP.
Data resets whenever the app restarts, because storage is in memory.
