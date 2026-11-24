"""
Day-34 - Coding Practice
Simple Student API built with FastAPI + Pydantic.

Endpoints:
    GET  /                -> Welcome message
    GET  /health          -> API health check
    GET  /students        -> List all students
    POST /students        -> Add a new student
    GET  /students/{id}   -> Get one student by id

Run:
    uvicorn main:app --reload
Swagger UI:
    http://127.0.0.1:8000/docs
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = FastAPI(
    title="Student Management API",
    description="Day-34 coding practice: a simple FastAPI app with Pydantic validation.",
    version="1.0.0",
)

START_TIME = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StudentCreate(BaseModel):
    """Data the client sends when adding a student (request body)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",  # reject unknown fields
        json_schema_extra={
            "example": {
                "name": "Ali Raza",
                "age": 22,
                "email": "ali.raza@example.com",
                "course": "Machine Learning",
                "gpa": 3.4,
            }
        },
    )

    name: str = Field(..., min_length=2, max_length=50, description="Full name of the student")
    age: int = Field(..., ge=16, le=60, description="Age between 16 and 60")
    email: str = Field(
        ...,
        max_length=100,
        pattern=r"^[\w\.\+\-]+@[\w\-]+(\.[\w\-]+)+$",
        description="A valid email address",
    )
    course: str = Field(..., min_length=2, max_length=50, description="Enrolled course")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA on a 4.0 scale (optional)")

    @field_validator("name")
    @classmethod
    def name_must_be_letters(cls, value: str) -> str:
        if not all(ch.isalpha() or ch in " .-'" for ch in value):
            raise ValueError("name may only contain letters, spaces, dots, hyphens and apostrophes")
        return value.title()

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, value: str) -> str:
        return value.lower()


class Student(StudentCreate):
    """Student as returned by the API (includes the server-generated id)."""

    id: int = Field(..., description="Unique student id")


class MessageResponse(BaseModel):
    message: str
    docs: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    total_students: int


# ---------------------------------------------------------------------------
# In-memory "database"
# ---------------------------------------------------------------------------
students_db: dict[int, Student] = {}
next_id = 1


def _add_student(data: StudentCreate) -> Student:
    global next_id
    student = Student(id=next_id, **data.model_dump())
    students_db[next_id] = student
    next_id += 1
    return student


# Seed data so GET /students returns something right away
for _seed in [
    StudentCreate(name="Muhammad Awais", age=23, email="awais@example.com", course="Artificial Intelligence", gpa=3.6),
    StudentCreate(name="Ayesha Khan", age=21, email="ayesha@example.com", course="Data Science", gpa=3.8),
    StudentCreate(name="Hamza Ali", age=22, email="hamza@example.com", course="Computer Vision"),
]:
    _add_student(_seed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=MessageResponse, tags=["General"])
def root():
    """Welcome message."""
    return {
        "message": "Welcome to the Student Management API!",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Simple health check to confirm the API is running."""
    now = datetime.now(timezone.utc)
    return {
        "status": "healthy",
        "timestamp": now,
        "uptime_seconds": round((now - START_TIME).total_seconds(), 2),
        "total_students": len(students_db),
    }


@app.get("/students", response_model=list[Student], tags=["Students"])
def get_students():
    """Return the list of all students."""
    return list(students_db.values())


@app.post(
    "/students",
    response_model=Student,
    status_code=status.HTTP_201_CREATED,
    tags=["Students"],
)
def create_student(student: StudentCreate):
    """Add a new student. The request body is validated by Pydantic."""
    if any(s.email == student.email for s in students_db.values()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A student with email '{student.email}' already exists",
        )
    return _add_student(student)


@app.get("/students/{student_id}", response_model=Student, tags=["Students"])
def get_student(student_id: int = Path(..., gt=0, description="The id of the student")):
    """Get a specific student by id."""
    student = students_db.get(student_id)
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id {student_id} not found",
        )
    return student
