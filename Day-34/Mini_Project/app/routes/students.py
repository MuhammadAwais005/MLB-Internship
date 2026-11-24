"""
Student routes: all CRUD + search endpoints live here.

Data is kept in an in-memory dictionary for now ({id: student_dict}).
It resets whenever the server restarts; a real database comes later.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.schemas.student import (
    ErrorResponse,
    MessageResponse,
    StudentCreate,
    StudentListResponse,
    StudentResponse,
    StudentUpdate,
)

router = APIRouter(prefix="/students", tags=["Students"])

# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------
students_db: dict[int, dict] = {}
_next_id = 1


def _insert(data: dict) -> dict:
    global _next_id
    student = {"id": _next_id, **data}
    students_db[_next_id] = student
    _next_id += 1
    return student


def _get_or_404(student_id: int) -> dict:
    student = students_db.get(student_id)
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id {student_id} not found",
        )
    return student


def _ensure_email_unique(email: str, exclude_id: Optional[int] = None) -> None:
    for sid, s in students_db.items():
        if s["email"] == email and sid != exclude_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A student with email '{email}' already exists",
            )


def reset_db(seed: bool = True) -> None:
    """Clear the store (used by tests) and optionally add sample students."""
    global _next_id
    students_db.clear()
    _next_id = 1
    if seed:
        for s in SEED_STUDENTS:
            _insert(StudentCreate(**s).model_dump())


SEED_STUDENTS = [
    {"name": "Muhammad Awais", "age": 23, "email": "awais@example.com", "course": "Artificial Intelligence", "gpa": 3.6},
    {"name": "Ayesha Khan", "age": 21, "email": "ayesha@example.com", "course": "Data Science", "gpa": 3.8},
    {"name": "Hamza Ali", "age": 22, "email": "hamza@example.com", "course": "Computer Vision", "gpa": 3.1},
]
reset_db()

# Error responses shown in Swagger UI
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Student not found"}}
CONFLICT = {409: {"model": ErrorResponse, "description": "Email already exists"}}
INVALID = {422: {"model": ErrorResponse, "description": "Validation error"}}


# ---------------------------------------------------------------------------
# Routes
# NOTE: /search must be declared before /{student_id}, otherwise FastAPI
# would try to read "search" as a student id.
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={**CONFLICT, **INVALID},
    summary="Add a student",
)
def add_student(student: StudentCreate):
    _ensure_email_unique(student.email)
    return _insert(student.model_dump())


@router.get("", response_model=StudentListResponse, summary="View all students")
def get_all_students():
    students = list(students_db.values())
    return {"count": len(students), "students": students}


@router.get(
    "/search",
    response_model=StudentListResponse,
    responses={**INVALID},
    summary="Search students",
)
def search_students(
    name: Optional[str] = Query(None, min_length=1, description="Part of the name (case-insensitive)"),
    course: Optional[str] = Query(None, min_length=1, description="Part of the course (case-insensitive)"),
    email: Optional[str] = Query(None, min_length=1, description="Exact email"),
    min_age: Optional[int] = Query(None, ge=0, description="Minimum age"),
    max_age: Optional[int] = Query(None, ge=0, description="Maximum age"),
):
    if all(v is None for v in (name, course, email, min_age, max_age)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one search parameter: name, course, email, min_age or max_age",
        )
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_age cannot be greater than max_age",
        )

    results = []
    for s in students_db.values():
        if name and name.lower() not in s["name"].lower():
            continue
        if course and course.lower() not in s["course"].lower():
            continue
        if email and email.lower() != s["email"]:
            continue
        if min_age is not None and s["age"] < min_age:
            continue
        if max_age is not None and s["age"] > max_age:
            continue
        results.append(s)
    return {"count": len(results), "students": results}


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    responses={**NOT_FOUND, **INVALID},
    summary="Get a student by id",
)
def get_student(student_id: int = Path(..., gt=0, description="Student id")):
    return _get_or_404(student_id)


@router.put(
    "/{student_id}",
    response_model=MessageResponse,
    responses={**NOT_FOUND, **CONFLICT, **INVALID},
    summary="Replace all of a student's information",
)
def replace_student(student: StudentCreate, student_id: int = Path(..., gt=0)):
    _get_or_404(student_id)
    _ensure_email_unique(student.email, exclude_id=student_id)
    students_db[student_id] = {"id": student_id, **student.model_dump()}
    return {"message": "Student updated successfully", "student": students_db[student_id]}


@router.patch(
    "/{student_id}",
    response_model=MessageResponse,
    responses={**NOT_FOUND, **CONFLICT, **INVALID},
    summary="Update some of a student's information",
)
def update_student(updates: StudentUpdate, student_id: int = Path(..., gt=0)):
    student = _get_or_404(student_id)
    changes = updates.model_dump(exclude_unset=True)
    if changes.get("email"):
        _ensure_email_unique(changes["email"], exclude_id=student_id)
    for field in ("name", "age", "email", "course"):
        if field in changes and changes[field] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{field}' cannot be null",
            )
    student.update(changes)
    return {"message": "Student updated successfully", "student": student}


@router.delete(
    "/{student_id}",
    response_model=MessageResponse,
    responses={**NOT_FOUND, **INVALID},
    summary="Delete a student",
)
def delete_student(student_id: int = Path(..., gt=0)):
    student = _get_or_404(student_id)
    del students_db[student_id]
    return {"message": f"Student '{student['name']}' deleted successfully", "student": student}
