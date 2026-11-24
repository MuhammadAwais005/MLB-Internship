"""
Entry point of the Student Management REST API.

Run from the Mini_Project folder:
    uvicorn app.main:app --reload
Swagger UI: http://127.0.0.1:8000/docs
"""

from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routes import students

app = FastAPI(
    title="Student Management REST API",
    description=(
        "Day-34 mini project: add, view, search, update and delete students. "
        "Data is stored in memory (no database yet)."
    ),
    version="1.0.0",
)

app.include_router(students.router)

START_TIME = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Consistent error responses
# Every error comes back as:
#   {"success": false, "status_code": 404, "error": "...", "details": [...]}
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "status_code": exc.status_code, "error": exc.detail, "details": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        # loc looks like ("body", "age") or ("path", "student_id"); drop the first part
        loc = [str(part) for part in err["loc"][1:]] or [str(err["loc"][0])]
        message = err["msg"].removeprefix("Value error, ")
        if err["type"] == "string_pattern_mismatch" and loc[-1] == "email":
            message = "Invalid email address"
        details.append({"field": ".".join(loc), "message": message})
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "status_code": 422,
            "error": "Validation failed",
            "details": details,
        },
    )


# ---------------------------------------------------------------------------
# General routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the Student Management REST API",
        "docs": "/docs",
        "endpoints": {
            "add": "POST /students",
            "view_all": "GET /students",
            "search": "GET /students/search?name=&course=&email=&min_age=&max_age=",
            "view_one": "GET /students/{id}",
            "replace": "PUT /students/{id}",
            "update": "PATCH /students/{id}",
            "delete": "DELETE /students/{id}",
        },
    }


@app.get("/health", tags=["General"])
def health():
    now = datetime.now(timezone.utc)
    return {
        "status": "healthy",
        "timestamp": now.isoformat(),
        "uptime_seconds": round((now - START_TIME).total_seconds(), 2),
        "total_students": len(students.students_db),
    }
