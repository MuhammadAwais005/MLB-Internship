"""
Pydantic schemas (data models) for the Student Management API.

Every request body is validated against these models before it reaches
the route functions, so invalid data never gets into the in-memory store.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Reusable field rules -------------------------------------------------------
NAME_RULES = dict(min_length=2, max_length=50, description="Full name (letters and spaces only)")
AGE_RULES = dict(ge=16, le=60, description="Age between 16 and 60")
EMAIL_RULES = dict(
    max_length=100,
    pattern=r"^[\w\.\+\-]+@[\w\-]+(\.[\w\-]+)+$",
    description="A valid email address (must be unique)",
)
COURSE_RULES = dict(min_length=2, max_length=50, description="Enrolled course")
GPA_RULES = dict(ge=0.0, le=4.0, description="GPA on a 4.0 scale")


def _clean_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if not all(ch.isalpha() or ch in " .-'" for ch in value):
        raise ValueError("Name may only contain letters, spaces, dots, hyphens and apostrophes")
    return " ".join(value.split()).title()


def _clean_email(value: Optional[str]) -> Optional[str]:
    return value.lower() if value is not None else value


# Request models -------------------------------------------------------------
class StudentCreate(BaseModel):
    """Body for POST /students and PUT /students/{id} (all fields required except gpa)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
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

    name: str = Field(..., **NAME_RULES)
    age: int = Field(..., **AGE_RULES)
    email: str = Field(..., **EMAIL_RULES)
    course: str = Field(..., **COURSE_RULES)
    gpa: Optional[float] = Field(None, **GPA_RULES)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        return _clean_name(value)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        return _clean_email(value)


class StudentUpdate(BaseModel):
    """Body for PATCH /students/{id}: send only the fields you want to change."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={"example": {"course": "Deep Learning", "gpa": 3.7}},
    )

    name: Optional[str] = Field(None, **NAME_RULES)
    age: Optional[int] = Field(None, **AGE_RULES)
    email: Optional[str] = Field(None, **EMAIL_RULES)
    course: Optional[str] = Field(None, **COURSE_RULES)
    gpa: Optional[float] = Field(None, **GPA_RULES)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        return _clean_name(value)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        return _clean_email(value)

    @model_validator(mode="after")
    def at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update")
        return self


# Response models ------------------------------------------------------------
class StudentResponse(BaseModel):
    """A student as returned by the API."""

    id: int
    name: str
    age: int
    email: str
    course: str
    gpa: Optional[float] = None


class StudentListResponse(BaseModel):
    count: int
    students: list[StudentResponse]


class MessageResponse(BaseModel):
    message: str
    student: Optional[StudentResponse] = None


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    """Shape of every error the API returns."""

    success: bool = False
    status_code: int
    error: str
    details: Optional[list[ErrorDetail]] = None
