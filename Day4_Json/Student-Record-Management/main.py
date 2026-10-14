
import json
import os

FILE_NAME = os.path.join(os.path.dirname(__file__), "students.json")


def load_students():
    """Load student records from the JSON file."""
    if not os.path.exists(FILE_NAME):
        return []

    try:
        with open(FILE_NAME, "r", encoding="utf-8") as file:
            students = json.load(file)

        if not isinstance(students, list):
            print("Error: JSON data must contain a list of students.")
            return []

        for student in students:
            if not isinstance(student, dict) or not all(
                key in student for key in ("id", "name", "age", "course")
            ):
                print("Error: Invalid student record in JSON file.")
                return []

        return students

    except json.JSONDecodeError:
        print("Error: students.json contains invalid JSON.")
        return []
    except OSError as error:
        print(f"File error while loading students: {error}")
        return []


def save_students(students):
    """Save all student records to the JSON file."""
    try:
        with open(FILE_NAME, "w", encoding="utf-8") as file:
            json.dump(students, file, indent=4, ensure_ascii=False)
        print("Records saved successfully.")

    except (OSError, TypeError, ValueError) as error:
        print(f"Error saving records: {error}")


def get_student_by_id(students, student_id):
    """Find a student by ID."""
    for student in students:
        if student["id"] == student_id:
            return student
    return None


def add_student(students):
    """Add a new student."""
    try:
        student_id = input("Enter student ID: ").strip()

        if not student_id:
            print("Student ID cannot be empty.")
            return

        if get_student_by_id(students, student_id):
            print("A student with this ID already exists.")
            return

        name = input("Enter student name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return

        age = int(input("Enter student age: "))
        if age <= 0:
            print("Age must be greater than zero.")
            return

        course = input("Enter student course: ").strip()
        if not course:
            print("Course cannot be empty.")
            return

        student = {
            "id": student_id,
            "name": name,
            "age": age,
            "course": course
        }

        students.append(student)
        save_students(students)
        print("Student added successfully.")

    except ValueError:
        print("Invalid age. Please enter a whole number.")
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")


def view_students(students):
    """Display all student records."""
    if not students:
        print("No student records found.")
        return

    print("\n--- All Students ---")
    for student in students:
        print(
            f"ID: {student['id']} | "
            f"Name: {student['name']} | "
            f"Age: {student['age']} | "
            f"Course: {student['course']}"
        )


def search_student(students):
    """Search for a student by ID."""
    student_id = input("Enter student ID to search: ").strip()
    student = get_student_by_id(students, student_id)

    if student:
        print("\nStudent found:")
        print(f"ID: {student['id']}")
        print(f"Name: {student['name']}")
        print(f"Age: {student['age']}")
        print(f"Course: {student['course']}")
    else:
        print("Student not found.")


def update_student(students):
    """Update an existing student's details."""
    student_id = input("Enter student ID to update: ").strip()
    student = get_student_by_id(students, student_id)

    if not student:
        print("Student not found.")
        return

    print("Press Enter to keep the current value.")

    try:
        name = input(f"Name [{student['name']}]: ").strip()
        age_text = input(f"Age [{student['age']}]: ").strip()
        course = input(f"Course [{student['course']}]: ").strip()

        new_age = student["age"]
        if age_text:
            new_age = int(age_text)
            if new_age <= 0:
                print("Age must be greater than zero.")
                return

        student["name"] = name or student["name"]
        student["age"] = new_age
        student["course"] = course or student["course"]

        save_students(students)
        print("Student updated successfully.")

    except ValueError:
        print("Invalid age. Please enter a whole number.")
    except (EOFError, KeyboardInterrupt):
        print("\nUpdate cancelled.")


def delete_student(students):
    """Delete a student by ID."""
    student_id = input("Enter student ID to delete: ").strip()
    student = get_student_by_id(students, student_id)

    if not student:
        print("Student not found.")
        return

    confirmation = input(
        f"Delete {student['name']}? Type yes to confirm: "
    ).strip().lower()

    if confirmation == "yes":
        students.remove(student)
        save_students(students)
        print("Student deleted successfully.")
    else:
        print("Deletion cancelled.")


def main():
    """Run the student management application."""
    students = load_students()
    print(f"Loaded {len(students)} student record(s).")

    while True:
        print("\n===== Student Record Management System =====")
        print("1. Add Student")
        print("2. View All Students")
        print("3. Search Student")
        print("4. Update Student")
        print("5. Delete Student")
        print("6. Exit")

        try:
            choice = input("Enter your choice (1-6): ").strip()

            if choice == "1":
                add_student(students)
            elif choice == "2":
                view_students(students)
            elif choice == "3":
                search_student(students)
            elif choice == "4":
                update_student(students)
            elif choice == "5":
                delete_student(students)
            elif choice == "6":
                print("Exiting Student Management System.")
                break
            else:
                print("Invalid choice. Please select 1-6.")

        except (EOFError, KeyboardInterrupt):
            print("\nProgram stopped.")
            break


if __name__ == "__main__":
    main()
