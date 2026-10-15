class Student:
    def __init__(self, student_id, name, course):
        self.student_id = student_id
        self.name = name
        self.course = course

    def introduce(self):
        print(f"My name is {self.name}.")
        print(f"I study {self.course}.")

    def display_details(self):
        print(
            f"ID: {self.student_id} | "
            f"Name: {self.name} | Course: {self.course}"
        )


if __name__ == "__main__":
    student = Student("S001", "Awais", "Computer Science")
    student.introduce()
    student.display_details()