class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"My name is {self.name}, and I am {self.age}.")

    def role(self):
        print("I am a person.")


class Student(Person):
    def __init__(self, name, age, student_id, course):
        super().__init__(name, age)
        self.student_id = student_id
        self.course = course

    def role(self):  # Method overriding
        print(f"I am a student studying {self.course}.")

    def study(self):
        print(f"{self.name} is studying.")


class Teacher(Person):
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.subject = subject

    def role(self):  # Method overriding
        print(f"I am a teacher of {self.subject}.")

    def teach(self):
        print(f"{self.name} is teaching {self.subject}.")


if __name__ == "__main__":
    student = Student("Ali", 21, "S001", "Computer Science")
    teacher = Teacher("Hina", 32, "Python")

    student.introduce()  # Inherited method
    student.role()       # Overridden method
    student.study()

    print()

    teacher.introduce()  # Inherited method
    teacher.role()       # Overridden method
    teacher.teach()