from student import Student
from employee import Employee
from car import Car


def main():
    # Multiple objects from the same Student class
    student1 = Student("S001", "Ali", "Computer Science")
    student2 = Student("S002", "Ayesha", "Artificial Intelligence")

    print("--- Students ---")
    student1.display_details()
    student2.display_details()

    # Multiple objects from the same Employee class
    employee1 = Employee("E001", "Usman", "Developer", 70000)
    employee2 = Employee("E002", "Sara", "QA Engineer", 65000)

    print("\n--- Employees ---")
    employee1.display_details()
    employee2.display_details()

    # Multiple objects from the same Car class
    car1 = Car("Toyota", "Corolla", 2022, "White")
    car2 = Car("Honda", "Civic", 2024, "Black")

    print("\n--- Cars ---")
    car1.display_details()
    car2.display_details()


if __name__ == "__main__":
    main()