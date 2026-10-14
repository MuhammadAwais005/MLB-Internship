class Employee:
    def __init__(self, employee_id, name, position, salary):
        self.employee_id = employee_id
        self.name = name
        self.position = position
        self.salary = salary

    def display_details(self):
        print(
            f"ID: {self.employee_id} | Name: {self.name} | "
            f"Position: {self.position} | Salary: {self.salary}"
        )

    def give_raise(self, amount):
        if amount <= 0:
            raise ValueError("Raise must be greater than zero.")

        self.salary += amount
        print(f"Updated salary: {self.salary}")


if __name__ == "__main__":
    employee = Employee("E001", "Awais", "Python Intern", 50000)
    employee.display_details()
    employee.give_raise(5000)