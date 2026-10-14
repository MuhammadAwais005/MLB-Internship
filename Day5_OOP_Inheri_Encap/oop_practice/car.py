class Car:
    def __init__(self, brand, model, year, color):
        self.brand = brand
        self.model = model
        self.year = year
        self.color = color
        self.is_running = False

    def start(self):
        if self.is_running:
            print("The car is already running.")
        else:
            self.is_running = True
            print(f"{self.brand} {self.model} started.")

    def stop(self):
        self.is_running = False
        print("The car has stopped.")

    def display_details(self):
        print(
            f"{self.year} {self.color} {self.brand} {self.model}"
        )


if __name__ == "__main__":
    car = Car("Toyota", "Corolla", 2022, "White")
    car.display_details()
    car.start()
    car.stop()