
import json
import os
from pathlib import Path

DATA_FILE = Path(__file__).with_name("library.json")


class Book:
    def __init__(self, book_id, title, author, is_borrowed=False):
        self.book_id = book_id
        self.title = title
        self.author = author
        self._is_borrowed = is_borrowed

    @property
    def is_borrowed(self):
        return self._is_borrowed

    def borrow(self):
        if self._is_borrowed:
            raise ValueError("This book is already borrowed.")
        self._is_borrowed = True

    def return_book(self):
        if not self._is_borrowed:
            raise ValueError("This book is not currently borrowed.")
        self._is_borrowed = False

    def to_dict(self):
        return {
            "type": "book",
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "is_borrowed": self.is_borrowed
        }

    def display(self):
        status = "Borrowed" if self.is_borrowed else "Available"
        print(
            f"ID: {self.book_id} | Title: {self.title} | "
            f"Author: {self.author} | Status: {status}"
        )


class EBook(Book):
    def __init__(
        self, book_id, title, author, file_size_mb,
        is_borrowed=False
    ):
        super().__init__(book_id, title, author, is_borrowed)

        if file_size_mb <= 0:
            raise ValueError("File size must be greater than zero.")

        self.file_size_mb = file_size_mb

    def to_dict(self):
        data = super().to_dict()
        data["type"] = "ebook"
        data["file_size_mb"] = self.file_size_mb
        return data

    def display(self):
        super().display()
        print(f"   File size: {self.file_size_mb} MB")


class Library:
    def __init__(self, filename=DATA_FILE):
        self.filename = Path(filename)
        self.books = self.load_books()

    def load_books(self):
        if not self.filename.exists():
            return []

        try:
            with self.filename.open("r", encoding="utf-8") as file:
                records = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in library file at line {error.lineno}."
            ) from error

        if not isinstance(records, list):
            raise ValueError("The JSON file must contain a list.")

        books = []
        seen_ids = set()

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each book must be a JSON object.")

            book_id = record.get("book_id")
            title = record.get("title")
            author = record.get("author")
            borrowed = record.get("is_borrowed", False)
            book_type = record.get("type", "book")

            if not all(
                isinstance(value, str) and value.strip()
                for value in (book_id, title, author)
            ):
                raise ValueError("Book ID, title, and author are required.")

            if book_id in seen_ids:
                raise ValueError(f"Duplicate book ID: {book_id}")

            if type(borrowed) is not bool:
                raise ValueError("Invalid borrowing status in JSON.")

            if book_type == "book":
                book = Book(book_id, title, author, borrowed)
            elif book_type == "ebook":
                book = EBook(
                    book_id,
                    title,
                    author,
                    record.get("file_size_mb"),
                    borrowed
                )
            else:
                raise ValueError(f"Unknown book type: {book_type}")

            books.append(book)
            seen_ids.add(book_id)

        return books

    def save_books(self):
        temp_file = self.filename.with_suffix(".tmp")

        try:
            with temp_file.open("w", encoding="utf-8") as file:
                json.dump(
                    [book.to_dict() for book in self.books],
                    file,
                    indent=4,
                    ensure_ascii=False
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temp_file, self.filename)

        except (OSError, TypeError, ValueError) as error:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError(f"Could not save books: {error}") from error

    def find_by_id(self, book_id):
        for book in self.books:
            if book.book_id.casefold() == book_id.casefold():
                return book
        return None

    def add_book(self, book):
        if self.find_by_id(book.book_id):
            raise ValueError("A book with this ID already exists.")

        self.books.append(book)
        try:
            self.save_books()
        except OSError:
            self.books.remove(book)
            raise

    def search_books(self, query):
        query = query.casefold()
        return [
            book for book in self.books
            if query in book.book_id.casefold()
            or query in book.title.casefold()
            or query in book.author.casefold()
        ]

    def view_books(self):
        if not self.books:
            print("The library is empty.")
            return

        print("\n--- All Books ---")
        for book in self.books:
            book.display()

    def borrow_book(self, book_id):
        book = self.find_by_id(book_id)

        if book is None:
            raise ValueError("Book not found.")

        book.borrow()
        try:
            self.save_books()
        except OSError:
            book.return_book()
            raise

        print(f"Borrowed successfully: {book.title}")

    def return_book(self, book_id):
        book = self.find_by_id(book_id)

        if book is None:
            raise ValueError("Book not found.")

        book.return_book()
        try:
            self.save_books()
        except OSError:
            book.borrow()
            raise

        print(f"Returned successfully: {book.title}")


def required_text(prompt):
    value = input(prompt).strip()
    if not value:
        raise ValueError("This field cannot be empty.")
    return value


def positive_float(prompt):
    try:
        value = float(input(prompt))
    except ValueError as error:
        raise ValueError("Please enter a valid number.") from error

    if value <= 0:
        raise ValueError("Enter a number greater than zero.")

    return value


def main():
    try:
        library = Library()
    except (OSError, ValueError) as error:
        print(f"Could not load library: {error}")
        return

    print(f"Loaded {len(library.books)} book(s).")

    while True:
        print("\n===== Library Management System =====")
        print("1. Add a book")
        print("2. View all books")
        print("3. Search for a book")
        print("4. Borrow a book")
        print("5. Return a book")
        print("6. Exit")

        try:
            choice = input("Choose an option (1-6): ").strip()

            if choice == "1":
                book_id = required_text("Book ID: ")
                title = required_text("Title: ")
                author = required_text("Author: ")

                print("1. Regular book")
                print("2. E-book")
                kind = input("Book type (1-2): ").strip()

                if kind == "1":
                    book = Book(book_id, title, author)
                elif kind == "2":
                    size = positive_float("File size in MB: ")
                    book = EBook(book_id, title, author, size)
                else:
                    print("Invalid book type.")
                    continue

                library.add_book(book)
                print("Book added successfully.")

            elif choice == "2":
                library.view_books()

            elif choice == "3":
                query = required_text("Search by ID, title, or author: ")
                results = library.search_books(query)

                if results:
                    for book in results:
                        book.display()
                else:
                    print("No matching books found.")

            elif choice == "4":
                book_id = required_text("Book ID to borrow: ")
                library.borrow_book(book_id)

            elif choice == "5":
                book_id = required_text("Book ID to return: ")
                library.return_book(book_id)

            elif choice == "6":
                print("Goodbye!")
                break

            else:
                print("Invalid option. Choose 1-6.")

        except ValueError as error:
            print(f"Input or operation error: {error}")

        except OSError as error:
            print(f"File/storage error: {error}")

        except (EOFError, KeyboardInterrupt):
            print("\nProgram stopped.")
            break


if __name__ == "__main__":
    main()
