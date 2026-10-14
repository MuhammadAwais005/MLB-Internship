# Day-5 — OOP Practice and Library Management System

## Overview

This folder contains Python OOP practice programs and a console-based
Library Management System developed for the MLB-Internship task.

## Project Structure

```text
Day-5/
├── oop_practice/
│   ├── student.py
│   ├── employee.py
│   ├── car.py
│   ├── multiple_objects.py
│   └── inheritance.py
├── library_management/
│   ├── main.py
│   └── library.json
└── README.md
```

## Requirements

- Python 3.9 or newer recommended
- No third-party packages required

## OOP Practice

Run the following commands from the Day-5 directory:

```bash
python oop_practice/student.py
python oop_practice/employee.py
python oop_practice/car.py
python oop_practice/multiple_objects.py
python oop_practice/inheritance.py
```

Topics covered:

- Classes and objects
- Attributes and methods
- Constructors and `self`
- Multiple objects from the same class
- Inheritance and `super()`
- Method overriding

## Library Management System

Run the application:

```bash
python library_management/main.py
```

### Features

1. Add a regular book or e-book
2. View all books
3. Search by ID, title, or author
4. Borrow a book
5. Return a book
6. Exit the application

### Concepts Used

- **OOP:** Book, EBook, and Library classes
- **Inheritance:** EBook inherits from Book
- **Encapsulation:** Borrowing status is managed through methods
- **Method overriding:** EBook customizes `to_dict()` and `display()`
- **JSON:** Records persist between program runs
- **Exception handling:** Invalid inputs, duplicate IDs, missing books,
  malformed JSON, and storage errors are handled
- **Safer file saving:** A temporary file is written before replacing
  the main JSON file

## Testing Checklist

- [ ] Add a regular book
- [ ] Add an e-book
- [ ] View all books
- [ ] Search for a book
- [ ] Borrow an available book
- [ ] Try borrowing it again
- [ ] Return a borrowed book
- [ ] Try returning it again
- [ ] Restart the application and check saved records
- [ ] Test invalid menu choices and invalid file-size input

## Sample Data

`library.json` contains sample records. Keep them for testing, or replace
the file contents with `[]` to start with an empty library.
