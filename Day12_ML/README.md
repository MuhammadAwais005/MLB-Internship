# Student Score Prediction System — follow-along guide

## Setup
Install Python 3.10 or newer. Extract this project, open its folder in VS Code, and open a terminal in that folder.

```bash
python -m venv .venv
```

On Windows Command Prompt:
```bat
.venv\Scripts\activate.bat
```
On macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:
```bash
python -m pip install -r requirements.txt
python main.py --show-plot --predict
```
For evaluation only: `python main.py`.
The dataset is included in `data/student_performance.csv`.

## Design decisions
The file contains 20 students, 11 columns, no missing values, and no duplicate rows. Average_Score already equals the arithmetic mean of Python, Mathematics, Statistics, and Machine_Learning.

Target: Average_Score. Features: Age, Program, Attendance. Student_ID and Name are identifiers. The four subjects are excluded because they directly determine the target. Performance is excluded because its origin is undocumented and it may be derived from grades. This assumes attendance is available at prediction time; use attendance measured before the prediction date in a real deployment.

Program uses one-hot encoding because AI, DS, and SE have no numeric ranking. Dropping one category avoids redundant dummy columns with the regression intercept. Numeric features are standardized; this is optional for ordinary unregularized linear regression, but gives consistent scales. Encoding, imputation, and scaling are fitted ONLY on training data through a Pipeline. The target remains in score units. Imputation is included for future missing input values, although this dataset needs none.

The split is 80% training (16 rows) and 20% testing (4 rows), with random_state=42 for reproducibility. New-student predictions use this same trained pipeline. Predictions are not clipped or rounded before evaluation.

## Modules in order
Create each file below in the same project folder. Copy its complete code. You only run main.py; it imports the other modules.

## Module 1: requirements.txt

Dependency list.

```text
pandas>=2.0,<3.0
numpy>=1.24,<3.0
scikit-learn>=1.3,<2.0
matplotlib>=3.7,<4.0
```

## Module 2: preprocessing.py

Load the CSV, verify or create Average_Score, select X/y, split, and define preprocessing.

```python
"""Load data, create the target, split data, and define preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["Age", "Program", "Attendance"]
SUBJECTS = ["Python", "Mathematics", "Statistics", "Machine_Learning"]


def load_data(path):
    df = pd.read_csv(path)
    required = FEATURES + SUBJECTS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in ["Age", "Attendance"] + SUBJECTS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[SUBJECTS].isna().any().any():
        raise ValueError("Subject scores must be complete to calculate the target.")
    calculated_average = df[SUBJECTS].mean(axis=1)
    if "Average_Score" not in df.columns:
        df["Average_Score"] = calculated_average
    else:
        df["Average_Score"] = pd.to_numeric(df["Average_Score"], errors="raise")
        if not np.allclose(df["Average_Score"], calculated_average):
            raise ValueError("Average_Score does not match the four-subject mean.")
    return df


def split_data(df):
    # Exclude identifiers, subject scores, and the possibly outcome-derived label.
    X = df[FEATURES].copy()
    y = df["Average_Score"].copy()
    return train_test_split(X, y, test_size=0.20, random_state=42)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, ["Age", "Attendance"]),
        ("categorical", categorical, ["Program"]),
    ])
```

## Module 3: model.py

Fit preprocessing and Linear Regression on the training set.

```python
"""Train preprocessing and linear regression together to prevent leakage."""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessor


def train_model(X_train, y_train):
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    return model
```

## Module 4: evaluation.py

Compute MAE, MSE, R² and construct the comparison table.

```python
"""Evaluate unrounded predictions and build the comparison table."""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_test, predictions):
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }
    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions,
        "Error (Actual - Predicted)": y_test - predictions,
    })
    comparison.index.name = "Dataset_Row_Index"
    return metrics, comparison
```

## Module 5: visualization.py

Plot actual scores on the x-axis and predictions on the y-axis; the dashed line represents perfect predictions.

```python
"""Save an Actual vs Predicted scatter plot."""
import matplotlib.pyplot as plt


def plot_predictions(y_test, predictions, output_path, show=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, color="royalblue", s=80, label="Test students")
    low = min(min(y_test), min(predictions)) - 2
    high = max(max(y_test), max(predictions)) + 2
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual Average Score", ylabel="Predicted Average Score",
           title="Student Scores: Actual vs Predicted", xlim=(low, high),
           ylim=(low, high))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
```

## Module 6: main.py

Connect all modules, save results, and optionally accept a new student through terminal input.

```python
"""Run: python main.py --show-plot --predict"""
import argparse
import json
from pathlib import Path
import pandas as pd
from preprocessing import load_data, split_data
from model import train_model
from evaluation import evaluate_model
from visualization import plot_predictions

BASE = Path(__file__).resolve().parent


def predict_student(model, df):
    programs = sorted(df["Program"].dropna().unique())
    try:
        age = float(input("Student age: "))
        program = input(f"Program ({', '.join(programs)}): ").strip().upper()
        attendance = float(input("Attendance percentage (0-100): "))
        if not 0 < age <= 120 or not 0 <= attendance <= 100:
            raise ValueError("Enter a valid age and attendance percentage.")
        if program not in programs:
            raise ValueError(f"Program must be one of {programs}.")
        student = pd.DataFrame([{
            "Age": age, "Program": program, "Attendance": attendance,
        }])
        score = model.predict(student)[0]
        print(f"Predicted Average Score: {score:.2f}")
        if score < 0 or score > 100:
            print("This extrapolated prediction falls outside the score range.")
    except ValueError as error:
        print(f"Invalid input: {error}")


def main():
    parser = argparse.ArgumentParser(description="Student Score Prediction System")
    parser.add_argument("--data", type=Path, default=BASE / "data/student_performance.csv")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--predict", action="store_true", help="Enter a new student")
    args = parser.parse_args()
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    metrics, comparison = evaluate_model(y_test, predictions)
    print(f"Total students: {len(df)} | Training: {len(X_train)} | Testing: {len(X_test)}")
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nActual vs Predicted:")
    print(comparison.round(2).to_string())
    output = BASE / "outputs"
    output.mkdir(exist_ok=True)
    comparison.to_csv(output / "actual_vs_predicted.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_predictions(y_test, predictions, output / "actual_vs_predicted.png", args.show_plot)
    print(f"\nResults saved in: {output}")
    print("Only four test students: these metrics are not a reliable generalization estimate.")
    if args.predict:
        predict_student(model, df)


if __name__ == "__main__":
    main()
```

## Verified results on your dataset

| Metric | Value |
| --- | ---: |
| MAE | 3.4866 |
| MSE | 17.9018 |
| R² | 0.7397 |

MAE means the absolute prediction error averages about 3.49 score points on these four test students. MSE measures squared error. R² compares the model against predicting the test-set mean; 1 is perfect, 0 equals that baseline, and negative values are possible. R² is not an accuracy percentage.

| Actual | Predicted |
| ---: | ---: |
| 85.75 | 88.86 |
| 97.25 | 96.43 |
| 89.50 | 86.84 |
| 74.25 | 81.61 |

The outputs folder contains metrics.json, actual_vs_predicted.csv, and actual_vs_predicted.png. Only four test observations means results can change considerably with a different split; this is an educational demonstration, not evidence of reliable deployment performance.
