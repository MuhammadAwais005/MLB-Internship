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
