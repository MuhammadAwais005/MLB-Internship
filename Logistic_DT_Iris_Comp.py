
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Loading Existing Trained Models and Results
lr = joblib.load("iris_logistic.pkl")
dt = joblib.load("iris_decision_tree.pkl")

# Evaluation Function
def evaluate_model(name, data):
    y_test = data["y_test"]
    y_pred = data["y_pred"]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
    }

# Comparing Evaluation Metrics
comparison = pd.DataFrame([
    evaluate_model("Logistic Regression", lr),
    evaluate_model("Decision Tree", dt)
])

print("\nModel Evaluation Comparison:")
print(comparison.round(4).to_string(index=False))


# Best Model Based on Accuracy

max_accuracy = comparison["Accuracy"].max()

best_models = comparison[
    comparison["Accuracy"] == max_accuracy
]["Model"].tolist()

if len(best_models) == 1:
    print("\nBest Model Based on Accuracy:", best_models[0])
else:
    print("\nBoth models have the same accuracy!")
    print("Models:", ", ".join(best_models))
    print(f"Accuracy: {max_accuracy:.4f}")
    print("No single best model can be selected based on accuracy alone.")


# Comparing Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name, data in [
    (axes[0], "Logistic Regression", lr),
    (axes[1], "Decision Tree", dt)
]:
    cm = confusion_matrix(data["y_test"], data["y_pred"])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Comparing Metrics Visually
comparison.set_index("Model").plot(
    kind="bar", figsize=(9, 5), ylim=(0, 1.05)
)

plt.title("Iris Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
