
# =====================================================
# IRIS FLOWER CLASSIFICATION - MODEL COMPARISON
# Logistic Regression vs Decision Tree
# =====================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# LOAD PREVIOUSLY SAVED MODEL RESULTS


lr = joblib.load("iris_logistic.pkl")
dt = joblib.load("iris_decision_tree.pkl")

print("Models and saved test results loaded successfully!")

def evaluate_model(name, data):
    y_test = data["y_test"]
    y_pred = data["y_pred"]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )
    }


comparison = pd.DataFrame([
    evaluate_model("Logistic Regression", lr),
    evaluate_model("Decision Tree", dt)
])

print("\n========== TEST SET COMPARISON ==========")
print(comparison.round(4).to_string(index=False))


#COMPARE CONFUSION MATRICES


iris = load_iris()
class_names = iris.target_names

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name, data in [
    (axes[0], "Logistic Regression", lr),
    (axes[1], "Decision Tree", dt)
]:
    cm = confusion_matrix(
        data["y_test"],
        data["y_pred"],
        labels=[0, 1, 2]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")

plt.suptitle("Confusion Matrix Comparison")
plt.tight_layout()
plt.show()


# VISUAL COMPARISON OF EVALUATION METRICS


comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1.05)
)

plt.title("Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# CLASSIFICATION REPORTS


print("\n========== LOGISTIC REGRESSION REPORT ==========")

print(classification_report(
    lr["y_test"],
    lr["y_pred"],
    labels=[0, 1, 2],
    target_names=class_names,
    zero_division=0
))

print("\n========== DECISION TREE REPORT ==========")

print(classification_report(
    dt["y_test"],
    dt["y_pred"],
    labels=[0, 1, 2],
    target_names=class_names,
    zero_division=0
))


# FIVE-FOLD CROSS-VALIDATION


X = iris.data
y = iris.target

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

cv_results = []

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    cv_results.append({
        "Model": name,
        "Mean CV Accuracy": mean_accuracy,
        "Std Deviation": std_accuracy
    })

    print(f"\n{name}")
    print("Fold Scores:", scores.round(4))
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")

cv_comparison = pd.DataFrame(cv_results)

print("\n========== CROSS-VALIDATION SUMMARY ==========")
print(cv_comparison.round(4).to_string(index=False))


#FINAL MODEL COMPARISON
best_cv_accuracy = cv_comparison["Mean CV Accuracy"].max()

best_models = cv_comparison[
    (best_cv_accuracy - cv_comparison["Mean CV Accuracy"]).abs()
    < 1e-9
]["Model"].tolist()

print("\n========== FINAL CONCLUSION ==========")

if len(best_models) == 1:
    print("Best average cross-validation accuracy:", best_models[0])
else:
    print("Both models have the same mean cross-validation accuracy.")
    print("Models:", ", ".join(best_models))
    print("Consider consistency, interpretability, and model complexity.")

print("\nNote: Cross-validation evaluates fresh model instances.")
print("The saved models were used for the test-set comparison.")
