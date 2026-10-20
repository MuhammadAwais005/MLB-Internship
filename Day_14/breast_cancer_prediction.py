
# ============================================
# BREAST CANCER PREDICTION SYSTEM
# Logistic Regression + GridSearchCV
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 2. LOAD DATASET
# ============================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target, name="target")

# Scikit-learn uses:
# 0 = malignant
# 1 = benign

print("\nDataset shape:", X.shape)

print("\nFirst five rows:")
print(X.head())

print("\nDataset information:")
X.info()

print("\nStatistical summary:")
print(X.describe())

print("\nTarget distribution:")
print(y.map({0: "malignant", 1: "benign"}).value_counts())


# ========================
# 3. SPLIT DATASET
# ========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================
# 4. CREATE BASELINE PIPELINE
# ============================================

# Scaling is fitted only on training data.


baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

# Train the baseline model
baseline_model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    # Treat malignant (class 0) as positive
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, pos_label=0, zero_division=0
        )
    }

    print(f"\n========== {name} ==========")

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            zero_division=0
        )
    )

    return results, predictions




baseline_results, baseline_predictions = evaluate_model(
    baseline_model,
    X_test,
    y_test,
    "Baseline Model"
)


# HYPERPARAMETER TUNING


param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

# Find the best settings using training data
grid_search.fit(X_train, y_train)

print("\n========== GRID SEARCH RESULTS ==========")

print("Best parameters:")
print(grid_search.best_params_)

print(
    "Best mean CV recall (malignant class):",
    round(grid_search.best_score_, 4)
)



# EVALUATE TUNED MODEL


best_model = grid_search.best_estimator_

tuned_results, tuned_predictions = evaluate_model(
    best_model,
    X_test,
    y_test,
    "Tuned Model"
)



# COMPARE BASELINE AND TUNED MODELS

comparison = pd.DataFrame([
    baseline_results,
    tuned_results
])

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))



# CONFUSION MATRIX HEATMAPS


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = [
    ("Baseline Model", baseline_predictions),
    ("Tuned Model", tuned_predictions)
]

for ax, (name, predictions) in zip(axes, models):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.tight_layout()
plt.show()



#COMPARE METRICS VISUALLY


comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(10, 6), ylim=(0, 1))

plt.title("Baseline vs Tuned Model Performance")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
