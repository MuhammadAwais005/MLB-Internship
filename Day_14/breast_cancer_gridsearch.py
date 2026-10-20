import pandas as pd

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

data = load_breast_cancer()

# Convert features into a Pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column
df["target"] = data.target

# Add readable target labels
df["diagnosis"] = df["target"].map({
    0: "malignant",
    1: "benign"
})

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nTarget class distribution:")
print(df["diagnosis"].value_counts())


# Features: measurements used for prediction
X = df.drop(columns=["target", "diagnosis"])

# Target: the class we want to predict
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


def evaluate_model(model, X_test, y_test, model_name):

    # Predict the test labels
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} ===")

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    # Class 0 is malignant, so use it as the positive class
    print(
        "Precision:",
        round(precision_score(y_test, y_pred, pos_label=0), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_test, y_pred, pos_label=0), 4)
    )

    print(
        "F1-Score:",
        round(f1_score(y_test, y_pred, pos_label=0), 4)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["malignant", "benign"],
            zero_division=0
        )
    )

# Scaling helps Logistic Regression work with
# features that have different numerical ranges.

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

# Train the baseline model
baseline_model.fit(X_train, y_train)

# Evaluate baseline performance
evaluate_model(
    baseline_model,
    X_test,
    y_test,
    "BASELINE MODEL"
)




# Define the settings we want to test
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5, #cross-validation folds 
    scoring="f1",
    n_jobs=-1 
)

# Search for the best settings using training data
grid_search.fit(X_train, y_train)

print("\n=== GRID SEARCH RESULTS ===")

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest mean cross-validation F1-score:")
print(round(grid_search.best_score_, 4))

#Evaluation

best_model = grid_search.best_estimator_

evaluate_model(
    best_model,
    X_test,
    y_test,
    "TUNED MODEL"
)


baseline_pred = baseline_model.predict(X_test)
tuned_pred = best_model.predict(X_test)

baseline_f1 = f1_score(
    y_test, baseline_pred, pos_label=0
)

tuned_f1 = f1_score(
    y_test, tuned_pred, pos_label=0
)

print("\n=== FINAL COMPARISON ===")

print("Baseline F1-score:", round(baseline_f1, 4))
print("Tuned F1-score:", round(tuned_f1, 4))

if tuned_f1 > baseline_f1:
    print("Result: F1-score improved after tuning.")
elif tuned_f1 < baseline_f1:
    print("Result: F1-score decreased after tuning.")
else:
    print("Result: F1-score remained the same.")
