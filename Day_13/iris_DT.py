
# Importing Libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading Built-in Iris Dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

print("\nMissing Values:\n", df.isnull().sum())

# Encoding
labelencoder = LabelEncoder()
df["species"] = labelencoder.fit_transform(df["species"])

# Splitting
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(
          y_test, y_pred, target_names=labelencoder.classes_
      )
)

# Testing with New Flower Data
#sepal length, sepal width, petal length, petal width
new_data = [[5.1, 3.5, 1.4, 0.2]]

new_data = scaler.transform(new_data)
prediction = model.predict(new_data)

predicted_species = labelencoder.inverse_transform(prediction)

print("\nPrediction:", predicted_species[0])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=labelencoder.classes_,
    yticklabels=labelencoder.classes_
)
plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Actual vs Predicted Values
results = pd.DataFrame({
    "Actual": labelencoder.inverse_transform(y_test),
    "Predicted": labelencoder.inverse_transform(y_pred)
})

print("\nSample Predictions:\n", results.head(10))






joblib.dump(
    {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    },
    "iris_decision_tree.pkl"
)
