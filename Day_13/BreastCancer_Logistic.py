import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Loading Dataset "Breast Cancer" Kaggle downloaded
df = pd.read_csv(r"C:\Users\92348\Desktop\MLBench\Day_13\data.csv")
print(df.shape)
print("\nMissing values in dataset:\n", df.isnull().sum())
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
print(df.isnull().sum())

#encoding
labelencoder = LabelEncoder()
df["diagnosis"] = labelencoder.fit_transform(df["diagnosis"])

#splitting
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

#prediction and evaluation
y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))


#testing by new data out put should be Malignant (Cancerous)

new_data = [[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
             1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
             0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654,
             0.4601, 0.1189]]

prediction = model.predict(new_data)


if prediction[0] == 1:
    print(" Prediction: Malignant (Cancerous)")
else:
    print(" Prediction: Benign (Non-Cancerous)")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
