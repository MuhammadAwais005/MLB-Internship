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
