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
