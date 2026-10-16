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
