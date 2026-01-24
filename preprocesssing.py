# preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ["AIRLINE", "ORIGIN", "DEST", "WEATHER"]
numerical_features = ["dep_hour","arr_hour","day_of_week","month","DISTANCE","TAXI_OUT"]

# ColumnTransformer for pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

def preprocess_input(df):
    """
    Apply any manual feature engineering before the pipeline (if needed)
    """
    df = df.copy()
    df["dep_hour"] = df["dep_hour"]
    df["arr_hour"] = df["arr_hour"]
    df["day_of_week"] = df["day_of_week"]
    df["month"] = df["month"]
    df["DISTANCE"] = df["DISTANCE"]
    df["TAXI_OUT"] = df["TAXI_OUT"]
    return df
