import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ["AIRLINE", "ORIGIN", "DEST", "WEATHER"]
numerical_features = ["dep_hour","arr_hour","day_of_week","month","DISTANCE","TAXI_OUT"]

# -------- Pipeline Preprocessor --------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# -------- Safe Input Preprocessing --------
def preprocess_input(df):
    """
    Ensures input has all required columns in correct format
    """
    df = df.copy()

    # Add missing categorical columns
    for col in categorical_features:
        if col not in df.columns:
            df[col] = "Unknown"

    # Add missing numerical columns
    for col in numerical_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[categorical_features + numerical_features]

    return df