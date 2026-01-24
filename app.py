# app.py
import pandas as pd
import streamlit as st
from model import train_save_pipelines, load_pipelines
from preprocessing import preprocess_input

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")

# Load data (once)
df = pd.read_csv("flight_data.csv", low_memory=False)
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
df["is_delayed"] = (df["ARR_DELAY"] >= 15).astype(int)
df["dep_hour"] = df["CRS_DEP_TIME"] // 100
df["arr_hour"] = df["CRS_ARR_TIME"] // 100
df["day_of_week"] = df["FL_DATE"].dt.dayofweek
df["month"] = df["FL_DATE"].dt.month
if "WEATHER" not in df.columns:
    df["WEATHER"] = "Clear"

features = ["AIRLINE","ORIGIN","DEST","WEATHER","dep_hour","arr_hour","day_of_week",
            "month","DISTANCE","TAXI_OUT"]

df_model = df[features + ["ARR_DELAY","is_delayed","FL_DATE"]].dropna()
df_model = df_model.sort_values("FL_DATE")

# Train or load pipelines
try:
    clf_pipeline, reg_pipeline = load_pipelines()
except:
    split_date = df_model["FL_DATE"].quantile(0.8)
    train = df_model[df_model["FL_DATE"] <= split_date]
    X_train = train[features]
    y_class_train = train["is_delayed"]
    y_reg_train = train["ARR_DELAY"]
    clf_pipeline, reg_pipeline = train_save_pipelines(X_train, y_class_train, y_reg_train)

# Streamlit UI
st.title("✈️ Flight Delay Prediction")
st.subheader("🧾 Flight Input")

# ... (your existing Streamlit input code here)
# Example:
airline = st.selectbox("Airline", df_model["AIRLINE"].unique())
# ... other inputs ...
input_df = pd.DataFrame([{
    "AIRLINE": airline,
    # fill other features...
}])

if st.button("Predict"):
    input_df = preprocess_input(input_df)
    delay_min = max(0, reg_pipeline.predict(input_df)[0])
    st.write(f"Predicted Delay: {round(delay_min,2)} minutes")
