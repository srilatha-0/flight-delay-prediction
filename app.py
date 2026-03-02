import streamlit as st
import pandas as pd
import boto3
from model import load_pipelines
from preprocessing import preprocess_input

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")

# ---------- S3 CONFIG ----------
BUCKET_NAME = "predict"
DATA_FILE = "flight_data.csv"

s3 = boto3.client("s3")

# ---------- Load dataset from S3 ----------
@st.cache_data
def load_data():
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=DATA_FILE)
    df = pd.read_csv(obj["Body"], low_memory=False)
    return df

df = load_data()

# ---------- Basic Cleaning ----------
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]

df["dep_hour"] = df["CRS_DEP_TIME"] // 100
df["arr_hour"] = df["CRS_ARR_TIME"] // 100
df["day_of_week"] = df["FL_DATE"].dt.dayofweek
df["month"] = df["FL_DATE"].dt.month

if "WEATHER" not in df.columns:
    df["WEATHER"] = "Clear"

features = ["AIRLINE","ORIGIN","DEST","WEATHER",
            "dep_hour","arr_hour","day_of_week",
            "month","DISTANCE","TAXI_OUT"]

df_model = df[features].dropna()

# ---------- Load ML Pipelines ----------
clf_pipeline, reg_pipeline = load_pipelines()

# ---------- UI ----------
st.title("✈️ Flight Delay Prediction (Cloud Version)")

airline = st.selectbox("Airline", df_model["AIRLINE"].unique())
origin = st.selectbox("Origin", df_model["ORIGIN"].unique())
dest = st.selectbox("Destination", df_model["DEST"].unique())

input_df = pd.DataFrame([{
    "AIRLINE": airline,
    "ORIGIN": origin,
    "DEST": dest,
    "WEATHER": "Clear",
    "dep_hour": 10,
    "arr_hour": 12,
    "day_of_week": 2,
    "month": 5,
    "DISTANCE": 500,
    "TAXI_OUT": 15
}])

if st.button("Predict"):
    input_df = preprocess_input(input_df)
    delay_min = max(0, reg_pipeline.predict(input_df)[0])
    st.success(f"Predicted Delay: {round(delay_min,2)} minutes")