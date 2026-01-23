import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Flight Delay Prediction",
    layout="wide"
)
# =========================
# GLOBAL UI STYLING
# =========================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ✈️ Flight Delay Predictor")
    st.write("**ML-powered delay estimation**")
    st.divider()
    st.write("🔹 Random Forest Classifier & Regressor")
    st.write("🔹 Time-aware train-test split")
    st.write("🔹 Binary delay classification (15+ min)")
    st.write("🔹 Regression for delay minutes")
    st.divider()
    st.caption("Hackathon-ready ML Dashboard")

# =========================
# TITLE
# =========================
st.title("✈️ Flight Delay Prediction System")
st.caption("Predict whether a flight will be delayed and estimate delay duration using Machine Learning")

st.divider()

# =========================
# LOAD DATA
# =========================
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

num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].fillna(0)

features = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "WEATHER",
    "dep_hour",
    "arr_hour",
    "day_of_week",
    "month",
    "DISTANCE",
    "TAXI_OUT"
]

df_model = df[features + ["ARR_DELAY", "is_delayed", "FL_DATE"]].dropna()

# =========================
# ENCODING
# =========================
encoders = {}
for col in ["AIRLINE", "ORIGIN", "DEST", "WEATHER"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

df_model = df_model.sort_values("FL_DATE")

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(df_model, features):
    split_date = df_model["FL_DATE"].quantile(0.8)

    train = df_model[df_model["FL_DATE"] <= split_date]
    test = df_model[df_model["FL_DATE"] > split_date]

    X_train = train[features]
    y_class_train = train["is_delayed"]
    y_reg_train = train["ARR_DELAY"]

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    clf.fit(X_train, y_class_train)
    reg.fit(X_train, y_reg_train)

    return clf, reg

clf, reg = train_models(df_model, features)

# =========================
# FLIGHT INPUT
# =========================
st.subheader("🧾 Flight Input")

col1, col2, col3 = st.columns(3)

airline = col1.selectbox("Airline", encoders["AIRLINE"].classes_)
origin = col1.selectbox("Origin Airport", encoders["ORIGIN"].classes_)
dest = col1.selectbox("Destination Airport", encoders["DEST"].classes_)

dep_time = col2.time_input("Scheduled Departure Time")
arr_time = col2.time_input("Scheduled Arrival Time")
distance = col2.number_input("Distance (km)", value=1700)

day = col3.selectbox(
    "Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

month = col3.selectbox(
    "Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"]
)

weather = col3.selectbox("Weather", encoders["WEATHER"].classes_)
taxi_out = col3.number_input("Taxi-Out Time (min)", value=18)

day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
             "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

input_df = pd.DataFrame([{
    "AIRLINE": encoders["AIRLINE"].transform([airline])[0],
    "ORIGIN": encoders["ORIGIN"].transform([origin])[0],
    "DEST": encoders["DEST"].transform([dest])[0],
    "WEATHER": encoders["WEATHER"].transform([weather])[0],
    "dep_hour": dep_time.hour,
    "arr_hour": arr_time.hour,
    "day_of_week": day_map[day],
    "month": month_map[month],
    "DISTANCE": distance,
    "TAXI_OUT": taxi_out
}])

# =========================
# PREDICTION OUTPUT
# =========================
if st.button("🚀 Predict Flight Delay"):
    minutes = max(0, reg.predict(input_df)[0])

    st.markdown(
        f"""
        <div class="card">
            ⏱ Predicted Delay: {round(minutes,2)} minutes
        </div>
        """,
        unsafe_allow_html=True
    )

    if minutes < 15:
        st.success("✅ Flight Will Be On Time")
    elif minutes < 60:
        st.warning("⚠️ Minor Delay Expected")
    else:
        st.error("❌ Significant Delay Expected")

st.divider()

# =========================
# MODEL EVALUATION
# =========================
st.subheader("📈 Model Evaluation")

split_date = df_model["FL_DATE"].quantile(0.8)
test = df_model[df_model["FL_DATE"] > split_date]

X_test = test[features]
y_test = test["is_delayed"]

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["On Time", "Delayed"],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose().round(2)

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Classifier Accuracy", f"{accuracy*100:.2f}%")
    st.metric("F1 Score", f"{f1:.2f}")

with col2:
    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)

st.divider()

# =========================
# EDA SECTION
# =========================
st.subheader("📊 Flight Delay Analysis")

df_eda = df[df["ARR_DELAY"].between(-30, 180)]

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(
    ["By Airline", "By Airport", "By Time", "By Weather"]
)

with eda_tab1:
    st.markdown("#### ✈️ Average Delay by Airline")
    airline_delays = df_eda.groupby("AIRLINE")["ARR_DELAY"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=airline_delays.index, y=airline_delays.values, ax=ax)
    st.pyplot(fig)

with eda_tab2:
    st.markdown("#### 🏢 Average Delay by Origin Airport")
    airport_delays = df_eda.groupby("ORIGIN")["ARR_DELAY"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(x=airport_delays.index, y=airport_delays.values, ax=ax)
    st.pyplot(fig)

with eda_tab3:
    st.markdown("#### ⏰ Delay Patterns Over Time")
    fig, axs = plt.subplots(1,3,figsize=(20,5))
    sns.boxplot(x="dep_hour", y="ARR_DELAY", data=df_eda, ax=axs[0])
    sns.boxplot(x="day_of_week", y="ARR_DELAY", data=df_eda, ax=axs[1])
    sns.boxplot(x="month", y="ARR_DELAY", data=df_eda, ax=axs[2])
    st.pyplot(fig)

with eda_tab4:
    st.markdown("#### 🌦 Delay by Weather Condition")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x="WEATHER", y="ARR_DELAY", data=df_eda, ax=ax)
    st.pyplot(fig)

st.divider()
st.caption("✈️ Flight Delay Prediction System | Built with Streamlit & Machine Learning")
