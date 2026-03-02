from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import boto3
from io import BytesIO
from preprocessing import preprocessor

# ---------- S3 CONFIG ----------
BUCKET_NAME = "predict"
CLF_FILE = "clf_pipeline.pkl"
REG_FILE = "reg_pipeline.pkl"

s3 = boto3.client("s3")

def load_pipelines():
    clf_obj = s3.get_object(Bucket=BUCKET_NAME, Key=CLF_FILE)
    reg_obj = s3.get_object(Bucket=BUCKET_NAME, Key=REG_FILE)

    clf = joblib.load(BytesIO(clf_obj["Body"].read()))
    reg = joblib.load(BytesIO(reg_obj["Body"].read()))

    return clf, reg