# model.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from preprocessing import preprocessor, categorical_features, numerical_features

# Create classifier pipeline
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
    ))
])

# Create regressor pipeline
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    ))
])

def train_save_pipelines(X_train, y_class_train, y_reg_train):
    clf_pipeline.fit(X_train, y_class_train)
    reg_pipeline.fit(X_train, y_reg_train)
    joblib.dump(clf_pipeline, "clf_pipeline.pkl")
    joblib.dump(reg_pipeline, "reg_pipeline.pkl")
    return clf_pipeline, reg_pipeline

def load_pipelines():
    clf = joblib.load("clf_pipeline.pkl")
    reg = joblib.load("reg_pipeline.pkl")
    return clf, reg
