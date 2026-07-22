import os
import urllib.request
import zipfile

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = "data/bank.csv"
MODEL_PATH = "models/model.joblib"


def ensure_dataset(dataset_path=DATASET_PATH):
    if os.path.exists(dataset_path):
        return dataset_path
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    archive_path = "data/bank.zip"
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        archive_path,
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall("data")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Expected dataset at {dataset_path} after extraction.")
    return dataset_path


def build_pipeline(frame):
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_and_save_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    dataset_path = ensure_dataset(dataset_path)
    frame = pd.read_csv(dataset_path, sep=";").dropna()
    features = frame.drop(columns="y")
    target = (frame["y"] == "yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline(features)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    mlflow.set_experiment("Bank_Marketing_Drift_Demo")
    with mlflow.start_run():
        mlflow.log_params({"model": "RandomForestClassifier", "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "feature_columns": list(features.columns)},
        model_path,
    )
    return metrics


if __name__ == "__main__":
    print(train_and_save_model())
