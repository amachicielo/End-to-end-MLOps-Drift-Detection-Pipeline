from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = "models/model.joblib"
app = FastAPI(title="Bank Marketing Model API", version="1.0.0")


@lru_cache(maxsize=1)
def load_bundle():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run the training pipeline first.",
        ) from error


@app.get("/")
def root():
    return {"message": "MLOps drift detection API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict[str, Any]):
    bundle = load_bundle()
    required = bundle["feature_columns"]
    missing = [column for column in required if column not in payload]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    frame = pd.DataFrame([{column: payload[column] for column in required}])
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {"prediction": prediction, "probability": round(probability, 6)}
