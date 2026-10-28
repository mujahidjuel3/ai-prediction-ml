"""Private local inference service. No dataset or trained model is included."""
from __future__ import annotations

import json
import os
import hmac
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_DIR = Path(os.environ.get("ML_ARTIFACT_DIR", "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
app = FastAPI(title="Academic Sentinel ML", version="0.1.0")
SERVICE_TOKEN = os.environ.get("ML_SERVICE_TOKEN", "")


def require_service_token(token: str | None) -> None:
    if not SERVICE_TOKEN or token is None or not hmac.compare_digest(token, SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Internal service authentication failed")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | int | str | None]


def load_model() -> tuple[Any, dict[str, Any]] | None:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        return None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    return model, metadata


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    loaded = load_model()
    return {"status": "ready" if loaded else "model_required", "modelAvailable": loaded is not None}


@app.get("/v1/model")
def model_metadata(x_ml_service_token: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(x_ml_service_token)
    loaded = load_model()
    if loaded is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_REQUIRED", "message": "No trained model is available. Supply an approved dataset and run the documented training command."})
    return {"data": loaded[1]}


@app.post("/v1/predict")
def predict(request: PredictionInput, x_ml_service_token: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(x_ml_service_token)
    loaded = load_model()
    if loaded is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_REQUIRED", "message": "No trained model is available. Prediction cannot be performed."})
    model, metadata = loaded
    features = metadata["features"]
    if set(request.features) != set(features):
        raise HTTPException(status_code=422, detail={"code": "FEATURE_SCHEMA_MISMATCH", "expected": features})
    frame = pd.DataFrame([{key: request.features[key] for key in features}], columns=features)
    prediction = model.predict(frame)[0]
    response: dict[str, Any] = {"modelVersion": metadata["modelVersion"], "prediction": str(prediction), "targetDefinition": metadata["targetDefinition"], "probabilityCalibrated": bool(metadata.get("probabilityCalibrated", False))}
    if response["probabilityCalibrated"] and hasattr(model, "predict_proba"):
        response["probabilities"] = {str(label): float(value) for label, value in zip(model.classes_, model.predict_proba(frame)[0], strict=True)}
    return {"data": response}
