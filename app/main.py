"""Private FastAPI inference service for Academic Sentinel.

Never presents synthetic outputs as actual student risk assessments.
NestJS keeps POST /predictions/generate disabled until university gates pass.
"""
from __future__ import annotations

import hmac
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import OperatingMode, mode_disclaimer, operating_mode, service_token
from app.logging_config import configure_logging, log_extra
from app.model_store import list_versions, load_model
from app.schemas import PredictionInput

configure_logging()
logger = logging.getLogger("academic_sentinel_ml")

app = FastAPI(
    title="Academic Sentinel ML",
    version="0.2.0",
    description="Private ML service. Synthetic demo outputs are never actual student risk assessments.",
)


def require_service_token(token: str | None) -> None:
    expected = service_token()
    if not expected or token is None or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "Internal service authentication failed"})


def error_detail(code: str, message: str, **details: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return body


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    log_extra(logger, logging.WARNING, "request_validation_failed", errors=exc.errors())
    return JSONResponse(status_code=422, content={"detail": error_detail("REQUEST_VALIDATION_FAILED", "Request validation failed", errors=exc.errors())})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    log_extra(logger, logging.ERROR, "unhandled_exception", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": error_detail("INTERNAL_ERROR", "Unexpected ML service error")})


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    mode = operating_mode()
    loaded = load_model()
    model_available = loaded is not None
    status = "ready" if model_available else "model_required"
    response = {
        "status": status,
        "modelAvailable": model_available,
        "operatingMode": mode.value,
        "disclaimer": mode_disclaimer(mode),
        "isSynthetic": mode is OperatingMode.SYNTHETIC_DEMO or (loaded is not None and bool(loaded[1].get("isSynthetic"))),
        "versions": list_versions()[-5:],
    }
    if loaded is not None:
        response["modelVersion"] = loaded[1].get("modelVersion")
        response["artifactOperatingMode"] = loaded[1].get("operatingMode")
    return response


@app.get("/v1/model")
def model_metadata(x_ml_service_token: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(x_ml_service_token)
    loaded = load_model()
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_REQUIRED", "No trained model is available. Supply an approved or synthetic fixture dataset and run the documented training command."),
        )
    return {"data": loaded[1]}


@app.get("/v1/versions")
def model_versions(x_ml_service_token: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(x_ml_service_token)
    return {"data": {"versions": list_versions()}}


@app.post("/v1/predict")
def predict(request: PredictionInput, x_ml_service_token: str | None = Header(default=None)) -> dict[str, Any]:
    require_service_token(x_ml_service_token)
    loaded = load_model()
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_REQUIRED", "No trained model is available. Prediction cannot be performed."),
        )
    model, metadata = loaded
    expected_features = list(metadata["features"])
    provided = set(request.features)
    expected = set(expected_features)
    if provided != expected:
        raise HTTPException(
            status_code=422,
            detail=error_detail(
                "FEATURE_SCHEMA_MISMATCH",
                "Feature keys must exactly match the trained model schema.",
                expected=expected_features,
                missing=sorted(expected - provided),
                unexpected=sorted(provided - expected),
            ),
        )

    frame = pd.DataFrame([{key: request.features[key] for key in expected_features}], columns=expected_features)
    try:
        prediction = model.predict(frame)[0]
    except Exception as exc:  # noqa: BLE001
        log_extra(logger, logging.ERROR, "inference_failed", error=str(exc), modelVersion=metadata.get("modelVersion"))
        raise HTTPException(status_code=500, detail=error_detail("INFERENCE_FAILED", "Model inference failed")) from exc

    artifact_mode = str(metadata.get("operatingMode", OperatingMode.SYNTHETIC_DEMO.value))
    is_synthetic = bool(metadata.get("isSynthetic")) or artifact_mode == OperatingMode.SYNTHETIC_DEMO.value
    response: dict[str, Any] = {
        "modelVersion": metadata["modelVersion"],
        "prediction": str(prediction),
        "targetDefinition": metadata["targetDefinition"],
        "probabilityCalibrated": bool(metadata.get("probabilityCalibrated", False)),
        "operatingMode": artifact_mode,
        "isSynthetic": is_synthetic,
        "disclaimer": metadata.get("disclaimer") or mode_disclaimer(OperatingMode(artifact_mode) if artifact_mode in OperatingMode._value2member_map_ else OperatingMode.SYNTHETIC_DEMO),
    }
    if response["probabilityCalibrated"] and hasattr(model, "predict_proba"):
        classes = getattr(model, "classes_", None)
        if classes is not None:
            response["probabilities"] = {
                str(label): float(value)
                for label, value in zip(classes, model.predict_proba(frame)[0], strict=True)
            }
    log_extra(
        logger,
        logging.INFO,
        "inference_completed",
        modelVersion=response["modelVersion"],
        operatingMode=artifact_mode,
        isSynthetic=is_synthetic,
    )
    return {"data": response}
