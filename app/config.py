"""Runtime configuration for the private Academic Sentinel ML service."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class OperatingMode(str, Enum):
    """Clear separation of synthetic demo, validated, and production prediction modes.

    SYNTHETIC_DEMO — pipeline/integration testing with synthetic fixtures only.
      Never represents actual student risk.
    VALIDATED_MODEL — university-approved dataset trained and reviewed; still not
      automatically enabled for student-facing generation in NestJS.
    PRODUCTION_PREDICTION — reserved for fully approved institutional use.
      NestJS will not enable POST /predictions/generate without separate gates.
    """

    SYNTHETIC_DEMO = "synthetic_demo"
    VALIDATED_MODEL = "validated_model"
    PRODUCTION_PREDICTION = "production_prediction"


def artifact_dir() -> Path:
    return Path(os.environ.get("ML_ARTIFACT_DIR", "artifacts")).resolve()


def service_token() -> str:
    return os.environ.get("ML_SERVICE_TOKEN", "")


def operating_mode() -> OperatingMode:
    raw = (os.environ.get("ML_OPERATING_MODE") or OperatingMode.SYNTHETIC_DEMO.value).strip().lower()
    try:
        return OperatingMode(raw)
    except ValueError as exc:
        raise ValueError(
            "ML_OPERATING_MODE must be one of: synthetic_demo, validated_model, production_prediction"
        ) from exc


def mode_disclaimer(mode: OperatingMode) -> str:
    if mode is OperatingMode.SYNTHETIC_DEMO:
        return (
            "Synthetic demo mode only. Outputs and scores are fixtures for pipeline testing "
            "and must never be presented as actual student risk assessments."
        )
    if mode is OperatingMode.VALIDATED_MODEL:
        return (
            "Validated model mode. A reviewed artifact may be loaded for controlled evaluation. "
            "Student-facing prediction generation remains disabled until NestJS university gates pass."
        )
    return (
        "Production prediction mode is reserved for university-approved use. "
        "NestJS must still enforce dataset, target, horizon, threshold, feature, review, and consent gates."
    )
