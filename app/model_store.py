"""Load versioned model artifacts with metadata validation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from app.config import OperatingMode, artifact_dir, mode_disclaimer, operating_mode

logger = logging.getLogger("academic_sentinel_ml.model_store")


def model_paths(base: Path | None = None) -> tuple[Path, Path, Path]:
    root = base or artifact_dir()
    return root / "model.joblib", root / "metadata.json", root / "versions"


def load_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {"modelVersion", "features", "targetDefinition", "classes", "positiveLabel"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Model metadata missing required keys: {missing}")
    if not isinstance(metadata["features"], list) or not metadata["features"]:
        raise ValueError("Model metadata features must be a non-empty list")
    return metadata


def normalize_mode(metadata: dict[str, Any]) -> OperatingMode:
    raw = str(metadata.get("operatingMode") or metadata.get("dataMode") or OperatingMode.SYNTHETIC_DEMO.value).lower()
    try:
        return OperatingMode(raw)
    except ValueError:
        # Legacy artifacts without mode are treated as synthetic to avoid accidental production claims.
        return OperatingMode.SYNTHETIC_DEMO


def load_model(base: Path | None = None) -> tuple[Any, dict[str, Any]] | None:
    model_path, metadata_path, _ = model_paths(base)
    if not model_path.is_file() or not metadata_path.is_file():
        return None
    metadata = load_metadata(metadata_path)
    model = joblib.load(model_path)
    artifact_mode = normalize_mode(metadata)
    runtime_mode = operating_mode()
    metadata = {
        **metadata,
        "operatingMode": artifact_mode.value,
        "runtimeOperatingMode": runtime_mode.value,
        "isSynthetic": artifact_mode is OperatingMode.SYNTHETIC_DEMO or bool(metadata.get("isSynthetic", False)),
        "disclaimer": mode_disclaimer(artifact_mode),
    }
    if runtime_mode is OperatingMode.PRODUCTION_PREDICTION and artifact_mode is OperatingMode.SYNTHETIC_DEMO:
        logger.warning("Synthetic artifact loaded while runtime mode is production_prediction; predictions remain non-authoritative")
    return model, metadata


def list_versions(base: Path | None = None) -> list[str]:
    _, _, versions = model_paths(base)
    if not versions.is_dir():
        return []
    return sorted(path.name for path in versions.iterdir() if path.is_dir())
