"""Request/response schemas with strict input validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | int | str | None] = Field(min_length=1)


class HealthLiveResponse(BaseModel):
    status: str


class HealthReadyResponse(BaseModel):
    status: str
    modelAvailable: bool
    operatingMode: str
    disclaimer: str
    isSynthetic: bool


class ModelMetadataResponse(BaseModel):
    data: dict[str, Any]


class PredictionResponseData(BaseModel):
    modelVersion: str
    prediction: str
    targetDefinition: str
    probabilityCalibrated: bool
    operatingMode: str
    isSynthetic: bool
    disclaimer: str
    probabilities: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    data: PredictionResponseData


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
