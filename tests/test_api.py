from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fastapi.testclient import TestClient

from train import train


def _train_synthetic(artifact_dir: Path, synthetic_csv: Path) -> dict:
    return train(
        Namespace(
            csv=synthetic_csv,
            target="outcome",
            target_definition="synthetic fixture fail by term end",
            features="attendance,prior_gpa,assessment_avg",
            positive_label="1",
            student_column="student",
            time_column="term",
            unique_key="student,term",
            output=artifact_dir,
            seed=2026,
            test_size=0.25,
            data_mode="synthetic_demo",
            i_confirm_university_approvals=False,
        )
    )


def test_health_endpoints_without_model(service_token, artifact_dir, monkeypatch):
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(artifact_dir))
    from app.main import app

    client = TestClient(app)
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready").json()
    assert ready["status"] == "model_required"
    assert ready["modelAvailable"] is False
    assert ready["operatingMode"] == "synthetic_demo"
    assert "never" in ready["disclaimer"].lower() or "Synthetic" in ready["disclaimer"]


def test_predict_requires_token_and_exact_feature_schema(service_token, artifact_dir, synthetic_csv, monkeypatch):
    metadata = _train_synthetic(artifact_dir, synthetic_csv)
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(artifact_dir))
    from importlib import reload
    import app.main as main

    reload(main)
    client = TestClient(main.app)

    denied = client.post("/v1/predict", json={"features": {"attendance": 80}})
    assert denied.status_code == 401

    mismatch = client.post(
        "/v1/predict",
        headers={"X-ML-Service-Token": service_token},
        json={"features": {"attendance": 80}},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["code"] == "FEATURE_SCHEMA_MISMATCH"

    ok = client.post(
        "/v1/predict",
        headers={"X-ML-Service-Token": service_token},
        json={
            "features": {
                "attendance": 80,
                "prior_gpa": 3.1,
                "assessment_avg": 70,
            }
        },
    )
    assert ok.status_code == 200
    body = ok.json()["data"]
    assert body["modelVersion"] == metadata["modelVersion"]
    assert body["isSynthetic"] is True
    assert "risk assessment" in body["disclaimer"].lower() or "Synthetic" in body["disclaimer"]


def test_model_metadata_endpoint(service_token, artifact_dir, synthetic_csv, monkeypatch):
    _train_synthetic(artifact_dir, synthetic_csv)
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(artifact_dir))
    from importlib import reload
    import app.main as main

    reload(main)
    client = TestClient(main.app)
    response = client.get("/v1/model", headers={"X-ML-Service-Token": service_token})
    assert response.status_code == 200
    assert response.json()["data"]["operatingMode"] == "synthetic_demo"
