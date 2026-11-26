"""Verify bundled or freshly trained synthetic demo artifacts load for inference."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.dataset import synthetic_fixture

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists(), reason="demo artifacts not prepared")
def test_bundled_demo_artifacts_are_synthetic_and_predict():
    meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    assert meta["isSynthetic"] is True
    assert meta["operatingMode"] == "synthetic_demo"
    assert "synthetic" in meta["targetDefinition"].lower() or "fixture" in meta["targetDefinition"].lower()
    model = joblib.load(ARTIFACTS / "model.joblib")
    features = meta["features"]
    row = synthetic_fixture(40, seed=2026).iloc[[0]][features]
    predicted = model.predict(row)
    assert len(predicted) == 1
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)
        assert probs.shape[0] == 1


def test_demo_csv_manifest_documents_synthetic_origin():
    csv_path = ROOT / "data" / "synthetic_demo.csv"
    manifest_path = ROOT / "data" / "synthetic_demo.manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        pytest.skip("demo dataset not generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["origin"] == "SYNTHETIC_DEMO"
    frame = pd.read_csv(csv_path)
    assert len(frame) >= 20
    assert set(manifest["features"]).issubset(frame.columns)
