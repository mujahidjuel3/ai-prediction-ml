from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.dataset import synthetic_fixture


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    frame = synthetic_fixture(40)
    path = tmp_path / "synthetic-fixture.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def service_token(monkeypatch: pytest.MonkeyPatch) -> str:
    token = "synthetic-ml-service-token-for-tests-32b"
    monkeypatch.setenv("ML_SERVICE_TOKEN", token)
    monkeypatch.setenv("ML_OPERATING_MODE", "synthetic_demo")
    return token


@pytest.fixture()
def artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "artifacts"
    path.mkdir()
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(path))
    return path
