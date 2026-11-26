"""Dataset validation and preprocessing helpers shared by training and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    target: str
    target_definition: str
    features: list[str]
    positive_label: str
    student_column: str | None = None
    time_column: str | None = None
    unique_key: list[str] | None = None


def parse_features(raw: str) -> list[str]:
    features = [part.strip() for part in raw.split(",") if part.strip()]
    if not features or len(features) != len(set(features)):
        raise ValueError("--features must list one or more unique column names")
    return features


def validate_frame(frame: pd.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    reserved = {spec.target, spec.student_column, spec.time_column, *(spec.unique_key or [])} - {None}
    if reserved.intersection(spec.features):
        raise ValueError("Target, student ID, time, and unique-key columns cannot be features")
    required = set(spec.features + [spec.target])
    if spec.student_column:
        required.add(spec.student_column)
    if spec.time_column:
        required.add(spec.time_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) < 20:
        raise ValueError("At least 20 labeled rows are required for a development evaluation")
    labels = frame[spec.target].dropna().astype(str)
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"This initial training pipeline requires exactly two target classes; found {classes}")
    positive = str(spec.positive_label)
    if positive not in classes:
        raise ValueError(f"--positive-label {positive!r} is not present in target classes {classes}")
    key_cols = spec.unique_key or []
    missing_values = frame[spec.features + [spec.target]].isna().sum().to_dict()
    return {
        "rows": int(len(frame)),
        "featureColumns": list(spec.features),
        "targetColumn": spec.target,
        "targetDefinition": spec.target_definition,
        "classes": classes,
        "positiveLabel": positive,
        "targetMissing": int(frame[spec.target].isna().sum()),
        "missingByColumn": {k: int(v) for k, v in missing_values.items()},
        "duplicateRows": int(frame.duplicated(subset=key_cols or None).sum()),
        "uniqueKey": key_cols,
        "labelDistribution": {str(k): int(v) for k, v in frame[spec.target].value_counts(dropna=False).items()},
    }


def drop_incomplete_targets(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame.loc[frame[target].notna()].reset_index(drop=True)


def synthetic_fixture(rows: int = 40, seed: int = 2026) -> pd.DataFrame:
    """Generate clearly synthetic rows for pipeline tests and demos. Not university data.

    Target ``outcome`` is binary: 1 = at-risk of failing the next midterm window
    (synthetic prediction horizon: ~4 weeks before final assessments). Features are
    observed at cutoff and never include the future outcome itself.
    """
    import numpy as np

    if rows < 20:
        raise ValueError("Synthetic fixtures require at least 20 rows")
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(35, 100, size=rows)
    prior_gpa = rng.uniform(1.2, 4.0, size=rows)
    quiz_avg = rng.uniform(20, 100, size=rows)
    midterm = rng.uniform(25, 100, size=rows)
    assignment_avg = rng.uniform(30, 100, size=rows)
    assessment_avg = 0.3 * quiz_avg + 0.4 * midterm + 0.3 * assignment_avg
    # Deterministic risk score → binary label (not institutional policy).
    risk_score = (
        (100 - attendance) * 0.35
        + (4.0 - prior_gpa) * 12.0
        + (100 - assessment_avg) * 0.25
        + rng.normal(0, 4, size=rows)
    )
    outcome = (risk_score > float(np.median(risk_score))).astype(int)
    half = rows // 2
    return pd.DataFrame(
        {
            "student": [f"synthetic-{i:04d}" for i in range(rows)],
            "term": ([1] * half) + ([2] * (rows - half)),
            "attendance": np.round(attendance, 2),
            "prior_gpa": np.round(prior_gpa, 2),
            "quiz_avg": np.round(quiz_avg, 2),
            "midterm": np.round(midterm, 2),
            "assignment_avg": np.round(assignment_avg, 2),
            "assessment_avg": np.round(assessment_avg, 2),
            "outcome": outcome,
        }
    )
