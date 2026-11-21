from argparse import Namespace

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.dataset import synthetic_fixture
from train import build_pipeline, split, train, validate_and_report


def fixture_frame():
    return synthetic_fixture(40).rename(columns={"assessment_avg": "assessment_avg"})


def test_schema_requires_explicit_binary_target_and_features():
    frame = fixture_frame()
    args = Namespace(
        features="attendance,prior_gpa",
        target="outcome",
        target_definition="synthetic fixture fail by term end",
        positive_label="1",
        student_column="student",
        time_column="term",
        unique_key="student,term",
    )
    report = validate_and_report(args, frame)
    assert report["rows"] == 40
    assert report["duplicateRows"] == 0
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_and_report(args, frame.drop(columns=["prior_gpa"]))


def test_time_split_is_chronological_and_student_id_is_not_feature():
    frame = fixture_frame()
    args = Namespace(
        features="attendance,prior_gpa",
        target="outcome",
        target_definition="synthetic fixture fail by term end",
        positive_label="1",
        student_column="student",
        time_column="term",
        unique_key=None,
        test_size=0.5,
        seed=2,
    )
    y = frame.outcome.astype(str).to_numpy()
    train_idx, test_idx, method = split(frame, args, y)
    assert method == "temporal_holdout"
    # With equal halves on term 1 then term 2, a 50% temporal cut keeps later terms only in the holdout.
    assert frame.iloc[train_idx].term.max() < frame.iloc[test_idx].term.min()
    assert "student" not in args.features.split(",")


def test_preprocessing_pipeline_fits_numeric_and_categorical_features():
    frame = fixture_frame().assign(category=["a", "b"] * 20)
    model = build_pipeline(frame, ["attendance", "category"], LogisticRegression(max_iter=1000))
    model.fit(frame[["attendance", "category"]].iloc[:30], frame.outcome.iloc[:30].astype(str))
    assert len(model.predict(frame[["attendance", "category"]].iloc[30:])) == 10


def test_train_writes_versioned_synthetic_artifact(tmp_path, synthetic_csv):
    args = Namespace(
        csv=synthetic_csv,
        target="outcome",
        target_definition="synthetic fixture fail by term end; features cut off at week 6",
        features="attendance,prior_gpa,assessment_avg",
        positive_label="1",
        student_column="student",
        time_column="term",
        unique_key="student,term",
        output=tmp_path / "artifacts",
        seed=2026,
        test_size=0.25,
        data_mode="synthetic_demo",
        i_confirm_university_approvals=False,
    )
    metadata = train(args)
    assert metadata["isSynthetic"] is True
    assert metadata["operatingMode"] == "synthetic_demo"
    assert "Not an actual student risk" in metadata["disclaimer"]
    assert (tmp_path / "artifacts" / "model.joblib").is_file()
    assert (tmp_path / "artifacts" / "metadata.json").is_file()
    version_dir = tmp_path / "artifacts" / "versions" / metadata["modelVersion"]
    assert version_dir.is_dir()
    assert (version_dir / "model.joblib").is_file()


def test_production_mode_requires_explicit_confirmation(tmp_path, synthetic_csv):
    args = Namespace(
        csv=synthetic_csv,
        target="outcome",
        target_definition="synthetic fixture",
        features="attendance,prior_gpa,assessment_avg",
        positive_label="1",
        student_column="student",
        time_column="term",
        unique_key=None,
        output=tmp_path / "artifacts",
        seed=1,
        test_size=0.25,
        data_mode="production_prediction",
        i_confirm_university_approvals=False,
    )
    with pytest.raises(ValueError, match="university"):
        train(args)
