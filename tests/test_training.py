from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from train import build_pipeline, split, validate_and_report
from sklearn.linear_model import LogisticRegression


def fixture_frame():
    return pd.DataFrame({"student": [f"s{i}" for i in range(40)], "term": np.repeat([1, 2], 20),
                         "attendance": np.linspace(40, 99, 40), "prior_gpa": np.linspace(1.5, 4, 40),
                         "outcome": [0, 1] * 20})


def test_schema_requires_explicit_binary_target_and_features():
    frame = fixture_frame()
    args = Namespace(features="attendance,prior_gpa", target="outcome", target_definition="fail by term end",
                     positive_label="1", student_column="student", time_column="term", unique_key="student,term")
    report = validate_and_report(args, frame)
    assert report["rows"] == 40
    assert report["duplicateRows"] == 0
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_and_report(args, frame.drop(columns=["prior_gpa"]))


def test_time_split_is_chronological_and_student_id_is_not_feature():
    frame = fixture_frame()
    args = Namespace(features="attendance,prior_gpa", target="outcome", target_definition="fail by term end",
                     positive_label="1", student_column="student", time_column="term", unique_key=None,
                     test_size=0.2, seed=2)
    _, report = validate_and_report(args, frame), None
    y = frame.outcome.astype(str).to_numpy()
    train, test, method = split(frame, args, y)
    assert method == "temporal_holdout"
    assert frame.iloc[train].term.max() < frame.iloc[test].term.min()
    assert "student" not in args.features.split(",")


def test_preprocessing_pipeline_fits_numeric_and_categorical_features():
    frame = fixture_frame().assign(category=["a", "b"] * 20)
    x = frame[["attendance", "category"]]
    model = build_pipeline(frame, ["attendance", "category"], LogisticRegression(max_iter=1000))
    model.fit(x.iloc[:30], frame.outcome.iloc[:30].astype(str))
    assert len(model.predict(x.iloc[30:])) == 10
