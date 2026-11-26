import pytest

from app.dataset import DatasetSpec, synthetic_fixture, validate_frame


def test_synthetic_fixture_is_clearly_labeled_and_valid():
    frame = synthetic_fixture(40)
    assert frame["student"].str.startswith("synthetic-").all()
    for column in ("quiz_avg", "midterm", "assignment_avg"):
        assert column in frame.columns
    report = validate_frame(
        frame,
        DatasetSpec(
            target="outcome",
            target_definition="synthetic fixture",
            features=["attendance", "prior_gpa", "assessment_avg"],
            positive_label="1",
            student_column="student",
            time_column="term",
            unique_key=["student", "term"],
        ),
    )
    assert report["rows"] == 40
    assert set(report["classes"]) == {"0", "1"}


def test_synthetic_fixture_is_deterministic_for_demo_seed():
    a = synthetic_fixture(40, seed=2026)
    b = synthetic_fixture(40, seed=2026)
    assert a.equals(b)


def test_validate_frame_rejects_feature_leakage_columns():
    frame = synthetic_fixture(40)
    with pytest.raises(ValueError, match="cannot be features"):
        validate_frame(
            frame,
            DatasetSpec(
                target="outcome",
                target_definition="synthetic fixture",
                features=["attendance", "outcome"],
                positive_label="1",
            ),
        )
