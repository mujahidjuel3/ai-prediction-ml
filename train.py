"""Train a reproducible binary classifier from an institution-approved or synthetic CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import OperatingMode
from app.dataset import DatasetSpec, drop_incomplete_targets, parse_features, validate_frame


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--target", required=True, help="Explicit binary label column; define the outcome and lead time in --target-definition.")
    p.add_argument("--target-definition", required=True, help="Plain-language outcome, cutoff, and prediction lead time.")
    p.add_argument("--features", required=True, help="Comma-separated feature columns observed at prediction cutoff.")
    p.add_argument("--positive-label", required=True, help="Exact positive class value, e.g. 1 or fail.")
    p.add_argument("--student-column", help="Optional student/group ID column; never used as a feature.")
    p.add_argument("--time-column", help="Optional term/time column. With this set, later rows are held out for testing.")
    p.add_argument("--unique-key", help="Optional comma-separated columns used to report duplicate records.")
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--data-mode",
        choices=[mode.value for mode in OperatingMode],
        default=OperatingMode.SYNTHETIC_DEMO.value,
        help="synthetic_demo (default for fixtures), validated_model, or production_prediction.",
    )
    p.add_argument(
        "--i-confirm-university-approvals",
        action="store_true",
        help="Required only when --data-mode=production_prediction.",
    )
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_report(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        target=args.target,
        target_definition=args.target_definition,
        features=parse_features(args.features),
        positive_label=str(args.positive_label),
        student_column=args.student_column,
        time_column=args.time_column,
        unique_key=[part.strip() for part in args.unique_key.split(",") if part.strip()] if args.unique_key else [],
    )
    return validate_frame(frame, spec)


def split(frame: pd.DataFrame, args: argparse.Namespace, y: np.ndarray):
    indices = np.arange(len(frame))
    if args.time_column:
        order = frame[args.time_column].sort_values(kind="stable").index.to_numpy()
        cut = int(len(order) * (1 - args.test_size))
        if cut <= 0 or cut >= len(order):
            raise ValueError("Invalid temporal test split")
        train_idx, test_idx = order[:cut], order[cut:]
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            raise ValueError("Temporal split must contain both target classes in train and test")
        return train_idx, test_idx, "temporal_holdout"
    if args.student_column:
        groups = frame[args.student_column].astype(str).to_numpy()
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed).split(indices, y, groups))
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            raise ValueError("Group split must contain both target classes in train and test")
        return train_idx, test_idx, "student_group_holdout"
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=y)
    return train_idx, test_idx, "stratified_random_holdout"


def build_pipeline(frame: pd.DataFrame, features: list[str], estimator: Any) -> Pipeline:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")), ("model", estimator)])


def evaluate(y_true: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray, positive: str, classes: Any) -> dict[str, Any]:
    pos = np.asarray(y_true).astype(str) == positive
    pred_pos = np.asarray(predicted).astype(str) == positive
    positive_index = list(map(str, classes)).index(positive)
    probs = probabilities[:, positive_index]
    report: dict[str, Any] = {
        "precision": float(precision_score(pos, pred_pos, zero_division=0)),
        "recall": float(recall_score(pos, pred_pos, zero_division=0)),
        "f1": float(f1_score(pos, pred_pos, zero_division=0)),
        "confusionMatrix": confusion_matrix(pos, pred_pos, labels=[False, True]).tolist(),
        "classificationReport": classification_report(y_true, predicted, output_dict=True, zero_division=0),
        "prAuc": float(average_precision_score(pos, probs)),
        "brierScore": float(brier_score_loss(pos, probs)),
    }
    report["rocAuc"] = float(roc_auc_score(pos, probs)) if len(np.unique(pos)) == 2 else None
    return report


def write_versioned_artifacts(output: Path, model: Any, metadata: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    version_dir = output / "versions" / metadata["modelVersion"]
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"
    joblib.dump(model, model_path)
    metadata["modelSha256"] = sha256(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    # Latest pointers for the FastAPI service.
    shutil.copy2(model_path, output / "model.joblib")
    shutil.copy2(metadata_path, output / "metadata.json")
    return version_dir


def train(args: argparse.Namespace) -> dict[str, Any]:
    mode = OperatingMode(args.data_mode)
    if mode is OperatingMode.PRODUCTION_PREDICTION and not args.i_confirm_university_approvals:
        raise ValueError(
            "production_prediction requires --i-confirm-university-approvals after university dataset, "
            "target, horizon, thresholds, features, review, and consent approvals"
        )
    if mode is OperatingMode.SYNTHETIC_DEMO and "synthetic" not in args.target_definition.lower() and "fixture" not in args.target_definition.lower():
        # Soft guidance only: still allow if the operator labels the mode synthetic.
        pass

    frame = pd.read_csv(args.csv)
    report = validate_and_report(args, frame)
    frame = drop_incomplete_targets(frame, args.target)
    y = frame[args.target].astype(str).to_numpy()
    features: list[str] = report["featureColumns"]
    train_idx, test_idx, split_name = split(frame, args, y)
    x_train, x_test = frame.loc[train_idx, features], frame.loc[test_idx, features]
    y_train, y_test = y[train_idx], y[test_idx]
    if min(pd.Series(y_train).value_counts()) < 2:
        raise ValueError("Training split has fewer than two rows in a class; obtain more data")

    candidates = {
        "dummy_prior": build_pipeline(frame, features, DummyClassifier(strategy="prior")),
        "logistic_regression": build_pipeline(frame, features, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        "random_forest": build_pipeline(frame, features, RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=args.seed, n_jobs=1)),
    }
    results = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        results[name] = evaluate(y_test, candidate.predict(x_test), candidate.predict_proba(x_test), report["positiveLabel"], candidate.classes_)

    winner = max(("logistic_regression", "random_forest"), key=lambda name: results[name]["prAuc"])
    base = candidates[winner]
    probability_calibrated = False
    try:
        counts = pd.Series(y_train).value_counts()
        folds = min(3, int(counts.min()))
        if folds >= 2:
            calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=folds, n_jobs=1)
            calibrated.fit(x_train, y_train)
            base = calibrated
            probability_calibrated = True
    except (ValueError, TypeError):
        probability_calibrated = False

    final_metrics = evaluate(y_test, base.predict(x_test), base.predict_proba(x_test), report["positiveLabel"], base.classes_)
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + winner
    metadata: dict[str, Any] = {
        "modelVersion": model_version,
        "algorithm": winner,
        "features": features,
        "targetColumn": args.target,
        "targetDefinition": args.target_definition,
        "positiveLabel": report["positiveLabel"],
        "classes": report["classes"],
        "probabilityCalibrated": probability_calibrated,
        "split": split_name,
        "trainRows": int(len(train_idx)),
        "testRows": int(len(test_idx)),
        "dataSha256": sha256(args.csv),
        "python": platform.python_version(),
        "scikitLearn": sklearn.__version__,
        "seed": args.seed,
        "metrics": final_metrics,
        "candidateMetrics": results,
        "dataQuality": report,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "operatingMode": mode.value,
        "isSynthetic": mode is OperatingMode.SYNTHETIC_DEMO,
        "disclaimer": (
            "Synthetic demo artifact for pipeline testing only. Not an actual student risk assessment."
            if mode is OperatingMode.SYNTHETIC_DEMO
            else "Artifact produced under the declared operating mode. NestJS generation remains gated by university approvals."
        ),
    }
    version_dir = write_versioned_artifacts(args.output, base, metadata)
    report_path = Path("reports") / f"training-{metadata['modelVersion']}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    metadata["artifactVersionDir"] = str(version_dir)
    return metadata


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        result = train(arguments)
        print(
            json.dumps(
                {
                    "modelVersion": result["modelVersion"],
                    "operatingMode": result["operatingMode"],
                    "isSynthetic": result["isSynthetic"],
                    "metrics": result["metrics"],
                    "probabilityCalibrated": result["probabilityCalibrated"],
                    "disclaimer": result["disclaimer"],
                },
                indent=2,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise SystemExit(f"Training stopped: {exc}") from exc
