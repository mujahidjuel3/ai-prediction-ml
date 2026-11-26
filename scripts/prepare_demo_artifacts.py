#!/usr/bin/env python3
"""Generate synthetic CSV and train a synthetic_demo scikit-learn artifact.

Keeps ML_OPERATING_MODE / --data-mode at synthetic_demo. Never enables
production_prediction. Artifacts are labeled isSynthetic=true.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "synthetic_demo.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    gen = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_demo_dataset.py"),
            "--rows",
            str(args.rows),
            "--seed",
            str(args.seed),
            "--output",
            str(args.csv),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if gen.returncode != 0:
        raise SystemExit(gen.stderr or gen.stdout or "dataset generation failed")

    features = "attendance,prior_gpa,quiz_avg,midterm,assignment_avg,assessment_avg"
    target_definition = (
        "Synthetic binary label: outcome=1 means demo at-risk of course failure "
        "within a ~4-week horizon before final assessments. Fixture/synthetic only."
    )
    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--csv",
            str(args.csv),
            "--target",
            "outcome",
            "--target-definition",
            target_definition,
            "--features",
            features,
            "--positive-label",
            "1",
            "--student-column",
            "student",
            "--time-column",
            "term",
            "--unique-key",
            "student,term",
            "--output",
            str(args.output),
            "--seed",
            str(args.seed),
            "--data-mode",
            "synthetic_demo",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if train.returncode != 0:
        raise SystemExit(train.stderr or train.stdout or "training failed")

    meta_path = args.output / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("isSynthetic") or meta.get("operatingMode") != "synthetic_demo":
        raise SystemExit("Refusing non-synthetic demo artifact")
    print(train.stdout)


if __name__ == "__main__":
    main()
