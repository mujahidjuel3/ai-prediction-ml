#!/usr/bin/env python3
"""Write a deterministic synthetic academic CSV for Academic Sentinel demos.

This dataset is NOT institutional student data. Every row is synthetic.
Binary target ``outcome``: 1 = demo at-risk for failing within a ~4-week
prediction horizon before finals; 0 = not at-risk (synthetic only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataset import synthetic_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_demo.csv"))
    args = parser.parse_args()

    frame = synthetic_fixture(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    manifest = {
        "path": str(args.output).replace("\\", "/"),
        "rows": int(len(frame)),
        "seed": args.seed,
        "origin": "SYNTHETIC_DEMO",
        "disclaimer": (
            "Synthetic academic fixture for Academic Sentinel semester demos only. "
            "Not university records. Not validated student risk predictions."
        ),
        "target": "outcome",
        "targetDefinition": (
            "Synthetic binary label: outcome=1 means demo at-risk of course failure "
            "within a ~4-week horizon before final assessments. Features are "
            "attendance, prior_gpa, quiz_avg, midterm, assignment_avg, assessment_avg "
            "observed at prediction cutoff. Not an institutional policy threshold."
        ),
        "features": [
            "attendance",
            "prior_gpa",
            "quiz_avg",
            "midterm",
            "assignment_avg",
            "assessment_avg",
        ],
        "positiveLabel": "1",
        "labelDistribution": {str(k): int(v) for k, v in frame["outcome"].value_counts().items()},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
