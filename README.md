# Academic Sentinel ML service

This is a private local FastAPI service. No university data, trained model, training metric, or model artifact is included. The service returns `MODEL_REQUIRED` until a model has been trained from authorized data. Bind Uvicorn to `127.0.0.1`; only the NestJS backend should call prediction endpoints. Set the same random `ML_SERVICE_TOKEN` in the backend and ML process.

## Local Windows setup

Install Python 3.11 or 3.12 from the official Python distribution if `py --version` is unavailable. From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:ML_SERVICE_TOKEN = 'use-a-random-local-secret-of-your-own'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoints: `GET /health/live`, `GET /health/ready`. A ready response with `model_required` is expected until training has completed. `GET /v1/model` and `POST /v1/predict` require `X-ML-Service-Token`.

## Dataset and target setup

Place a locally authorized CSV under `data/` (the folder is git-ignored). Never commit student records. Obtain institutional approval and define a single binary target, the observation cutoff, and prediction lead time before training. Do not combine course failure, low GPA, and dropout into one target. Choose only features available by that cutoff. Exclude identity, final outcome fields, post-cutoff grades/attendance, intervention fields, and any derived leakage columns. Use `--student-column` to keep a student's rows in one split; use `--time-column` for a later-time holdout when the source has a reliable chronological field.

Synthetic format example (for documentation only; not training data):

```csv
term_code,attendance_pct,prior_gpa,assessment_avg,failed_course
TERM-2025-FALL,82,3.10,76,0
TERM-2025-FALL,54,2.05,43,1
```

Example command, after replacing column names/target definition with the approved schema:

```powershell
python train.py --csv data/university-approved.csv --target failed_course `
  --target-definition "At least one enrolled course failed by term end; features cut off at week 6, 6-week lead time" `
  --features attendance_pct,prior_gpa,assessment_avg --positive-label 1 `
  --student-column student_number --time-column term_index
```

The trainer checks the target/feature schema, reports missingness, duplicates and label distribution, requires two classes and a minimum data size, performs a reproducible temporal holdout (or student-group/stratified split), fits preprocessing only on training data, compares a prior baseline, Logistic Regression and Random Forest using PR-AUC, and emits test metrics and versioned artifact metadata. Calibration is attempted on training folds and is marked explicitly. Metrics from a synthetic test fixture are not evidence of institutional model quality. Review all data-quality, leakage, calibration, cohort fairness and workload results before enabling the model.

Artifacts in `artifacts/` and reports in `reports/` are git-ignored. Back up and version approved artifacts only through a separately reviewed secure process. Joblib artifacts are executable serialization: load only artifacts produced by this trusted training workflow.
