# Academic Sentinel ML service

Private FastAPI service for training and inference. **No university dataset or production student risk model is included.**

## Operating modes

| Mode | Purpose | NestJS generation |
|---|---|---|
| `synthetic_demo` (default) | Pipeline/integration tests with synthetic fixtures only | Always disabled. Never store/show as student risk. |
| `validated_model` | University-approved dataset trained and reviewed | Still disabled until Nest gates pass |
| `production_prediction` | Reserved for approved institutional use | Requires Nest `ML_UNIVERSITY_POLICY_APPROVED` + `ML_PREDICTION_PERSISTENCE_ENABLED` |

Set `ML_OPERATING_MODE` on the ML process. Training writes `operatingMode` / `isSynthetic` / `disclaimer` into artifact metadata.

## Local Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:ML_SERVICE_TOKEN = 'use-a-random-local-secret-of-at-least-32-chars'
$env:ML_OPERATING_MODE = 'synthetic_demo'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health: `GET /health/live`, `GET /health/ready`. Protected routes require `X-ML-Service-Token`:

- `GET /v1/model`
- `GET /v1/versions`
- `POST /v1/predict`

Bind to loopback / private network only. Only NestJS should call this service.

## Synthetic training (tests / demo pipeline only)

```powershell
python -c "from app.dataset import synthetic_fixture; synthetic_fixture(40).to_csv('data/synthetic-fixture.csv', index=False)"
python train.py --csv data/synthetic-fixture.csv --target outcome `
  --target-definition "synthetic fixture fail by term end; features cut off at week 6" `
  --features attendance,prior_gpa,assessment_avg --positive-label 1 `
  --student-column student --time-column term --data-mode synthetic_demo
```

Artifacts land in `artifacts/` (latest pointers) and `artifacts/versions/<modelVersion>/`. Metrics from synthetic fixtures are **not** institutional model quality evidence.

## University-approved training

Place an approved local CSV under ignored `data/`. Never commit student records. Obtain approvals for target, cutoff/horizon, features, thresholds, review, and consent before any production claim. Use `--data-mode validated_model` or, only after explicit confirmation, `--data-mode production_prediction --i-confirm-university-approvals`.

## NestJS integration

Backend uses `ML_SERVICE_URL`, `ML_SERVICE_TOKEN`, timeout/retries, and `GET /api/v1/ml/status` (Admin).  
`POST /api/v1/predictions/generate` remains **HTTP 503** until university policy gates and non-synthetic persistence are explicitly enabled. Prediction rows are stored only through the existing Prisma `ModelVersion` / `Prediction` design when those gates pass in a future step — this service does not write to PostgreSQL itself.
