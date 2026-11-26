# Private FastAPI ML service — bind to private network only; never expose publicly.
# When demo artifacts exist under ./artifacts (model.joblib + metadata.json), they are
# bundled for synthetic_demo serving. Production prediction mode is never enabled here.
FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ML_OPERATING_MODE=synthetic_demo
RUN useradd --system --uid 10001 ml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY train.py .
COPY scripts ./scripts
COPY data ./data
# Bundle trained synthetic artifacts when present (prepare_demo_artifacts.py).
COPY artifacts ./artifacts
RUN mkdir -p artifacts data && chown -R ml:ml /app
USER ml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
