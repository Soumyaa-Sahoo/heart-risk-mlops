# Model-serving image for the heart-disease risk API.
# Build after training (models/model.joblib must exist):
#   docker build -t heart-risk-api:latest .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    LOG_LEVEL=INFO

WORKDIR /app

# Unprivileged runtime user
RUN useradd --create-home --uid 10001 appuser

# Dependencies first so this layer is cached across code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/metadata.json models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
