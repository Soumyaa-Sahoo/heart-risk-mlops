"""FastAPI service for heart-disease risk prediction.

Endpoints
    GET  /              service info
    GET  /health        liveness + readiness (503 until the model is loaded)
    GET  /model-info    metadata of the loaded model
    POST /predict       score one patient
    POST /predict/batch score up to 100 patients
    GET  /metrics       Prometheus metrics

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.routing import Match
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import BatchRequest, BatchResponse, Patient, PredictionResponse
from heart import config
from heart.predict import HeartModel


# ---------------------------------------------------------------------------
# Structured JSON logging (one line per event, easy to grep / ship to Loki)
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("heart.api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


log = _configure_logging()


def _log(level: int, msg: str, **fields) -> None:
    log.log(level, msg, extra={"extra_fields": fields})


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency",
                    ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("heart_api_predicted_probability", "Predicted P(disease)",
                        buckets=[i / 10 for i in range(1, 11)])
MODEL_INFO = Gauge("heart_api_model_info", "Loaded model (value always 1)",
                   ["model_name", "model_version"])


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))
    metadata_path = Path(os.getenv("METADATA_PATH", config.METADATA_PATH))
    try:
        app.state.model = HeartModel.load(model_path, metadata_path)
        meta = app.state.model.metadata
        MODEL_INFO.labels(meta.get("model_name", "unknown"), app.state.model.version).set(1)
        _log(logging.INFO, "model loaded", path=str(model_path),
             model_name=meta.get("model_name"), model_version=app.state.model.version)
    except Exception as exc:  # keep serving /health so the failure is visible
        app.state.model = None
        _log(logging.ERROR, "model failed to load", path=str(model_path), error=str(exc))
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description=(
        "Predicts presence of heart disease from 13 clinical features (UCI dataset). "
        "Academic MLOps exercise: outputs are model estimates, not medical advice "
        "or a diagnostic rule."
    ),
    lifespan=lifespan,
)


def _route_template(request: Request) -> str:
    """Path template (e.g. '/predict') rather than the raw URL, so metric
    label cardinality stays bounded even under random-path scans."""
    route = request.scope.get("route")
    if route is not None:
        return route.path
    for r in request.app.routes:
        if r.matches(request.scope)[0] == Match.FULL:
            return r.path
    return "unmatched"


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        endpoint = _route_template(request)
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            _log(logging.INFO, "request", request_id=request_id, method=request.method,
                 path=request.url.path, status=status, latency_ms=round(elapsed * 1000, 2),
                 client=request.client.host if request.client else None)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-API-Version"] = config.API_VERSION
    return response


def _model(request: Request) -> HeartModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _record(preds: list[dict], request_id: str) -> None:
    for p in preds:
        PREDICTIONS.labels(p["label"]).inc()
        PROBABILITY.observe(p["probability_disease"])
        # Log the outcome, not the patient's clinical values (health data).
        _log(logging.INFO, "prediction", request_id=request_id, **p)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "api_version": config.API_VERSION,
            "docs": "/docs", "health": "/health",
            "disclaimer": "Academic exercise; not medical advice."}


@app.get("/health")
def health(request: Request):
    loaded = getattr(request.app.state, "model", None) is not None
    body = {"status": "ok" if loaded else "degraded", "model_loaded": loaded}
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.get("/model-info")
def model_info(request: Request):
    m = _model(request).metadata
    keys = ["model_name", "feature_set", "engineered_features", "model_version",
            "selection_rule", "inference_latency_ms", "decision_threshold", "trained_at",
            "cv_metrics", "oof_metrics", "oof_subgroup_metrics", "test_metrics", "best_params",
            "input_features", "sklearn_version"]
    return {k: m.get(k) for k in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: Patient, request: Request):
    model = _model(request)
    pred = model.predict([patient.model_dump()])[0]
    _record([pred], request.state.request_id)
    return PredictionResponse(**pred, api_version=config.API_VERSION,
                              model_version=model.version,
                              decision_threshold=model.threshold,
                              request_id=request.state.request_id)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    model = _model(request)
    preds = model.predict([p.model_dump() for p in batch.patients])
    _record(preds, request.state.request_id)
    return BatchResponse(predictions=preds, api_version=config.API_VERSION,
                         model_version=model.version,
                         decision_threshold=model.threshold,
                         request_id=request.state.request_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
