"""Tests for the FastAPI service."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_files, monkeypatch):
    model_path, meta_path = model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METADATA_PATH", str(meta_path))
    from api.main import app

    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1
    assert body["model_version"] == "test"
    assert body["api_version"] == "v1"
    assert r.headers["X-Request-ID"] == body["request_id"]
    assert r.headers["X-API-Version"] == "v1"


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    payload = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [("age", 5), ("cp", 9), ("chol", -1), ("sex", 2)])
def test_predict_rejects_invalid_values(client, sample_patient, field, value):
    r = client.post("/predict", json={**sample_patient, field: value})
    assert r.status_code == 422


def test_validation_errors_name_the_offending_field(client, sample_patient):
    r = client.post("/predict", json={**sample_patient, "cp": 9})
    locs = [e["loc"] for e in r.json()["detail"]]
    assert any("cp" in loc for loc in locs)


def test_client_supplied_request_id_is_echoed(client, sample_patient):
    r = client.post("/predict", json=sample_patient, headers={"X-Request-ID": "trace-123"})
    assert r.json()["request_id"] == "trace-123"


def test_predict_rejects_unknown_fields(client, sample_patient):
    assert client.post("/predict", json={**sample_patient, "bmi": 30}).status_code == 422


def test_batch_predict(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_metrics_endpoint_exposes_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text


def test_health_reports_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={}).status_code in (422, 503)
