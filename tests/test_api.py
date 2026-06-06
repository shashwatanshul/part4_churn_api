from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.train_model_for_api import main as train_model

if not (ROOT / "app" / "model.pkl").exists():
    train_model()

from app.main import app  # noqa: E402

client = TestClient(app)


def sample_payload():
    path = ROOT / "sample_payload.json"
    if not path.exists():
        train_model()
    return json.loads(path.read_text(encoding="utf-8"))


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_count"] > 0


def test_predict_one_customer():
    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["predicted_class"] in [0, 1]
    assert body["risk_band"] in ["low", "medium", "high"]
    assert isinstance(body["risk_explanation"], str)


def test_batch_predict():
    payload = sample_payload()
    response = client.post("/batch_predict", json=[payload, payload])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all("churn_probability" in item for item in body)


def test_validation_rejects_bad_payload():
    payload = sample_payload()
    payload["recency_days"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
