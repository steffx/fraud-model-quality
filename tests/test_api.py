"""FastAPI service: contract, validation and consistency with the model's thresholds."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api

LOW_RISK = {
    "amount": 42.0,
    "hour": 13,
    "merchant_category": "groceries",
    "country": "DE",
    "is_new_payee": False,
    "device_trusted": True,
    "txn_last_24h": 2,
    "account_age_days": 1200,
    "amount_to_avg_ratio": 1.0,
}
HIGH_RISK = {
    **LOW_RISK,
    "amount": 1800.0,
    "hour": 2,
    "merchant_category": "electronics",
    "country": "NG",
    "is_new_payee": True,
    "device_trusted": False,
    "txn_last_24h": 6,
    "amount_to_avg_ratio": 9.0,
}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] in {"logistic_regression", "random_forest", "gradient_boosting"}


def test_model_info_contract(client):
    info = client.get("/model").json()
    assert set(info) == {"champion", "model_version", "trained_at", "thresholds", "quality_bar", "models"}
    assert len(info["model_version"]) == 10
    assert info["models"][info["champion"]]["passes_quality_bar"] is True
    assert set(info["models"]) == {"logistic_regression", "random_forest", "gradient_boosting"}


def test_predict_contract(client):
    res = client.post("/predict", json=LOW_RISK)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"fraud_probability", "decision", "model", "model_version"}
    assert 0 <= body["fraud_probability"] <= 1


def test_low_risk_payment_is_approved(client):
    assert client.post("/predict", json=LOW_RISK).json()["decision"] == "approve"


def test_high_risk_payment_is_not_approved(client):
    assert client.post("/predict", json=HIGH_RISK).json()["decision"] in {"review", "block"}


def test_decision_matches_published_thresholds(client):
    thresholds = client.get("/model").json()["thresholds"]
    for payload in (LOW_RISK, HIGH_RISK, {**LOW_RISK, "device_trusted": False, "is_new_payee": True, "amount": 600.0}):
        body = client.post("/predict", json=payload).json()
        p = body["fraud_probability"]
        expected = "block" if p >= thresholds["block"] else "review" if p >= thresholds["review"] else "approve"
        assert body["decision"] == expected


@pytest.mark.parametrize(
    "change, field",
    [
        ({"amount": -5}, "amount"),
        ({"amount": 0}, "amount"),
        ({"amount": 50_000.01}, "amount"),
        ({"amount": "lots"}, "amount"),
        ({"hour": 24}, "hour"),
        ({"merchant_category": "casino"}, "merchant_category"),
        ({"country": "de"}, "country"),
        ({"txn_last_24h": -1}, "txn_last_24h"),
        ({"unexpected_field": 1}, "unexpected_field"),
    ],
    ids=["negative amount", "zero amount", "over limit", "text amount", "hour 24", "unknown category", "lowercase country", "negative count", "extra field"],
)
def test_invalid_input_is_rejected(client, change, field):
    res = client.post("/predict", json={**LOW_RISK, **change})
    assert res.status_code == 422
    assert field in [err["loc"][-1] for err in res.json()["detail"]]


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in LOW_RISK.items() if k != "device_trusted"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"][-1] == "device_trusted"


def test_batch_matches_single_predictions_in_order(client):
    batch = [LOW_RISK, HIGH_RISK, LOW_RISK]
    results = client.post("/predict/batch", json=batch).json()
    singles = [client.post("/predict", json=p).json() for p in batch]
    assert results == singles


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_oversized_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[LOW_RISK] * 1001).status_code == 413


def test_openapi_documents_the_prediction_endpoint(client):
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    assert "Transaction" in spec["components"]["schemas"]
