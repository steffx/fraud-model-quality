"""FastAPI service that serves the champion model, plus a small dashboard.

    uvicorn fraudmodel.service:create_app --factory     # http://localhost:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import ARTIFACTS_DIR, COUNTRIES, FEATURES, MERCHANT_CATEGORIES

MerchantCategory = Literal[tuple(MERCHANT_CATEGORIES)]  # type: ignore[valid-type]
Country = Literal[tuple(COUNTRIES)]  # type: ignore[valid-type]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0, le=50_000, description="Payment amount in EUR")
    hour: int = Field(ge=0, le=23, description="Local hour of the payment")
    merchant_category: MerchantCategory
    country: Country
    is_new_payee: bool
    device_trusted: bool
    txn_last_24h: int = Field(ge=0, le=500)
    account_age_days: int = Field(ge=0, le=36_500)
    amount_to_avg_ratio: float = Field(ge=0, le=1_000, description="Amount divided by the customer's average payment")


class Prediction(BaseModel):
    fraud_probability: float
    decision: Literal["approve", "review", "block"]
    model: str
    model_version: str


def decide(probability: float, thresholds: dict) -> str:
    if probability >= thresholds["block"]:
        return "block"
    if probability >= thresholds["review"]:
        return "review"
    return "approve"


def create_app(artifacts_dir: Path | str | None = None) -> FastAPI:
    artifacts = Path(artifacts_dir or os.environ.get("ARTIFACTS_DIR", ARTIFACTS_DIR))
    if not (artifacts / "model.joblib").exists():
        from .train import save_artifacts, train_all

        save_artifacts(train_all(), artifacts)

    bundle = joblib.load(artifacts / "model.joblib")
    summary = json.loads((artifacts / "metrics.json").read_text())
    pipeline, thresholds, name = bundle["pipeline"], bundle["thresholds"], bundle["name"]
    version = summary["model_version"]

    app = FastAPI(title="NovaPay fraud model", version=version)

    def score(rows: list[Transaction]) -> list[Prediction]:
        frame = pd.DataFrame([r.model_dump() for r in rows])[FEATURES]
        probs = pipeline.predict_proba(frame)[:, 1]
        return [
            Prediction(fraud_probability=round(float(p), 4), decision=decide(float(p), thresholds), model=name, model_version=version)
            for p in probs
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": name, "model_version": version}

    @app.get("/model")
    def model_info() -> dict:
        return {
            "champion": name,
            "model_version": version,
            "trained_at": summary["trained_at"],
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "quality_bar": summary["quality_bar"],
            "models": summary["models"],
        }

    @app.post("/predict", response_model=Prediction)
    def predict(transaction: Transaction) -> Prediction:
        return score([transaction])[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(transactions: list[Transaction]) -> list[Prediction]:
        if not transactions:
            raise HTTPException(status_code=422, detail="Send at least one transaction")
        if len(transactions) > 1000:
            raise HTTPException(status_code=413, detail="At most 1000 transactions per request")
        return score(transactions)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (Path(__file__).parent / "dashboard.html").read_text()

    return app

