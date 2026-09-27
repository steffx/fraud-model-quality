"""Synthetic card-payment data with realistic fraud patterns.

Fraud and genuine payments overlap on purpose: fraudsters sometimes use trusted devices,
genuine customers sometimes buy expensive electronics abroad at night. A model should
be good, not perfect, and the tests must cope with that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EU_COUNTRIES, HIGH_RISK_COUNTRIES, HOME_COUNTRY, MERCHANT_CATEGORIES


def _choice(rng: np.random.Generator, options: list[str], weights: list[float], size: int) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    return rng.choice(options, size=size, p=p / p.sum())


def _countries(rng, size, home, eu, other, high_risk):
    other_countries = ["GB", "US", "TR"]
    return _choice(
        rng,
        [HOME_COUNTRY, *EU_COUNTRIES, *other_countries, *HIGH_RISK_COUNTRIES],
        [home] + [eu / len(EU_COUNTRIES)] * len(EU_COUNTRIES) + [other / 3] * 3 + [high_risk / len(HIGH_RISK_COUNTRIES)] * len(HIGH_RISK_COUNTRIES),
        size,
    )


def generate_transactions(n: int = 50_000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Returns n card payments with an ``is_fraud`` label. Same seed -> same data."""
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # --- genuine payments ---
    legit = pd.DataFrame(
        {
            "amount": np.round(rng.lognormal(3.6, 1.1, n_legit), 2),
            "hour": np.clip(np.round(rng.normal(14, 4.5, n_legit)), 0, 23).astype(int),
            "merchant_category": _choice(rng, MERCHANT_CATEGORIES, [28, 18, 12, 10, 10, 3, 2, 17], n_legit),
            "country": _countries(rng, n_legit, home=0.82, eu=0.12, other=0.05, high_risk=0.01),
            "is_new_payee": rng.random(n_legit) < 0.18,
            "device_trusted": rng.random(n_legit) < 0.92,
            "txn_last_24h": rng.poisson(2.5, n_legit),
            "account_age_days": rng.integers(30, 4000, n_legit),
            "amount_to_avg_ratio": np.round(rng.lognormal(0.0, 0.6, n_legit), 3),
            "is_fraud": 0,
        }
    )

    # --- fraud: a mix of account takeover, scams and card testing ---
    kind = _choice(rng, ["takeover", "scam", "card_testing"], [0.5, 0.3, 0.2], n_fraud)
    amount = np.where(
        kind == "card_testing",
        np.round(rng.uniform(0.5, 5, n_fraud), 2),
        np.round(rng.lognormal(5.4, 1.0, n_fraud), 2),
    )
    night = rng.random(n_fraud) < 0.45
    fraud = pd.DataFrame(
        {
            "amount": amount,
            "hour": np.where(night, rng.integers(0, 6, n_fraud), np.clip(np.round(rng.normal(15, 5, n_fraud)), 0, 23)).astype(int),
            "merchant_category": _choice(rng, MERCHANT_CATEGORIES, [9, 7, 3, 12, 18, 16, 13, 22], n_fraud),
            "country": _countries(rng, n_fraud, home=0.62, eu=0.17, other=0.12, high_risk=0.09),
            "is_new_payee": rng.random(n_fraud) < 0.55,
            "device_trusted": rng.random(n_fraud) < np.where(kind == "scam", 0.85, 0.35),
            "txn_last_24h": rng.poisson(np.where(kind == "card_testing", 6, 3.5), n_fraud),
            "account_age_days": rng.integers(30, 4000, n_fraud),
            "amount_to_avg_ratio": np.round(
                np.where(kind == "card_testing", rng.lognormal(-1.0, 0.7, n_fraud), rng.lognormal(1.2, 0.8, n_fraud)), 3
            ),
            "is_fraud": 1,
        }
    )

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["amount"] = df["amount"].clip(0.5, 50_000)
    df.insert(0, "customer_id", [f"C{c:05d}" for c in rng.integers(0, 5_000, len(df))])
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    return df
