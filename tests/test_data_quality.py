"""Data quality: the data passes every check, and every check catches the problem it exists for."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudmodel.config import TARGET
from fraudmodel.data import generate_transactions
from fraudmodel.quality import DataQualityError, run_checks
from fraudmodel.train import split, train_all

pytestmark = pytest.mark.data

CHECK_NAMES = [c.name for c in run_checks(generate_transactions(n=5_000, seed=1))]


@pytest.mark.parametrize("check_name", CHECK_NAMES)
def test_training_data_passes_check(dataset, check_name):
    result = next(r for r in run_checks(dataset) if r.name == check_name)
    assert result.passed, result.detail


def test_generation_is_reproducible():
    a = generate_transactions(n=2_000, seed=7)
    b = generate_transactions(n=2_000, seed=7)
    c = generate_transactions(n=2_000, seed=8)
    assert a.equals(b)
    assert not a.equals(c)


def test_fraud_rate_matches_configuration(dataset):
    assert dataset[TARGET].mean() == pytest.approx(0.02, abs=0.001)


def test_split_has_no_customer_in_two_sets(dataset):
    train, valid, test = split(dataset)
    sets = [set(s["customer_id"]) for s in (train, valid, test)]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert len(train) + len(valid) + len(test) == len(dataset)


def test_split_keeps_a_similar_fraud_rate(dataset):
    rates = [s[TARGET].mean() for s in split(dataset)]
    assert max(rates) - min(rates) < 0.01, rates


# --- negative tests: each corruption must be caught by the right check ---


def _corrupt(df, how):
    df = df.copy()
    if how == "null amounts":
        df.loc[:9, "amount"] = np.nan
    elif how == "negative amount":
        df.loc[0, "amount"] = -12.5
    elif how == "hour 24":
        df.loc[0, "hour"] = 24
    elif how == "unknown category":
        df.loc[0, "merchant_category"] = "casino"
    elif how == "lowercase country":
        df.loc[0, "country"] = "de"
    elif how == "duplicate id":
        df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]
    elif how == "label value 2":
        df.loc[0, TARGET] = 2
    elif how == "booleans as text":
        df["is_new_payee"] = df["is_new_payee"].map({True: "yes", False: "no"})
    elif how == "missing column":
        df = df.drop(columns=["device_trusted"])
    elif how == "fraud rate 50%":
        df[TARGET] = np.arange(len(df)) % 2
    elif how == "leaked label":
        # e.g. a "chargeback count" column that is only known after the fraud happened
        df["txn_last_24h"] = df[TARGET] * 20 + np.random.default_rng(0).integers(0, 3, len(df))
    elif how == "duplicated rows":
        copies = df.iloc[:200].assign(transaction_id=lambda d: d["transaction_id"] + "X")
        df = pd.concat([df, copies], ignore_index=True)
    return df


CORRUPTIONS = {
    "null amounts": "no missing values",
    "negative amount": "amount between 0 and 50,000",
    "hour 24": "hour between 0 and 23",
    "unknown category": "known merchant categories",
    "lowercase country": "known country codes",
    "duplicate id": "transaction ids are unique",
    "label value 2": "label is binary",
    "booleans as text": "column types",
    "missing column": "required columns present",
    "fraud rate 50%": "fraud rate in expected range",
    "leaked label": "no target leakage",
    "duplicated rows": "no duplicated payments",
}


@pytest.mark.parametrize("corruption, expected_check", CORRUPTIONS.items(), ids=list(CORRUPTIONS))
def test_check_catches_corruption(corruption, expected_check):
    df = _corrupt(generate_transactions(n=10_000, seed=3), corruption)
    failed = {r.name for r in run_checks(df) if not r.passed}
    assert expected_check in failed


def test_training_refuses_bad_data():
    df = _corrupt(generate_transactions(n=10_000, seed=3), "leaked label")
    with pytest.raises(DataQualityError, match="no target leakage"):
        train_all(df)
