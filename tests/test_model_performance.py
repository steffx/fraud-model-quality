"""Model performance: thresholds agreed with the fraud team, plus behavioural checks."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

from fraudmodel.config import FEATURES, QUALITY_BAR, TARGET
from fraudmodel.train import train_all

pytestmark = pytest.mark.model

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]


# --- quality bar ---


@pytest.mark.parametrize(
    "metric, limit, direction",
    [
        ("pr_auc", QUALITY_BAR["min_pr_auc"], ">="),
        ("precision", QUALITY_BAR["min_precision"], ">="),
        ("recall", QUALITY_BAR["min_recall"], ">="),
        ("false_positive_rate", QUALITY_BAR["max_fpr"], "<="),
    ],
)
def test_champion_meets_quality_bar(champion, metric, limit, direction):
    value = champion["metrics"][metric]
    assert (value >= limit) if direction == ">=" else (value <= limit), f"{metric}={value} (limit {direction} {limit})"


def test_champion_is_the_best_model_that_passes(run):
    passing = {n: r["metrics"]["pr_auc"] for n, r in run["results"].items() if r["passes_quality_bar"]}
    assert run["champion"] == max(passing, key=passing.get)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_beats_a_dummy_baseline(run, name):
    train, _valid, test = run["splits"]
    dummy = DummyClassifier(strategy="prior").fit(train[FEATURES], train[TARGET])
    baseline = average_precision_score(test[TARGET], dummy.predict_proba(test[FEATURES])[:, 1])
    assert run["results"][name]["metrics"]["pr_auc"] > baseline * 10


def test_simple_baseline_is_not_good_enough(run):
    """Characterisation: a linear model cannot capture tiny card-testing payments AND large takeovers,
    so it misses the recall target. This documents why a tree model is used."""
    lr = run["results"]["logistic_regression"]
    assert not lr["passes_quality_bar"]
    assert lr["metrics"]["recall"] < QUALITY_BAR["min_recall"]


# --- decision thresholds ---


def test_thresholds_are_ordered_and_valid(champion):
    t = champion["thresholds"]
    assert 0 < t["review"] <= t["block"] < 1


def test_blocking_is_almost_always_right(champion):
    """Blocking a genuine payment is the worst customer experience, so blocks need >= 90% precision."""
    assert champion["metrics"]["block_precision"] >= 0.90


# --- slices: performance on specific fraud patterns, not just on average ---


def _slice_recall(run, champion, mask_fn):
    _train, _valid, test = run["splits"]
    fraud = test[test[TARGET] == 1]
    mask = mask_fn(fraud).to_numpy()
    scores = champion["pipeline"].predict_proba(fraud[FEATURES])[:, 1]
    caught = scores >= champion["thresholds"]["review"]
    return caught[mask].mean(), int(mask.sum())


@pytest.mark.parametrize(
    "slice_name, mask_fn",
    [
        ("large payments over 1,000 EUR", lambda d: d["amount"] > 1000),
        ("card testing under 5 EUR", lambda d: d["amount"] < 5),
    ],
)
def test_recall_on_important_fraud_patterns(run, champion, slice_name, mask_fn):
    recall, n = _slice_recall(run, champion, mask_fn)
    assert n >= 10, f"too few examples in slice '{slice_name}' to judge"
    assert recall >= 0.6, f"{slice_name}: recall {recall:.2f} on {n} fraud cases"


# --- behavioural tests (expected directions, invariance) ---

TYPICAL = {
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


def _score(champion, **overrides):
    row = pd.DataFrame([{**TYPICAL, **overrides}])[FEATURES]
    return float(champion["pipeline"].predict_proba(row)[:, 1][0])


def test_typical_payment_is_low_risk(champion):
    assert _score(champion) < champion["thresholds"]["review"]


@pytest.mark.parametrize(
    "change",
    [
        {"device_trusted": False},
        {"is_new_payee": True},
        {"amount_to_avg_ratio": 8.0, "amount": 900.0},
        {"merchant_category": "crypto"},
        {"hour": 3},
    ],
    ids=["untrusted device", "new payee", "unusually large", "crypto merchant", "3 am"],
)
def test_risk_signal_increases_the_score(champion, change):
    assert _score(champion, **change) > _score(champion)


def test_account_takeover_profile_is_flagged(champion):
    score = _score(
        champion,
        amount=1800.0,
        hour=2,
        merchant_category="electronics",
        country="NG",
        is_new_payee=True,
        device_trusted=False,
        txn_last_24h=6,
        amount_to_avg_ratio=9.0,
    )
    assert score >= champion["thresholds"]["review"]


def test_predictions_are_deterministic(run, champion):
    _train, _valid, test = run["splits"]
    X = test[FEATURES].head(500)
    np.testing.assert_array_equal(champion["pipeline"].predict_proba(X), champion["pipeline"].predict_proba(X))


def test_scores_1000_payments_in_under_200ms(run, champion):
    X = run["splits"][2][FEATURES].head(1000)
    champion["pipeline"].predict_proba(X)  # warm-up
    started = time.perf_counter()
    champion["pipeline"].predict_proba(X)
    assert time.perf_counter() - started < 0.2


# --- stability ---


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 99])
def test_quality_bar_holds_across_random_seeds(seed):
    """A result that only holds for one lucky seed is not a result."""
    result = train_all(seed=seed)
    assert result["results"][result["champion"]]["passes_quality_bar"], {
        n: r["metrics"] for n, r in result["results"].items()
    }
