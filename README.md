# Fraud model quality

[![Model quality](https://github.com/steffx/fraud-model-quality/actions/workflows/ci.yml/badge.svg)](https://github.com/steffx/fraud-model-quality/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-F7931E?logo=scikitlearn)
![Playwright](https://img.shields.io/badge/Playwright-Python-2EAD33?logo=playwright)

**Live report:** https://steffx.github.io/fraud-model-quality/

How do you test a machine-learning model that decides whether a card payment is fraud? This project
trains three candidate models on synthetic payment data and tests them the way a bank would before
letting one make real decisions:

- **Data quality**: training refuses to start on bad data, and every check is proven to catch the problem it exists for
- **Performance thresholds**: precision, recall, false positive rate and PR-AUC against a quality bar agreed with the fraud team
- **The model as a service**: a FastAPI endpoint with a strict input contract, tested with pytest
- **The dashboard**: a web page for analysts, tested end-to-end with Playwright for Python
- **A published report**: every CI run on `main` updates the GitHub Pages report

> All data is synthetic. No real customer data is used.

## Results

| Model | PR-AUC | Precision | Recall | False positive rate | Quality bar |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic regression | 0.708 | 85.6% | 49.0% | 0.2% | ✘ fails recall |
| Random forest | 0.811 | 81.8% | 67.5% | 0.3% | ✔ pass |
| **Gradient boosting** (champion) | **0.824** | **80.1%** | **70.4%** | **0.4%** | ✔ pass |

The simple linear model misses too much fraud: it can't learn that both *tiny* payments (card testing)
and *large* ones (account takeover) are risky. The tree models can. This result holds across random
seeds, which is itself a test.

## The quality bar

| Metric | Required | Why |
| --- | --- | --- |
| PR-AUC | ≥ 0.60 | Ranking quality that works on imbalanced data (accuracy is useless: "never fraud" is 98% accurate) |
| Precision | ≥ 80% | At least 4 of 5 alerts are real fraud, so analysts trust them |
| Recall | ≥ 50% | At least half of all fraud is caught |
| False positive rate | ≤ 0.5% | At most 1 in 200 genuine payments is interrupted |
| Block precision | ≥ 90% | A hard block on a genuine payment is the worst customer experience |

The model makes three decisions: **approve**, **review** (ask the customer for a one-time code), or **block**.
Both thresholds are chosen on a separate validation set. The review threshold targets 85% precision, a
margin above the 80% bar, because a threshold tuned on one dataset loses some precision on new data.

## Test suite (82 tests)

| File | Tests | Covers |
| --- | ---: | --- |
| `test_data_quality.py` | 30 | 13 checks pass on real data; 12 corruptions (nulls, bad ranges, unknown categories, duplicates, label leakage…) are each caught by the right check; training refuses bad data; no customer appears in two data splits; reproducible generation |
| `test_model_performance.py` | 25 | Quality bar, champion selection, beats a dummy baseline, threshold order, block precision, recall on specific fraud patterns, behavioural tests (each risk signal raises the score), determinism, latency, stability across seeds |
| `test_api.py` | 20 | Contract, decisions match the published thresholds, 10 invalid-input cases, batch consistency, request limits, OpenAPI spec |
| `test_dashboard.py` | 7 | Playwright: champion and version shown, table matches the API, scoring a normal and a risky payment, validation errors |

### Testing the tests

Each data-quality check has a matching negative test that corrupts the data on purpose and expects
that exact check to fail. This caught a real bug: with missing values present, the target-leakage
check crashed instead of reporting, so one bad column would have hidden every other problem.

### Behavioural tests

Metrics say how often the model is right on average, but not whether it behaves sensibly. Behavioural
tests take a typical grocery payment and change one thing at a time (untrusted device, new payee, 3 a.m.,
crypto merchant, unusually large). Each change must raise the fraud score.

## Getting started

Requirements: Python 3.11 or newer.

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium

python -m fraudmodel.train        # trains and compares the three models
python -m pytest                  # all tests (about 30 seconds)
python -m pytest -m "not slow"    # skip the multi-seed stability tests
python report/build_site.py       # builds site/index.html

uvicorn fraudmodel.service:create_app --factory   # API + dashboard on http://localhost:8000
```

Example request:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "amount": 1800, "hour": 2, "merchant_category": "electronics", "country": "NG",
  "is_new_payee": true, "device_trusted": false, "txn_last_24h": 6,
  "account_age_days": 900, "amount_to_avg_ratio": 9
}'
# {"fraud_probability": 0.9967, "decision": "block", "model": "gradient_boosting", ...}
```

## Project structure

```
fraudmodel/
  data.py          synthetic payments: account takeover, scams and card testing mixed with genuine spending
  quality.py       data quality checks (blocking)
  train.py         candidate models, customer-level split, threshold selection, champion
  evaluate.py      precision, recall, FPR, PR-AUC, quality bar
  service.py       FastAPI service
  dashboard.html   analyst dashboard
report/build_site.py   GitHub Pages report
tests/                 pytest + Playwright
```

## Design decisions

- **Split by customer, not by payment.** If the same customer is in training and test data, the model can memorise their habits and the test results look better than reality.
- **Synthetic data with overlap.** Fraudsters sometimes use trusted devices and genuine customers sometimes shop abroad at night, so no single feature gives the answer away. The leakage check enforces this.
- **Models compete; the bar decides.** The champion is the best model that passes the quality bar, not simply the one with the highest score.
- **Publishing GitHub Pages** needs one setting: *Settings → Pages → Source: GitHub Actions*.
