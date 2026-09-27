"""Shared constants: feature schema and the quality bar agreed with the fraud team."""

from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

MERCHANT_CATEGORIES = [
    "groceries",
    "restaurants",
    "utilities",
    "travel",
    "electronics",
    "gift_cards",
    "crypto",
    "p2p_transfer",
]
HOME_COUNTRY = "DE"
EU_COUNTRIES = ["FR", "NL", "AT", "IT", "ES", "PL", "BE"]
HIGH_RISK_COUNTRIES = ["NG", "RU", "KP", "IR"]
COUNTRIES = [HOME_COUNTRY, *EU_COUNTRIES, "GB", "US", "TR", *HIGH_RISK_COUNTRIES]

NUMERIC_FEATURES = ["amount", "hour", "txn_last_24h", "account_age_days", "amount_to_avg_ratio"]
BOOLEAN_FEATURES = ["is_new_payee", "device_trusted"]
CATEGORICAL_FEATURES = ["merchant_category", "country"]
FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
ID_COLUMNS = ["transaction_id", "customer_id"]

# Minimum quality for a model to be allowed in production.
QUALITY_BAR = {
    "min_pr_auc": 0.60,     # ranking quality on imbalanced data
    "min_precision": 0.80,  # of flagged payments, at least 80% are real fraud (at the review threshold)
    "min_recall": 0.50,     # at least half of all fraud is caught at that threshold
    "max_fpr": 0.005,       # at most 0.5% of genuine payments are flagged
}

# Precision targets used to pick the decision thresholds on validation data.
# The review target is set above the quality bar (0.80) as a safety margin: a threshold tuned
# on validation data usually loses a little precision on unseen data.
REVIEW_PRECISION_TARGET = 0.85  # score >= review threshold -> one-time code
BLOCK_PRECISION_TARGET = 0.95   # score >= block threshold  -> payment blocked
