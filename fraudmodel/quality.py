"""Data quality checks. Training refuses to start when a blocking check fails.

A model is only as good as its data: a silent schema change, a pipeline bug that fills
nulls with zeros, or a column that leaks the label will all produce a model that looks
excellent in testing and fails in production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import COUNTRIES, FEATURES, ID_COLUMNS, MERCHANT_CATEGORIES, TARGET


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


class DataQualityError(Exception):
    def __init__(self, failures: list[CheckResult]):
        self.failures = failures
        super().__init__("Data quality checks failed: " + "; ".join(f"{f.name} ({f.detail})" for f in failures))


EXPECTED_DTYPES = {
    "transaction_id": "object",
    "customer_id": "object",
    "amount": "float",
    "hour": "int",
    "merchant_category": "object",
    "country": "object",
    "is_new_payee": "bool",
    "device_trusted": "bool",
    "txn_last_24h": "int",
    "account_age_days": "int",
    "amount_to_avg_ratio": "float",
    TARGET: "int",
}


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "object"


def run_checks(df: pd.DataFrame, fraud_rate_range: tuple[float, float] = (0.005, 0.10)) -> list[CheckResult]:
    results: list[CheckResult] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append(CheckResult(name, bool(passed), detail or ("ok" if passed else "failed")))

    required = ID_COLUMNS + FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    check("required columns present", not missing, f"missing: {missing}" if missing else "")
    if missing:
        return results  # the other checks need the columns

    wrong_types = {c: _kind(df[c]) for c, k in EXPECTED_DTYPES.items() if _kind(df[c]) != k}
    check("column types", not wrong_types, f"unexpected: {wrong_types}" if wrong_types else "")

    nulls = {c: int(n) for c, n in df[required].isna().sum().items() if n}
    check("no missing values", not nulls, f"nulls: {nulls}" if nulls else "")

    dup_ids = int(df["transaction_id"].duplicated().sum())
    check("transaction ids are unique", dup_ids == 0, f"{dup_ids} duplicates" if dup_ids else "")

    dup_rows = int(df.drop(columns=["transaction_id"]).duplicated().sum())
    check("no duplicated payments", dup_rows <= len(df) * 0.001, f"{dup_rows} identical rows")

    bad_amount = int(((df["amount"] <= 0) | (df["amount"] > 50_000)).sum())
    check("amount between 0 and 50,000", bad_amount == 0, f"{bad_amount} out of range" if bad_amount else "")

    bad_hour = int((~df["hour"].between(0, 23)).sum())
    check("hour between 0 and 23", bad_hour == 0, f"{bad_hour} out of range" if bad_hour else "")

    check("counts are non-negative", bool((df[["txn_last_24h", "account_age_days"]] >= 0).all().all()))

    unknown_cat = sorted(set(df["merchant_category"]) - set(MERCHANT_CATEGORIES))
    check("known merchant categories", not unknown_cat, f"unknown: {unknown_cat}" if unknown_cat else "")

    unknown_country = sorted(set(df["country"]) - set(COUNTRIES))
    check("known country codes", not unknown_country, f"unknown: {unknown_country}" if unknown_country else "")

    labels = set(df[TARGET].unique())
    plain = sorted(v.item() if hasattr(v, "item") else v for v in labels)
    check("label is binary", labels <= {0, 1}, f"values: {plain}")

    rate = float(df[TARGET].mean())
    low, high = fraud_rate_range
    check("fraud rate in expected range", low <= rate <= high, f"{rate:.2%} (expected {low:.1%}-{high:.1%})")

    # Target leakage: a single feature that almost perfectly predicts fraud usually means
    # it was computed after the fact (e.g. "chargeback_filed") and won't exist at payment time.
    leaks = []
    if len(labels) == 2:
        for col in [c for c in FEATURES if c not in ("merchant_category", "country")]:
            # Rows with missing values are reported by their own check; skipping them here
            # keeps this check from crashing on exactly the data it should report on.
            valid = df[[col, TARGET]].dropna()
            if valid[TARGET].nunique() < 2:
                continue
            try:
                values = valid[col].astype(float)
            except (TypeError, ValueError):
                continue  # wrong type, reported by the column types check
            auc = roc_auc_score(valid[TARGET], values)
            if max(auc, 1 - auc) > 0.97:
                leaks.append(f"{col} (AUC {max(auc, 1 - auc):.3f})")
    check("no target leakage", not leaks, f"suspicious: {leaks}" if leaks else "")

    return results


def assert_quality(df: pd.DataFrame, **kwargs) -> list[CheckResult]:
    results = run_checks(df, **kwargs)
    failures = [r for r in results if not r.passed]
    if failures:
        raise DataQualityError(failures)
    return results
