"""Train several models, pick decision thresholds and select a champion.

    python -m fraudmodel.train            # writes artifacts/model.joblib and artifacts/metrics.json
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ARTIFACTS_DIR,
    BLOCK_PRECISION_TARGET,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    QUALITY_BAR,
    REVIEW_PRECISION_TARGET,
    TARGET,
)
from .data import generate_transactions
from .evaluate import evaluate, meets_quality_bar, threshold_for_precision
from .quality import assert_quality


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("prep", _preprocessor(scale=True)), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, random_state=seed)),
            ]
        ),
    }


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 train/validation/test split by customer, so no customer appears in two sets."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, groups=df["customer_id"]))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, valid_idx = next(inner.split(rest, groups=rest["customer_id"]))
    return rest.iloc[train_idx], rest.iloc[valid_idx], test


def train_all(df: pd.DataFrame | None = None, seed: int = 42) -> dict:
    """Validates data, trains every candidate, and returns models, thresholds and test metrics."""
    df = generate_transactions(seed=seed) if df is None else df
    quality = assert_quality(df)
    train, valid, test = split(df, seed)

    results = {}
    for name, pipeline in candidate_models(seed).items():
        started = time.perf_counter()
        pipeline.fit(train[FEATURES], train[TARGET])
        train_seconds = time.perf_counter() - started

        valid_scores = pipeline.predict_proba(valid[FEATURES])[:, 1]
        review = threshold_for_precision(valid[TARGET].to_numpy(), valid_scores, REVIEW_PRECISION_TARGET)
        block = max(review, threshold_for_precision(valid[TARGET].to_numpy(), valid_scores, BLOCK_PRECISION_TARGET))

        test_scores = pipeline.predict_proba(test[FEATURES])[:, 1]
        metrics = evaluate(test[TARGET].to_numpy(), test_scores, review, block)
        metrics["train_seconds"] = round(train_seconds, 2)
        results[name] = {
            "pipeline": pipeline,
            "thresholds": {"review": review, "block": block},
            "metrics": metrics,
            "passes_quality_bar": meets_quality_bar(metrics),
            "test_scores": test_scores,
        }

    eligible = [n for n, r in results.items() if r["passes_quality_bar"]] or list(results)
    champion = max(eligible, key=lambda n: results[n]["metrics"]["pr_auc"])
    return {"results": results, "champion": champion, "quality": quality, "splits": (train, valid, test), "seed": seed}


def save_artifacts(run: dict, out_dir: Path = ARTIFACTS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    champion = run["champion"]
    best = run["results"][champion]
    model_path = out_dir / "model.joblib"
    joblib.dump({"pipeline": best["pipeline"], "thresholds": best["thresholds"], "name": champion}, model_path)
    version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:10]

    train, valid, test = run["splits"]
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": version,
        "champion": champion,
        "quality_bar": QUALITY_BAR,
        "data": {
            "seed": run["seed"],
            "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
            "fraud_rate": {"train": round(float(train[TARGET].mean()), 4), "test": round(float(test[TARGET].mean()), 4)},
            "quality_checks": [c.to_dict() for c in run["quality"]],
        },
        "models": {
            name: {
                "thresholds": {k: round(v, 4) for k, v in r["thresholds"].items()},
                "metrics": r["metrics"],
                "passes_quality_bar": r["passes_quality_bar"],
            }
            for name, r in run["results"].items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        y_true=test[TARGET].to_numpy(),
        scores=best["test_scores"],
        **{f"scores_{name}": r["test_scores"] for name, r in run["results"].items()},
    )
    return summary


def main() -> None:
    run = train_all()
    summary = save_artifacts(run)
    print(f"{'model':<22}{'PR-AUC':>8}{'precision':>11}{'recall':>8}{'FPR':>8}  quality bar")
    for name, m in summary["models"].items():
        mm = m["metrics"]
        flag = "PASS" if m["passes_quality_bar"] else "FAIL"
        star = " (champion)" if name == summary["champion"] else ""
        print(f"{name:<22}{mm['pr_auc']:>8.3f}{mm['precision']:>11.3f}{mm['recall']:>8.3f}{mm['false_positive_rate']:>8.4f}  {flag}{star}")
    print(f"Saved champion '{summary['champion']}' version {summary['model_version']} to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
