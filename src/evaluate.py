from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.forecast_tabpfn import FEATURE_COLUMNS, _fit_model
from src.config import RESULTS_DIR


def _baseline_predictions(y: pd.Series) -> dict[str, pd.Series]:
    return {
        "same_as_yesterday": y.shift(1).fillna(y.iloc[0]),
        "same_weekday_last_week": y.shift(7).fillna(y.iloc[0]),
        "majority_class": pd.Series([y.mode().iloc[0]] * len(y), index=y.index),
    }


def evaluate_models(feature_df: pd.DataFrame) -> dict:
    df = feature_df.sort_values("date").reset_index(drop=True)
    X = df[FEATURE_COLUMNS]
    y = df["high_activity_day"]
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []

    for train_idx, test_idx in tscv.split(X):
        model = _fit_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        fold_scores.append({
            "accuracy": accuracy_score(y.iloc[test_idx], preds),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, zero_division=0),
        })

    metrics = {
        "tabpfn": {
            "accuracy": sum(score["accuracy"] for score in fold_scores) / max(len(fold_scores), 1),
            "balanced_accuracy": sum(score["balanced_accuracy"] for score in fold_scores) / max(len(fold_scores), 1),
            "f1": sum(score["f1"] for score in fold_scores) / max(len(fold_scores), 1),
        },
        "baselines": {}
    }

    baseline_scores = {}
    for name, preds in _baseline_predictions(y).items():
        baseline_scores[name] = {
            "accuracy": accuracy_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "f1": f1_score(y, preds, zero_division=0),
        }
    metrics["baselines"] = baseline_scores

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    labels = ["TabPFN", *list(baseline_scores.keys())]
    values = [metrics["tabpfn"]["accuracy"], *[baseline_scores[item]["accuracy"] for item in baseline_scores]]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Walk-forward accuracy vs. baselines")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics.png")
    plt.close()

    return metrics
