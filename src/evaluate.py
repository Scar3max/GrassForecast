from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import RESULTS_DIR
from src.forecast_tabpfn import FEATURE_COLUMNS, _fit_model, _model_name


def _scores(actual: list[int], predicted: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def evaluate_models(feature_df: pd.DataFrame) -> dict:
    df = feature_df.sort_values("date").reset_index(drop=True)
    if len(df) < 40:
        raise ValueError(
            f"Only {len(df)} complete labeled days are available; at least 40 are required for walk-forward evaluation."
        )
    if df["high_activity_day"].nunique() < 2:
        raise ValueError("Walk-forward evaluation needs both high- and low-activity labeled days.")

    X = df[FEATURE_COLUMNS]
    y = df["high_activity_day"].astype(int)
    splitter = TimeSeriesSplit(n_splits=3)
    actual: list[int] = []
    predicted: list[int] = []
    baseline_actual: list[int] = []
    baseline_predictions = {
        "same_as_yesterday": [],
        "same_weekday_last_week": [],
        "majority_class": [],
    }
    model_used = "RandomForest fallback"

    for train_idx, test_idx in splitter.split(X):
        model = _fit_model(X.iloc[train_idx], y.iloc[train_idx])
        model_used = _model_name(model)
        fold_predictions = model.predict(X.iloc[test_idx]).astype(int).tolist()
        actual.extend(y.iloc[test_idx].tolist())
        predicted.extend(fold_predictions)

        for index in test_idx:
            baseline_actual.append(int(y.iloc[index]))
            baseline_predictions["same_as_yesterday"].append(int(y.iloc[index - 1]))
            last_week_index = index - 7
            baseline_predictions["same_weekday_last_week"].append(
                int(y.iloc[last_week_index]) if last_week_index >= 0 else int(y.iloc[train_idx].mode().iloc[0])
            )
            baseline_predictions["majority_class"].append(int(y.iloc[train_idx].mode().iloc[0]))

    metrics = {
        "evaluation": {
            "method": "3-fold expanding walk-forward",
            "out_of_sample_days": len(actual),
            "labeled_days": len(df),
            "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        },
        "model_used": model_used,
        "tabpfn": _scores(actual, predicted),
        "baselines": {
            name: _scores(baseline_actual, predictions)
            for name, predictions in baseline_predictions.items()
        },
    }

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = ["TabPFN" if model_used == "TabPFN" else model_used, *metrics["baselines"]]
    values = [
        metrics["tabpfn"]["accuracy"],
        *[metrics["baselines"][name]["accuracy"] for name in metrics["baselines"]],
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Walk-forward accuracy vs. baselines")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "metrics.png")
    plt.close()
    return metrics
