from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from tabpfn import TabPFNClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    TabPFNClassifier = None

from src.config import RESULTS_DIR


FEATURE_COLUMNS = [
    "day_of_week", "weekend", "month", "day_of_year", "public_holiday",
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_hours", "wind_speed_10m_max", "uv_index_max",
    "daylight_duration", "cloud_cover_mid", "lag_1", "lag_7", "rolling_7day_mean"
]


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    if TabPFNClassifier is not None and os.getenv("TABPFN_TOKEN"):
        try:
            model = TabPFNClassifier(device='cpu')
            model.fit(X_train, y_train)
            return model
        except Exception:
            pass
    return RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)


def forecast_tomorrow(feature_df: pd.DataFrame) -> dict:
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    X = feature_df[FEATURE_COLUMNS]
    y_high = feature_df["high_activity_day"]
    y_peak = feature_df["peak_window"]

    last_row = X.iloc[[-1]].copy()
    if len(X) < 2:
        return {"high_activity_day": 1, "high_activity_probability": 0.5, "peak_window": "afternoon"}

    model_high = _fit_model(X.iloc[:-1], y_high.iloc[:-1])
    if hasattr(model_high, "predict_proba"):
        probabilities = model_high.predict_proba(last_row)
        if probabilities.shape[1] == 2:
            high_prob = float(probabilities[0, 1])
        else:
            high_prob = float(np.max(probabilities))
    else:
        high_prob = 0.5
    model_peak = _fit_model(X.iloc[:-1], y_peak.iloc[:-1])
    peak_window = model_peak.predict(last_row)[0]

    payload = {
        "high_activity_day": int(high_prob >= 0.5),
        "high_activity_probability": float(high_prob),
        "peak_window": str(peak_window),
    }
    (RESULTS_DIR / "tomorrow_prediction.json").write_text(json.dumps(payload, indent=2))
    return payload
