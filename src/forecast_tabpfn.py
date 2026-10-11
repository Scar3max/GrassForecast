from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import holidays
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from tabpfn import TabPFNClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    TabPFNClassifier = None

from src.config import RESULTS_DIR

FEATURE_COLUMNS = [
    "day_of_week", "weekend", "month", "day_of_year", "public_holiday",
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_hours", "wind_speed_10m_max",
    "daylight_duration", "cloud_cover_mean", "lag_1", "lag_7", "rolling_7day_mean",
]

MAX_STALE_FORECAST_HORIZON_DAYS = 30
COUNT_WEATHER_COLUMNS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_hours", "wind_speed_10m_max", "daylight_duration",
    "cloud_cover_mean",
]
COUNT_FEATURE_COLUMNS = [
    "forecast_horizon_days", "target_day_of_week", "target_weekend",
    "target_month", "target_day_of_year", "target_public_holiday",
    *[f"target_{column}" for column in COUNT_WEATHER_COLUMNS],
    "origin_activity_count", "origin_lag_7", "origin_rolling_7day_mean",
]


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    if TabPFNClassifier is not None and os.getenv("TABPFN_TOKEN"):
        try:
            model = TabPFNClassifier(device="cpu")
            model.fit(X_train, y_train)
            return model
        except Exception as exc:
            print(f"TabPFN unavailable; using RandomForest fallback: {exc}")
    return RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)


def _model_name(model) -> str:
    if TabPFNClassifier is not None and isinstance(model, TabPFNClassifier):
        return "TabPFN"
    return "RandomForest fallback"


def _build_count_training_data(
    history: pd.DataFrame,
    max_horizon_days: int,
) -> tuple[pd.DataFrame, pd.Series]:
    by_date = history.set_index("date").sort_index()
    if not by_date.index.is_unique:
        raise ValueError("Activity history contains duplicate dates; refusing to train a count forecast.")

    rows: list[dict[str, float | int]] = []
    targets: list[float] = []
    for horizon in range(1, max_horizon_days + 1):
        for target_date, target in by_date.iterrows():
            origin_date = target_date - pd.Timedelta(days=horizon)
            if origin_date not in by_date.index:
                continue
            origin = by_date.loc[origin_date]
            row: dict[str, float | int] = {
                "forecast_horizon_days": horizon,
                "target_day_of_week": int(target["day_of_week"]),
                "target_weekend": int(target["weekend"]),
                "target_month": int(target["month"]),
                "target_day_of_year": int(target["day_of_year"]),
                "target_public_holiday": int(target["public_holiday"]),
                "origin_activity_count": float(origin["activity_count"]),
                "origin_lag_7": float(origin["lag_7"]),
                "origin_rolling_7day_mean": float(origin["rolling_7day_mean"]),
            }
            row.update(
                {
                    f"target_{column}": float(target[column])
                    for column in COUNT_WEATHER_COLUMNS
                }
            )
            rows.append(row)
            targets.append(float(target["activity_count"]))

    X = pd.DataFrame(rows, columns=COUNT_FEATURE_COLUMNS)
    y = pd.Series(targets, dtype=float)
    valid_rows = X.notna().all(axis=1) & y.notna()
    return X.loc[valid_rows].reset_index(drop=True), y.loc[valid_rows].reset_index(drop=True)


def _forecast_stale_activity(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    tomorrow: pd.Timestamp,
    activity_through: pd.Timestamp,
) -> dict:
    horizon_days = int((tomorrow - activity_through).days)
    if horizon_days < 1:
        raise ValueError("The activity data date is not before the forecast target date.")
    if horizon_days > MAX_STALE_FORECAST_HORIZON_DAYS:
        raise ValueError(
            f"Divvy activity data is {horizon_days} days behind the forecast date; "
            f"the stale-data estimate is limited to {MAX_STALE_FORECAST_HORIZON_DAYS} days."
        )

    target_weather = weather.loc[weather["date"] == tomorrow]
    if target_weather.empty:
        raise ValueError(f"Open-Meteo has no forecast row for tomorrow ({tomorrow.date()}).")
    target_weather = target_weather.iloc[0]

    X_train, y_train = _build_count_training_data(history, horizon_days)
    if len(X_train) < 40:
        raise ValueError(
            f"Only {len(X_train)} historical examples are available for a {horizon_days}-day activity estimate; "
            "at least 40 are required."
        )

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X_train, y_train)

    origin = history.iloc[-1]
    target_features: dict[str, float | int] = {
        "forecast_horizon_days": horizon_days,
        "target_day_of_week": tomorrow.dayofweek,
        "target_weekend": int(tomorrow.dayofweek in (5, 6)),
        "target_month": tomorrow.month,
        "target_day_of_year": tomorrow.dayofyear,
        "target_public_holiday": int(tomorrow.date() in holidays.US()),
        "origin_activity_count": float(origin["activity_count"]),
        "origin_lag_7": float(origin["lag_7"]),
        "origin_rolling_7day_mean": float(origin["rolling_7day_mean"]),
    }
    target_features.update(
        {
            f"target_{column}": float(target_weather[column])
            for column in COUNT_WEATHER_COLUMNS
        }
    )
    target_row = pd.DataFrame([target_features], columns=COUNT_FEATURE_COLUMNS)
    tree_predictions = np.asarray(
        [tree.predict(target_row.to_numpy())[0] for tree in model.estimators_],
        dtype=float,
    )
    estimated_count = max(0, int(round(float(model.predict(target_row)[0]))))
    spread_low = max(0, int(np.floor(np.quantile(tree_predictions, 0.1))))
    spread_high = max(estimated_count, int(np.ceil(np.quantile(tree_predictions, 0.9))))
    staleness_days = max(0, int((tomorrow - pd.Timedelta(days=1) - activity_through).days))

    return {
        "target_date": tomorrow.strftime("%Y-%m-%d"),
        "activity_data_through": activity_through.strftime("%Y-%m-%d"),
        "forecast_kind": "stale_count_estimate",
        "activity_freshness": "stale",
        "activity_staleness_days": staleness_days,
        "forecast_horizon_days": horizon_days,
        "estimated_activity_count": estimated_count,
        "model_range_low": spread_low,
        "model_range_high": spread_high,
        "range_method": "10th-90th percentile across Random Forest trees; not a calibrated prediction interval",
        "model_used": "RandomForest direct multi-horizon count estimate",
    }


def forecast_tomorrow(feature_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict:
    history = feature_df.sort_values("date").reset_index(drop=True).copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    tomorrow = pd.Timestamp.now(tz="America/Chicago").tz_localize(None).normalize() + pd.Timedelta(days=1)
    activity_through = history["date"].max()
    activity_by_date = history.set_index("date")["activity_count"]

    if activity_through != tomorrow - pd.Timedelta(days=1):
        payload = _forecast_stale_activity(history, weather_df, tomorrow, activity_through)
        (RESULTS_DIR / "tomorrow_prediction.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    lag_1_date = tomorrow - pd.Timedelta(days=1)
    lag_7_date = tomorrow - pd.Timedelta(days=7)
    rolling_dates = pd.date_range(end=lag_1_date, periods=7, freq="D")
    if lag_1_date not in activity_by_date.index or lag_7_date not in activity_by_date.index:
        raise ValueError("Recent Divvy data does not include the dates needed to create activity lag features.")
    if not set(rolling_dates).issubset(activity_by_date.index):
        raise ValueError("Recent Divvy data has gaps in the seven-day activity window.")

    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    target_weather = weather.loc[weather["date"] == tomorrow]
    if target_weather.empty:
        raise ValueError(f"Open-Meteo has no forecast row for tomorrow ({tomorrow.date()}).")
    target_weather = target_weather.iloc[0]
    target_features = {
        "day_of_week": tomorrow.dayofweek,
        "weekend": int(tomorrow.dayofweek in (5, 6)),
        "month": tomorrow.month,
        "day_of_year": tomorrow.dayofyear,
        "public_holiday": int(tomorrow.date() in holidays.US()),
        "temperature_2m_max": target_weather["temperature_2m_max"],
        "temperature_2m_min": target_weather["temperature_2m_min"],
        "precipitation_sum": target_weather["precipitation_sum"],
        "precipitation_hours": target_weather["precipitation_hours"],
        "wind_speed_10m_max": target_weather["wind_speed_10m_max"],
        "daylight_duration": target_weather["daylight_duration"],
        "cloud_cover_mean": target_weather["cloud_cover_mean"],
        "lag_1": activity_by_date.loc[lag_1_date],
        "lag_7": activity_by_date.loc[lag_7_date],
        "rolling_7day_mean": activity_by_date.loc[rolling_dates].mean(),
    }
    target_row = pd.DataFrame([target_features], columns=FEATURE_COLUMNS)
    X = history[FEATURE_COLUMNS]
    y_high = history["high_activity_day"].astype(int)
    y_peak = history["peak_window"]
    if y_high.nunique() < 2:
        raise ValueError("The available training data contains only one activity class.")

    high_model = _fit_model(X, y_high)
    probability = high_model.predict_proba(target_row)[0]
    classes = list(high_model.classes_)
    high_probability = float(probability[classes.index(1)]) if 1 in classes else 0.0
    peak_model = _fit_model(X, y_peak)
    peak_window = peak_model.predict(target_row)[0]
    payload = {
        "target_date": tomorrow.strftime("%Y-%m-%d"),
        "activity_data_through": activity_through.strftime("%Y-%m-%d"),
        "forecast_kind": "high_activity_classifier",
        "activity_freshness": "fresh",
        "activity_staleness_days": 0,
        "forecast_horizon_days": 1,
        "high_activity_day": int(high_probability >= 0.5),
        "high_activity_probability": high_probability,
        "peak_window": str(peak_window),
        "model_used": _model_name(high_model),
    }
    (RESULTS_DIR / "tomorrow_prediction.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
