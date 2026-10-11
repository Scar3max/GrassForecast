from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


US_HOLIDAYS = holidays.US()


def _assign_peak_window(hourly_df: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_df.copy()
    hourly["window"] = hourly["hour"].map(
        lambda hour: (
            "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening"
        )
    )
    daily_window = hourly.groupby(["date", "window"], as_index=False)["activity_count"].sum()
    peak_rows = daily_window.sort_values(["date", "activity_count"], ascending=[True, False]).drop_duplicates("date")
    peak_rows = peak_rows[["date", "window"]].rename(columns={"window": "peak_window"})
    return peak_rows


def build_feature_table(hourly_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    daily_activity = hourly_df.groupby("date", as_index=False)["activity_count"].sum()
    daily_activity["date"] = pd.to_datetime(daily_activity["date"]).dt.normalize()
    full_date_range = pd.date_range(daily_activity["date"].min(), daily_activity["date"].max(), freq="D")
    daily_activity = (
        daily_activity.set_index("date")
        .reindex(full_date_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()

    merged = daily_activity.merge(weather_df, on="date", how="left")
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)
    merged["month"] = merged["date"].dt.month
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["public_holiday"] = merged["date"].map(lambda value: 1 if value in US_HOLIDAYS else 0)
    merged["lag_1"] = merged["activity_count"].shift(1)
    merged["lag_7"] = merged["activity_count"].shift(7)
    merged["rolling_7day_mean"] = merged["activity_count"].shift(1).rolling(7, min_periods=1).mean()
    merged["historical_activity_threshold"] = (
        merged["activity_count"].shift(1).rolling(90, min_periods=30).quantile(2 / 3)
    )
    merged["high_activity_day"] = np.where(
        merged["historical_activity_threshold"].notna(),
        (merged["activity_count"] > merged["historical_activity_threshold"]).astype(int),
        np.nan,
    )

    peak_window_df = _assign_peak_window(hourly_df)
    merged = merged.merge(peak_window_df, on="date", how="left")

    feature_columns = [
        "day_of_week", "weekend", "month", "day_of_year", "public_holiday",
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "precipitation_hours", "wind_speed_10m_max",
        "daylight_duration", "cloud_cover_mean", "lag_1", "lag_7", "rolling_7day_mean"
    ]

    features = merged[
        feature_columns + ["date", "activity_count", "high_activity_day", "peak_window"]
    ].copy()
    features = features.dropna(
        subset=feature_columns + ["high_activity_day", "peak_window"]
    ).reset_index(drop=True)
    features["high_activity_day"] = features["high_activity_day"].astype(int)
    features.to_csv(RESULTS_DIR / "features.csv", index=False)
    return features
