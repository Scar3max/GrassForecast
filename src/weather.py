from __future__ import annotations

import json

import pandas as pd
import requests

from src.config import CACHE_DIR

DAILY_VARIABLES = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,"
    "wind_speed_10m_max,uv_index_max,daylight_duration,cloud_cover_mean"
)
WEATHER_COLUMNS = [
    "date",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "uv_index_max",
    "daylight_duration",
    "cloud_cover_mean",
]


def _daily_frame(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily", {})
    if not daily or not daily.get("time"):
        raise ValueError("Open-Meteo returned no daily weather data.")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "temperature_2m_max": daily.get("temperature_2m_max", []),
            "temperature_2m_min": daily.get("temperature_2m_min", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "precipitation_hours": daily.get("precipitation_hours", []),
            "wind_speed_10m_max": daily.get("wind_speed_10m_max", []),
            "uv_index_max": daily.get("uv_index_max", []),
            "daylight_duration": daily.get("daylight_duration", []),
            "cloud_cover_mean": daily.get("cloud_cover_mean", []),
        }
    ).dropna(subset=["date"]).reset_index(drop=True)


def _request_daily(endpoint: str, params: dict) -> pd.DataFrame:
    response = requests.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    return _daily_frame(response.json())


def fetch_weather_snapshot(
    city: str,
    latitude: float,
    longitude: float,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Fetch historical weather for training dates and the current forecast."""
    cache_path = CACHE_DIR / f"weather_{city.lower().replace(' ', '_')}.json"
    history_start = pd.Timestamp(start_date).normalize()
    history_end = pd.Timestamp(end_date).normalize()
    tomorrow = pd.Timestamp.now(tz="America/Chicago").tz_localize(None).normalize() + pd.Timedelta(days=1)

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_json(cache_path)
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
            cached_dates = set(cached["date"])
            required_history_dates = set(pd.date_range(history_start, history_end, freq="D"))
            if required_history_dates.issubset(cached_dates) and tomorrow in cached_dates:
                return cached.sort_values("date").reset_index(drop=True)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": DAILY_VARIABLES,
        "timezone": "America/Chicago",
        "start_date": history_start.strftime("%Y-%m-%d"),
        "end_date": history_end.strftime("%Y-%m-%d"),
    }
    try:
        historical = _request_daily("https://archive-api.open-meteo.com/v1/archive", params)
    except (requests.RequestException, ValueError) as exc:
        if not cached.empty and cached["date"].min() <= history_start and cached["date"].max() >= history_end:
            historical = cached.loc[cached["date"].between(history_start, history_end)]
            print(f"Open-Meteo archive unavailable; using cached historical weather: {exc}")
        else:
            raise RuntimeError(f"Could not retrieve historical weather from Open-Meteo: {exc}") from exc

    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": DAILY_VARIABLES,
        "timezone": "America/Chicago",
        "forecast_days": 14,
    }
    try:
        forecast = _request_daily("https://api.open-meteo.com/v1/forecast", forecast_params)
    except (requests.RequestException, ValueError) as exc:
        if not cached.empty and tomorrow in set(cached["date"]):
            forecast = cached.loc[cached["date"] >= tomorrow]
            print(f"Open-Meteo forecast unavailable; using cached forecast: {exc}")
        else:
            raise RuntimeError(f"Could not retrieve tomorrow's weather forecast from Open-Meteo: {exc}") from exc

    combined = pd.concat([historical, forecast], ignore_index=True)
    combined = combined.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    missing_dates = set(pd.date_range(history_start, history_end, freq="D")) - set(combined["date"])
    if missing_dates:
        raise RuntimeError(
            f"Open-Meteo weather is missing {len(missing_dates)} historical dates; refusing to evaluate on incomplete data."
        )
    if tomorrow not in set(combined["date"]):
        raise RuntimeError("Open-Meteo response did not include tomorrow's weather forecast.")

    cache_frame = combined.copy()
    cache_frame["date"] = cache_frame["date"].dt.strftime("%Y-%m-%d")
    cache_path.write_text(json.dumps(cache_frame.to_dict(orient="records")), encoding="utf-8")
    return combined
