from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.config import CACHE_DIR


def _synthetic_weather(days: int = 14) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D")
    frame = pd.DataFrame({
        "date": dates,
        "temperature_2m_max": [20 + i % 12 for i in range(days)],
        "temperature_2m_min": [12 + i % 8 for i in range(days)],
        "precipitation_sum": [0.1 + (i % 3) * 0.2 for i in range(days)],
        "precipitation_hours": [0.5 + (i % 4) for i in range(days)],
        "wind_speed_10m_max": [8 + (i % 7) for i in range(days)],
        "uv_index_max": [4 + (i % 6) for i in range(days)],
        "daylight_duration": [600 + (i % 100) for i in range(days)],
        "cloud_cover_mid": [35 + (i % 25) for i in range(days)],
    })
    return frame


def fetch_weather_snapshot(city: str, latitude: float, longitude: float) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"weather_{city.lower().replace(' ', '_')}.json"
    if cache_path.exists():
        return pd.read_json(cache_path)

    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,wind_speed_10m_max,uv_index_max,daylight_duration,cloud_cover_mid",
            "timezone": "auto",
            "forecast_days": 14,
        }
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily", {})
        if not daily:
            raise ValueError("No daily weather data returned")

        frame = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "temperature_2m_max": daily.get("temperature_2m_max", []),
            "temperature_2m_min": daily.get("temperature_2m_min", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "precipitation_hours": daily.get("precipitation_hours", []),
            "wind_speed_10m_max": daily.get("wind_speed_10m_max", []),
            "uv_index_max": daily.get("uv_index_max", []),
            "daylight_duration": daily.get("daylight_duration", []),
            "cloud_cover_mid": daily.get("cloud_cover_mid", []),
        }).dropna(subset=["date"]).reset_index(drop=True)
    except Exception:
        frame = _synthetic_weather()

    frame = frame.copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    cache_path.write_text(json.dumps(frame.to_dict(orient="records")))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame
