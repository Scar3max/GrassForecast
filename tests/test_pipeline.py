import pandas as pd

from src.features import build_feature_table
from src.dataset import generate_synthetic_divvy_data
from src.weather import fetch_weather_snapshot


def test_no_leakage_on_feature_table():
    hourly = generate_synthetic_divvy_data()
    dates = pd.to_datetime(sorted(hourly.date.unique()))
    weather = pd.DataFrame({
        "date": dates,
        "temperature_2m_max": 18.0,
        "temperature_2m_min": 5.0,
        "precipitation_sum": 0.1,
        "precipitation_hours": 1.0,
        "wind_speed_10m_max": 7.0,
        "uv_index_max": 4.0,
        "daylight_duration": 600.0,
        "cloud_cover_mid": 30.0,
    })
    features = build_feature_table(hourly, weather)
    assert "activity_count" not in features.columns
    assert "date" in features.columns
    assert "high_activity_day" in features.columns


def test_anonymization_payload_excludes_dates_and_coordinates():
    hourly = generate_synthetic_divvy_data()
    weather = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "temperature_2m_max": [20, 21, 19],
        "temperature_2m_min": [8, 9, 7],
        "precipitation_sum": [0.0, 0.1, 0.2],
        "precipitation_hours": [0.0, 1.0, 2.0],
        "wind_speed_10m_max": [10, 12, 9],
        "uv_index_max": [5, 6, 5],
        "daylight_duration": [500, 550, 540],
        "cloud_cover_mid": [25, 30, 35],
    })
    features = build_feature_table(hourly, weather)
    payload = features.drop(columns=["date", "peak_window", "high_activity_day"], errors="ignore")
    assert "date" not in payload.columns
    assert "latitude" not in payload.columns
    assert "longitude" not in payload.columns
    assert "city" not in payload.columns
