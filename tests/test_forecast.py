import pandas as pd
import pytest

from src import forecast_tabpfn
from run_daily import _print_prediction_summary


def _activity_history(through: pd.Timestamp, days: int = 180) -> pd.DataFrame:
    dates = pd.date_range(end=through, periods=days, freq="D")
    rides = [900 + (date.dayofweek * 170) + ((index % 11) * 13) for index, date in enumerate(dates)]
    frame = pd.DataFrame(
        {
            "date": dates,
            "activity_count": rides,
            "high_activity_day": [index % 2 for index in range(days)],
            "peak_window": ["afternoon" if index % 2 else "evening" for index in range(days)],
            "day_of_week": dates.dayofweek,
            "weekend": [int(date.dayofweek in (5, 6)) for date in dates],
            "month": dates.month,
            "day_of_year": dates.dayofyear,
            "public_holiday": 0,
            "temperature_2m_max": [18 + (index % 8) for index in range(days)],
            "temperature_2m_min": [8 + (index % 5) for index in range(days)],
            "precipitation_sum": [float(index % 3) for index in range(days)],
            "precipitation_hours": [float(index % 2) for index in range(days)],
            "wind_speed_10m_max": [12 + (index % 6) for index in range(days)],
            "daylight_duration": [42000 - index * 10 for index in range(days)],
            "cloud_cover_mean": [20 + (index % 60) for index in range(days)],
            "lag_1": [rides[max(0, index - 1)] for index in range(days)],
            "lag_7": [rides[max(0, index - 7)] for index in range(days)],
            "rolling_7day_mean": [
                sum(rides[max(0, index - 7):index]) / max(len(rides[max(0, index - 7):index]), 1)
                for index in range(days)
            ],
        }
    )
    return frame


def _target_weather(target_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": target_date,
                "temperature_2m_max": 20,
                "temperature_2m_min": 10,
                "precipitation_sum": 0.2,
                "precipitation_hours": 1,
                "wind_speed_10m_max": 15,
                "daylight_duration": 40000,
                "cloud_cover_mean": 40,
            }
        ]
    )


def test_forecast_tomorrow_estimates_ride_count_for_stale_activity(tmp_path, monkeypatch):
    target = (
        pd.Timestamp.now(tz="America/Chicago")
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(days=1)
    )
    history = _activity_history(target - pd.Timedelta(days=6))
    monkeypatch.setattr(forecast_tabpfn, "RESULTS_DIR", tmp_path)

    forecast = forecast_tabpfn.forecast_tomorrow(history, _target_weather(target))

    assert forecast["forecast_kind"] == "stale_count_estimate"
    assert forecast["activity_freshness"] == "stale"
    assert forecast["activity_staleness_days"] == 5
    assert forecast["forecast_horizon_days"] == 6
    assert forecast["estimated_activity_count"] > 0
    assert forecast["model_range_low"] <= forecast["estimated_activity_count"] <= forecast["model_range_high"]
    assert "not a calibrated prediction interval" in forecast["range_method"]
    assert (tmp_path / "tomorrow_prediction.json").exists()


def test_fresh_activity_keeps_high_activity_classifier(tmp_path, monkeypatch):
    target = (
        pd.Timestamp.now(tz="America/Chicago")
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(days=1)
    )
    history = _activity_history(target - pd.Timedelta(days=1))
    monkeypatch.setattr(forecast_tabpfn, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(forecast_tabpfn, "TabPFNClassifier", None)

    forecast = forecast_tabpfn.forecast_tomorrow(history, _target_weather(target))

    assert forecast["forecast_kind"] == "high_activity_classifier"
    assert forecast["activity_freshness"] == "fresh"
    assert forecast["activity_staleness_days"] == 0
    assert forecast["forecast_horizon_days"] == 1
    assert 0 <= forecast["high_activity_probability"] <= 1


def test_stale_activity_estimate_is_withheld_beyond_maximum_horizon(tmp_path, monkeypatch):
    target = (
        pd.Timestamp.now(tz="America/Chicago")
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(days=1)
    )
    history = _activity_history(target - pd.Timedelta(days=31))
    monkeypatch.setattr(forecast_tabpfn, "RESULTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="limited to 30 days"):
        forecast_tabpfn.forecast_tomorrow(history, _target_weather(target))


def test_stale_forecast_summary_uses_estimate_fields(capsys):
    _print_prediction_summary(
        {
            "forecast_kind": "stale_count_estimate",
            "estimated_activity_count": 24525,
            "model_range_low": 22651,
            "model_range_high": 26850,
            "target_date": "2026-10-12",
            "activity_data_through": "2026-09-30",
            "activity_staleness_days": 11,
            "forecast_horizon_days": 12,
            "model_used": "RandomForest direct multi-horizon count estimate",
        }
    )

    output = capsys.readouterr().out
    assert "24,525 (model estimate, not observed)" in output
    assert "high_activity_day" not in output
    assert "12-day forecast lead" in output
