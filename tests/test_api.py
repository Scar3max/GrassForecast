import json

import pandas as pd

import api


def test_weather_suggestion_is_available_when_activity_data_is_stale(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    cache_dir.mkdir()
    results_dir.mkdir()
    tomorrow = (
        pd.Timestamp.now(tz="America/Chicago")
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(days=1)
    )
    (cache_dir / "weather_chicago.json").write_text(
        json.dumps(
            [
                {
                    "date": tomorrow.strftime("%Y-%m-%d"),
                    "temperature_2m_max": 28.9,
                    "temperature_2m_min": 14.7,
                    "precipitation_sum": 0.8,
                    "precipitation_hours": 2,
                    "wind_speed_10m_max": 22.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    (results_dir / "status.json").write_text(
        json.dumps({"status": "unavailable", "message": "Activity data is stale."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(api, "RESULTS_DIR", results_dir)

    payload = api._dashboard_payload()

    assert payload["state"] == "unavailable"
    assert payload["forecast"] is None
    assert payload["weather_suggestion"]["date"] == tomorrow.strftime("%Y-%m-%d")
    assert "hourly forecast" in payload["weather_suggestion"]["suggestion"]
    assert "not a prediction of city bike activity" in payload["weather_suggestion"]["basis"]


def test_weather_suggestion_handles_missing_or_unusable_weather():
    assert api.build_weather_suggestion(None) is None
    assert api.build_weather_suggestion({"date": "2026-10-12"}) is None
    assert api.build_weather_suggestion({"temperature_2m_max": 20}) is None


def test_activity_freshness_distinguishes_fresh_stale_and_missing_data():
    today = pd.Timestamp("2026-10-11")

    assert api.build_activity_freshness("2026-10-11", today) == {
        "status": "fresh",
        "days_behind": 0,
    }
    assert api.build_activity_freshness("2026-10-06", today) == {
        "status": "stale",
        "days_behind": 5,
    }
    assert api.build_activity_freshness(None, today) == {
        "status": "unavailable",
        "days_behind": None,
    }


def test_stale_activity_estimate_is_exposed_with_freshness_note(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    cache_dir.mkdir()
    results_dir.mkdir()
    target_date = (
        pd.Timestamp.now(tz="America/Chicago")
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(days=1)
    )
    activity_through = target_date - pd.Timedelta(days=6)
    (results_dir / "features.csv").write_text(
        f"date,activity_count\n{activity_through:%Y-%m-%d},1200\n",
        encoding="utf-8",
    )
    (results_dir / "status.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    (results_dir / "tomorrow_prediction.json").write_text(
        json.dumps(
            {
                "target_date": target_date.strftime("%Y-%m-%d"),
                "activity_data_through": activity_through.strftime("%Y-%m-%d"),
                "forecast_kind": "stale_count_estimate",
                "activity_freshness": "stale",
                "activity_staleness_days": 5,
                "forecast_horizon_days": 6,
                "estimated_activity_count": 1100,
                "model_range_low": 900,
                "model_range_high": 1300,
                "range_method": "tree spread",
                "model_used": "test model",
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "nudge.json").write_text(
        json.dumps(
            {
                "text": "Tomorrow’s ride count is estimated; Divvy data is through Oct 6.",
                "audio_path": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(api, "RESULTS_DIR", results_dir)

    payload = api._dashboard_payload()

    assert payload["state"] == "estimated"
    assert payload["forecast"]["forecast_kind"] == "stale_count_estimate"
    assert payload["activity_freshness"] == {"status": "stale", "days_behind": 5}
    assert "6-day forecast" in payload["message"]
    assert payload["nudge"]["text"].startswith("Tomorrow’s ride count is estimated")
