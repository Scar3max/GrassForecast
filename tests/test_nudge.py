import pandas as pd
import sys
import types

from src.nudge_gemma import generate_nudge


def test_stale_estimate_gets_a_clear_fallback_nudge_without_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    prediction = {
        "forecast_kind": "stale_count_estimate",
        "target_date": "2026-10-12",
        "activity_data_through": "2026-09-30",
        "estimated_activity_count": 24525,
    }
    weather = pd.DataFrame(
        [
            {
                "date": "2026-10-12",
                "temperature_2m_max": 22.4,
                "precipitation_sum": 0.8,
            }
        ]
    )

    nudge = generate_nudge(prediction, weather)

    assert "casual walk or ride could be nice" in nudge
    assert "ride count (~24,525)" in nudge
    assert "not observed" in nudge
    assert "2026-09-30" in nudge
    assert "forecast high near 22°C" in nudge
    assert "chance of light rain" in nudge
    assert len(nudge) <= 220


def test_stale_estimate_nudge_handles_missing_target_weather(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    prediction = {
        "forecast_kind": "stale_count_estimate",
        "target_date": "2026-10-12",
        "activity_data_through": "2026-09-30",
        "estimated_activity_count": 24525,
    }

    nudge = generate_nudge(prediction, pd.DataFrame())

    assert "Check tomorrow’s weather" in nudge
    assert "casual walk or ride could be nice" in nudge


def test_stale_estimate_uses_gemini_with_friendly_grounded_prompt(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents):
            captured["model"] = model
            captured["prompt"] = contents[0]
            return types.SimpleNamespace(
                text="A gentle walk could be lovely tomorrow. Divvy’s roughly 24,525 rides are an estimate from data through Sep 30, not observed."
            )

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemma-test")
    prediction = {
        "forecast_kind": "stale_count_estimate",
        "target_date": "2026-10-12",
        "activity_data_through": "2026-09-30",
        "estimated_activity_count": 24525,
    }
    weather = pd.DataFrame(
        [
            {
                "date": "2026-10-12",
                "temperature_2m_max": 22.4,
                "precipitation_sum": 0.8,
            }
        ]
    )

    nudge = generate_nudge(prediction, weather)

    assert captured["model"] == "gemma-test"
    assert captured["api_key"] == "test-key"
    assert "warm, conversational style" in captured["prompt"]
    assert "do not invent a busiest outdoor window" in captured["prompt"]
    assert "Weather hint: forecast high 22°C, 0.8 mm precipitation possible" in captured["prompt"]
    assert "24,525" in nudge
