from src.nudge_gemma import _fallback_nudge


def test_nudge_length_limit():
    prediction = {"high_activity_day": 1, "high_activity_probability": 0.8, "peak_window": "evening"}
    nudge = _fallback_nudge(prediction, __import__("pandas").DataFrame([{"precipitation_sum": 0.1}]))
    assert len(nudge) <= 220
    sentence_count = nudge.count(".") + nudge.count("!")
    assert sentence_count <= 2
