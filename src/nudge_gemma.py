from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import CONFIG_DIR

SYSTEM_PROMPT_PATH = CONFIG_DIR / "nudge_system_prompt.txt"


def _default_prompt() -> str:
    return (
        "You are a concise city outdoor forecaster. Write a short, specific, non-preachy nudge in 1-2 sentences. "
        "Never mention health claims or guilt. Mention the weather and the busiest outdoor window. Keep it under 220 characters."
    )


if not SYSTEM_PROMPT_PATH.exists():
    SYSTEM_PROMPT_PATH.write_text(_default_prompt(), encoding="utf-8")


def _fallback_nudge(prediction: dict, weather_df: pd.DataFrame) -> str:
    peak = prediction.get("peak_window", "evening")
    prob = prediction.get("high_activity_probability", 0.5)
    weather_today = weather_df.iloc[-1] if not weather_df.empty else None
    rain_hint = "dry" if weather_today is None or float(weather_today.get("precipitation_sum", 0) or 0) < 1 else "a little wet"
    if prediction.get("high_activity_day", 1):
        return f"Tomorrow looks like a good one: {rain_hint} and the city is busiest in the {peak}, so take a short walk then."
    return f"Tomorrow is less active overall: {rain_hint} and the busy window is the {peak}. Keep it light if you go out."


def generate_nudge(prediction: dict, weather_df: pd.DataFrame) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemma-3-4b-it")
    city = os.getenv("CITY", "Chicago")
    if not api_key:
        return _fallback_nudge(prediction, weather_df)

    weather_today = weather_df.iloc[-1] if not weather_df.empty else None
    rain_hint = "dry" if weather_today is None or float(weather_today.get("precipitation_sum", 0) or 0) < 1 else "a little wet"
    prompt = (
        f"City: {city}. "
        f"Tomorrow forecast: {rain_hint}. "
        f"Probability of a high outdoor-activity day: {prediction.get('high_activity_probability', 0.5):.0%}. "
        f"Peak window: {prediction.get('peak_window', 'evening')}. "
        "Write a specific, encouraging, non-preachy two-sentence nudge under 220 characters."
    )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=[prompt])
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates"):
            text = response.candidates[0].content.parts[0].text
        if text:
            return text.strip().replace("\n", " ")[:220]
    except Exception:
        pass

    return _fallback_nudge(prediction, weather_df)
