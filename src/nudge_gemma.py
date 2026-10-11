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
    target_date = prediction.get("target_date")
    weather_today = None
    if target_date and not weather_df.empty:
        dates = pd.to_datetime(weather_df["date"]).dt.strftime("%Y-%m-%d")
        matching = weather_df.loc[dates == target_date]
        if not matching.empty:
            weather_today = matching.iloc[0]
    if weather_today is None and not weather_df.empty:
        weather_today = weather_df.iloc[-1]
    rain_hint = "dry" if weather_today is None or float(weather_today.get("precipitation_sum", 0) or 0) < 1 else "a little wet"
    if prediction.get("high_activity_day", 1):
        return f"Tomorrow looks like a good one: {rain_hint} and the city is busiest in the {peak}, so take a short walk then."
    return f"Tomorrow is less active overall: {rain_hint} and the busy window is the {peak}. Keep it light if you go out."


def _stale_estimate_nudge(prediction: dict, weather_df: pd.DataFrame) -> str:
    target_date = prediction["target_date"]
    estimate = int(prediction["estimated_activity_count"])
    through = prediction["activity_data_through"]
    weather_today = _weather_for_date(weather_df, target_date)

    weather_parts: list[str] = []
    if weather_today is not None:
        temperature = weather_today.get("temperature_2m_max")
        precipitation = weather_today.get("precipitation_sum")
        if pd.notna(temperature):
            weather_parts.append(f"forecast high near {temperature:.0f}°C")
        if pd.notna(precipitation) and float(precipitation) >= 0.5:
            weather_parts.append("a chance of light rain")

    if weather_parts:
        weather_description = weather_parts[0]
        if len(weather_parts) > 1:
            weather_description += f", with {weather_parts[1]}"
        weather_hint = f"Tomorrow looks like {weather_description}."
    else:
        weather_hint = "Check tomorrow’s weather before heading out."

    return (
        f"{weather_hint} A casual walk or ride could be nice. Divvy’s ride count (~{estimate:,}) "
        f"is an estimate from data through {through}, not observed."
    )


def _weather_for_date(weather_df: pd.DataFrame, target_date: str | None) -> pd.Series | None:
    if not target_date or weather_df.empty:
        return None
    dates = pd.to_datetime(weather_df["date"]).dt.strftime("%Y-%m-%d")
    matching = weather_df.loc[dates == target_date]
    return matching.iloc[0] if not matching.empty else None


def generate_nudge(prediction: dict, weather_df: pd.DataFrame) -> str:
    is_stale_estimate = prediction.get("forecast_kind") == "stale_count_estimate"
    if is_stale_estimate:
        fallback = _stale_estimate_nudge(prediction, weather_df)
    else:
        fallback = _fallback_nudge(prediction, weather_df)

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemma-3-4b-it")
    if not api_key:
        return fallback

    target_date = prediction.get("target_date")
    weather_today = _weather_for_date(weather_df, target_date)
    if weather_today is None and not weather_df.empty:
        weather_today = weather_df.iloc[-1]
    rain_hint = "dry" if weather_today is None or float(weather_today.get("precipitation_sum", 0) or 0) < 1 else "a little wet"
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if is_stale_estimate:
        stale_weather_parts: list[str] = []
        if weather_today is not None:
            temperature = weather_today.get("temperature_2m_max")
            precipitation = weather_today.get("precipitation_sum")
            if pd.notna(temperature):
                stale_weather_parts.append(f"forecast high {float(temperature):.0f}°C")
            if pd.notna(precipitation):
                if float(precipitation) >= 0.5:
                    stale_weather_parts.append(f"{float(precipitation):.1f} mm precipitation possible")
                else:
                    stale_weather_parts.append("little precipitation expected")
        stale_weather_hint = ", ".join(stale_weather_parts) or "not available"
        prompt = (
            f"{system_prompt.replace('Mention the weather and the busiest outdoor window.', 'Mention the weather; do not invent a busiest outdoor window.')}\n"
            "Write in the same warm, conversational style as a friendly outdoor suggestion. Mention the weather and "
            "invite a relaxed walk or casual ride. Include the estimated ride count and say it is a model estimate, "
            "not observed activity, with the data-through date. Do not imply the city will be busy or invent a peak time. "
            "Use only these facts:\n"
            f"Target date: {target_date or 'tomorrow'}\n"
            f"Estimated ride count: {prediction['estimated_activity_count']}\n"
            f"Activity data through: {prediction['activity_data_through']}\n"
            f"Weather hint: {stale_weather_hint}\n"
            "Keep it under 220 characters."
        )
    else:
        prompt = (
            f"{system_prompt}\n"
            f"Forecast for {target_date or 'tomorrow'}: {rain_hint}. "
            f"Probability of a high outdoor-activity day: {prediction.get('high_activity_probability', 0.5):.0%}. "
            f"Peak window: {prediction.get('peak_window', 'evening')}."
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

    return fallback
