from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import BASE_DIR, CACHE_DIR, RESULTS_DIR

app = FastAPI(title="Grass Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_run_lock = threading.Lock()
_run_state: dict[str, str | None] = {
    "state": "idle",
    "message": None,
    "started_at": None,
    "finished_at": None,
}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _weather_value(weather: dict, key: str) -> float | None:
    value = weather.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_weather_suggestion(weather: dict | None) -> dict | None:
    if not weather or not weather.get("date"):
        return None

    temperature = _weather_value(weather, "temperature_2m_max")
    precipitation = _weather_value(weather, "precipitation_sum")
    wind_speed = _weather_value(weather, "wind_speed_10m_max")
    if temperature is None and precipitation is None and wind_speed is None:
        return None

    if temperature is not None and temperature >= 32:
        title = "A hot day is expected"
        suggestion = "If you head out, consider a brief outing during a cooler part of the day; check the hourly forecast first."
    elif temperature is not None and temperature >= 27:
        title = "A warm day is expected"
        suggestion = "A short, easy outing may feel better during a cooler part of the day; check the hourly forecast before choosing a time."
    elif temperature is not None and temperature <= 5:
        title = "A cold day is expected"
        suggestion = "If it feels comfortable for you, consider a brief outing with warm layers."
    else:
        title = "A day to make a little room for outside"
        suggestion = "A walk or casual ride could fit the forecast; choose what feels comfortable."

    if precipitation is not None and precipitation >= 5:
        weather_note = "Rain is expected, so consider a rain layer or an indoor alternative."
    elif precipitation is not None and precipitation >= 0.5:
        weather_note = "Light precipitation is possible; check the hourly forecast before heading out."
    else:
        weather_note = "Little precipitation is forecast."

    if wind_speed is not None and wind_speed >= 30:
        wind_note = "Peak winds may be strong, so a sheltered route could feel more comfortable."
    elif wind_speed is not None and wind_speed >= 20:
        wind_note = "It may be breezy, so choose a route that feels comfortable."
    else:
        wind_note = None

    return {
        "date": str(weather["date"]),
        "title": title,
        "suggestion": f"{suggestion} {weather_note}" + (f" {wind_note}" if wind_note else ""),
        "basis": "Based on the daily weather forecast only—not a prediction of city bike activity.",
    }


def build_activity_freshness(
    activity_through: str | None,
    local_today: pd.Timestamp | None = None,
) -> dict:
    if not activity_through:
        return {"status": "unavailable", "days_behind": None}

    today = local_today or pd.Timestamp.now(tz="America/Chicago").tz_localize(None).normalize()
    latest_date = pd.Timestamp(activity_through).normalize()
    days_behind = max(0, int((today - latest_date).days))
    return {
        "status": "fresh" if days_behind == 0 else "stale",
        "days_behind": days_behind,
    }


def _dashboard_payload() -> dict:
    status = _read_json(RESULTS_DIR / "status.json") or {}
    metrics = _read_json(RESULTS_DIR / "metrics.json")
    forecast = _read_json(RESULTS_DIR / "tomorrow_prediction.json")
    nudge = _read_json(RESULTS_DIR / "nudge.json")

    history: list[dict[str, int | str]] = []
    feature_path = RESULTS_DIR / "features.csv"
    if feature_path.exists():
        try:
            frame = pd.read_csv(feature_path, usecols=["date", "activity_count"])
            frame = frame.tail(30)
            history = [
                {"date": str(row.date), "rides": int(row.activity_count)}
                for row in frame.itertuples(index=False)
            ]
        except (OSError, ValueError, pd.errors.ParserError):
            history = []

    weather: dict | None = None
    weather_path = CACHE_DIR / "weather_chicago.json"
    if weather_path.exists():
        try:
            weather_frame = pd.read_json(weather_path)
            if not weather_frame.empty:
                local_today = pd.Timestamp.now(tz="America/Chicago").tz_localize(None).normalize()
                tomorrow = local_today + pd.Timedelta(days=1)
                row = weather_frame.loc[pd.to_datetime(weather_frame["date"]).dt.normalize() == tomorrow]
                if not row.empty:
                    weather = row.iloc[0].where(pd.notna(row.iloc[0]), None).to_dict()
                    weather["date"] = tomorrow.strftime("%Y-%m-%d")
        except (OSError, ValueError, KeyError):
            weather = None

    result_status = status.get("status")
    if _run_state["state"] == "running":
        message = "The daily forecast pipeline is running. This can take a few minutes."
        state = "running"
    elif _run_state["state"] == "error":
        message = _run_state["message"]
        state = "error"
    elif forecast and forecast.get("forecast_kind") == "stale_count_estimate":
        message = (
            f"Estimated activity using data through {forecast['activity_data_through']}; "
            f"the {forecast['forecast_horizon_days']}-day forecast extends beyond the latest published data."
        )
        state = "estimated"
    elif result_status == "unavailable":
        message = status.get("message", "A forecast is currently unavailable.")
        state = "unavailable"
    elif forecast and forecast.get("target_date"):
        message = None
        state = "ready"
    elif metrics:
        message = "Historical evaluation is ready. Run the daily pipeline to check whether the latest activity data supports a current forecast."
        state = "evaluation_only"
    else:
        message = "Run the daily pipeline to prepare the first forecast and evaluation."
        state = "not_run"

    if state not in {"ready", "estimated"}:
        forecast = None
    if state not in {"ready", "estimated"}:
        nudge = None

    return {
        "city": "Chicago",
        "state": state,
        "message": message,
        "updated_at": _run_state["finished_at"],
        "activity_through": history[-1]["date"] if history else None,
        "activity_freshness": build_activity_freshness(history[-1]["date"] if history else None),
        "history": history,
        "metrics": metrics,
        "forecast": forecast,
        "nudge": nudge,
        "weather": weather,
        "weather_suggestion": build_weather_suggestion(weather),
        "job": dict(_run_state),
    }


def _run_pipeline() -> None:
    try:
        completed = subprocess.run(
            [sys.executable, "run_daily.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            status = _read_json(RESULTS_DIR / "status.json") or {}
            _run_state["state"] = "finished"
            _run_state["message"] = status.get("message")
        else:
            details = (completed.stderr or completed.stdout).strip()
            _run_state["state"] = "error"
            _run_state["message"] = details[-1800:] or "The forecast pipeline failed without diagnostic output."
    except OSError as exc:
        _run_state["state"] = "error"
        _run_state["message"] = f"Could not start the Python pipeline: {exc}"
    finally:
        _run_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        _run_lock.release()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return _dashboard_payload()


@app.get("/api/audio")
def audio(path: str) -> FileResponse:
    audio_path = Path(path).resolve()
    nudge_dir = (BASE_DIR / "output" / "nudges").resolve()
    if not audio_path.is_relative_to(nudge_dir) or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio clip not found.")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.post("/api/run")
def run_daily() -> dict:
    if not _run_lock.acquire(blocking=False):
        return {"accepted": False, "state": "running", "message": "The daily pipeline is already running."}
    _run_state.update(
        state="running",
        message=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
    )
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return {"accepted": True, "state": "running"}


frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
