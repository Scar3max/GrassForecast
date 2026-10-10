from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR


st.set_page_config(page_title="Grass Forecast", layout="wide")
st.title("Grass Forecast")
st.caption("Outdoor-day forecast for Chicago using public activity data + weather + a tiny TabPFN-style forecaster.")

results_path = RESULTS_DIR / "metrics.json"
forecast_path = RESULTS_DIR / "tomorrow_prediction.json"

if results_path.exists():
    metrics = json.loads(results_path.read_text())
    st.subheader("Metrics")
    st.json(metrics)
else:
    st.info("Run `python run_daily.py` first to generate the forecast and metrics.")

if forecast_path.exists():
    forecast = json.loads(forecast_path.read_text())
    col1, col2, col3 = st.columns(3)
    col1.metric("High-activity chance", f"{forecast['high_activity_probability'] * 100:.1f}%")
    col2.metric("Peak window", forecast["peak_window"])
    col3.metric("High-activity day", "Yes" if forecast["high_activity_day"] else "No")
else:
    st.warning("No prediction found yet.")

nudge_path = Path("config/nudge_system_prompt.txt")
if nudge_path.exists():
    st.subheader("Nudge prompt")
    st.code(nudge_path.read_text())

mp3_path = list(Path("output/nudges").glob("*.mp3"))
if mp3_path:
    st.subheader("Voice nudge")
    st.audio(str(mp3_path[0]))
