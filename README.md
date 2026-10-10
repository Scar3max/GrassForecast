# Grass Forecast

Grass Forecast is a tiny outdoor-day forecaster built for the Touch Grass challenge. It learns from public outdoor activity data and historical weather to estimate whether tomorrow will be a high-activity day, which window is likely busiest, and then writes a short spoken nudge.

## What this does

- Pulls Chicago Divvy bike-share activity data from the public Divvy bucket
- Fetches daily weather from Open-Meteo for Chicago
- Builds a daily feature table with leakage-safe history and weather features
- Trains a TabPFN-style classifier when available and falls back gracefully to a scikit-learn classifier if the TabPFN token or model download is unavailable
- Generates a short nudge with Gemma through the Gemini API or a local fallback message
- Converts the nudge to a short audio file with ElevenLabs when a key is configured
- Shows the forecast and metrics in a tiny Streamlit dashboard

## Fresh-clone setup

```powershell
cd grass-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
copy .env.example .env
python run_daily.py
python -m streamlit run src/app.py
```

## Required secrets

Create a `.env` file with values such as:

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemma-3-4b-it
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaLk
TABPFN_TOKEN=
CITY=Chicago
LATITUDE=41.8781
LONGITUDE=-87.6298
```

## What leaves the machine

This project is intended to be honest about hosted calls. The current default behavior is:

- Open-Meteo: city-level weather coordinates for Chicago, no personal data, just weather history and forecast
- Divvy data: public bike-share trip data, aggregated by date and hour for model training; no personal rider identities are used in the pipeline
- Prior Labs TabPFN: only anonymized numeric rows are sent if the hosted TabPFN path is used; no dates, coordinates, or place names are included in the payload sent to the model
- Google AI Studio / Gemini: the prompt includes only a short anonymized forecast summary; no personal and no location details beyond the city name in the prompt
- ElevenLabs: the nudge text only, not the full forecast or raw activity data

The app fails gracefully if a hosted provider is unavailable and falls back to cached data or a text-only nudge.

## Dataset and licensing

- Dataset: Chicago Divvy bike-share trip data
- Source: https://divvybikes.com/system-data
- License: Divvy Data License Agreement, as published on the Divvy site
- Range: real public trip data, from the public Divvy S3 bucket, with the loader configured to pull recent months and cache them locally
- Interpretation: outdoor activity is the number of trips per day and per hour, with the prediction target defined as a high-activity day and the busiest window of the day


## Repo layout

```text
grass-forecast/
  README.md
  LICENSE
  .env.example
  config.example.toml
  .github/workflows/tests.yml
  data/
  src/
  results/
  output/
  tests/
```

## Definition of done

- A fresh clone can install dependencies and run with the README steps
- The app runs with cached or synthetic fallback data if a provider is unavailable
- The no-leakage check and anonymization checks are included in the test suite
- A short nudge can be generated and, when configured, voiced by ElevenLabs
- The README spells out what leaves the machine and what is kept local

## Project status

This is a working prototype scaffold for a rapid hackathon build. The architecture is designed to be honest about data flow, fail gracefully when a hosted API is down, and keep the code easy to swap between providers.
