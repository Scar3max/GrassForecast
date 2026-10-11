# Grass Forecast

Grass Forecast is a tiny outdoor-day forecaster built for the Touch Grass challenge. It learns from public outdoor activity data and historical weather to estimate whether tomorrow will be a high-activity day, which window is likely busiest, and then writes a short spoken nudge.

## What this does

- Pulls Chicago Divvy bike-share activity data from the public Divvy bucket
- Fetches historical daily weather for the training period and tomorrow's forecast from Open-Meteo for Chicago
- Builds a daily feature table with leakage-safe history and weather features
- Evaluates with expanding, time-ordered walk-forward folds and compares against three baselines
- Uses TabPFN when its token is configured and available; otherwise labels the scikit-learn fallback explicitly
- Generates a short, conversational nudge with Gemma through the Gemini API or a local fallback message, including a weather-aware estimate-based nudge when activity data is stale
- Converts fresh-data and stale-estimate nudges to short audio files with ElevenLabs when a key is configured
- Serves forecast data through FastAPI and displays it in a responsive React dashboard
- Shows a rules-based outing idea from tomorrow's weather even when Divvy activity is stale; this is explicitly not an activity prediction

## Fresh-clone setup

Requires Python 3.11+ and Node.js 22.12+ with npm.

```powershell
cd grass-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
copy .env.example .env
cd frontend
npm install
cd ..
```

Start the local API in the first terminal:

```powershell
python -m uvicorn api:app --reload
```

Start the React dashboard in a second terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). Use **Refresh outlook** to run the daily pipeline and update the dashboard. To run the Python job directly, use `python run_daily.py` from the project root.

To build the production frontend bundle, run `npm run build` from `frontend/`.
With that bundle built, `python -m uvicorn api:app` serves the dashboard and API together at `http://127.0.0.1:8000`.

## Required secrets

Create a `.env` file with values such as:

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemma-3-4b-it
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=SAz9YHcvj6GT2YYXdXww
TABPFN_TOKEN=
CITY=Chicago
LATITUDE=41.8781
LONGITUDE=-87.6298
```

## What leaves the machine

This project is intended to be honest about hosted calls. The current default behavior is:

- Open-Meteo: city-level weather coordinates for Chicago, no personal data, just weather history and forecast
- Divvy data: public bike-share trip data, aggregated by date and hour for model training; no personal rider identities are used in the pipeline
- Prior Labs TabPFN: this project uses the Python package for local inference; numeric feature rows are not sent to Prior Labs by this app. Initial model access may require authentication and downloading model weights
- Google AI Studio / Gemini: the prompt includes only a short forecast summary; it does not include dates, coordinates, or place names
- ElevenLabs: the nudge text only, not the full forecast or raw activity data. Audio generation requires both `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`; failures are recorded in `results/nudge.json` and shown in the dashboard instead of being silently ignored.

The app uses cached weather if a complete matching cache exists and falls back to a text-only nudge. It does not fill gaps with fabricated observations: when Divvy data is stale, it distinguishes a model estimate from measured activity and withholds the estimate if the source data is incomplete or outside the supported lead time.

Divvy publishes trips in monthly files, so its newest activity data can lag the current date. When activity is available through the day before the forecast target, the app uses its high-activity classifier and marks the data as fresh for that outlook. If it is behind, the app instead produces a direct multi-horizon ride-count estimate from historical origin/target examples, up to 30 days beyond the latest observed activity. It shows the activity-through date, days behind, forecast lead, and the estimated ride count. Beyond that 30-day limit the activity estimate is withheld.

The stale-data estimate is a separate model from the fresh-data high-activity classifier. Its displayed range is the 10th–90th percentile of Random Forest tree predictions, not a calibrated prediction interval. Historical training uses archived observed weather for target dates rather than archived day-ahead weather forecasts, so the estimate is experimental and may be optimistic; treat it as directional, not measured activity.

When an activity estimate is outside the supported horizon or unavailable, the dashboard can still show a weather-only outing idea from tomorrow's daily temperature, precipitation, and wind forecast. That idea does not estimate bike activity or select a specific hour; check the hourly forecast before choosing when to go.

Walk-forward evaluation uses Open-Meteo archived observed weather for historical target days, not archived day-ahead forecast snapshots. This means weather-feature scores are preliminary and may be optimistic; do not present them as a fully validated forecast until evaluation uses weather forecasts as they were available at each historical prediction time.

## Dataset and licensing

- Dataset: Chicago Divvy bike-share trip data
- Source: https://divvybikes.com/system-data
- License: Divvy Data License Agreement, as published on the Divvy site
- Range: recent real public trip data from the public Divvy S3 bucket, cached locally; availability and freshness depend on the source's published files
- Interpretation: outdoor activity is the number of trips per day and per hour, with the prediction target defined as a high-activity day and the busiest window of the day
- The historical Open-Meteo response currently has no UV-index values for this source/date range, so UV is omitted from the model features rather than filled with invented values


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
  api.py
  frontend/
  results/
  output/
  tests/
```

## Definition of done

- A fresh clone can install dependencies and run with the README steps
- The app runs with complete cached source data if a provider is unavailable; it does not silently substitute synthetic data in the real forecast pipeline
- The no-leakage check and anonymization checks are included in the test suite
- A short nudge can be generated and, when configured, voiced by ElevenLabs
- The README spells out what leaves the machine and what is kept local

## Project status

This is a working prototype scaffold for a rapid hackathon build. The architecture is designed to be honest about data flow, fail gracefully when a hosted API is down, and keep the code easy to swap between providers.
