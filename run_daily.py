import json

from src.config import RESULTS_DIR
from src.dataset import load_divvy_dataset
from src.weather import fetch_weather_snapshot
from src.features import build_feature_table
from src.evaluate import evaluate_models
from src.forecast_tabpfn import forecast_tomorrow
from src.nudge_gemma import generate_nudge
from src.voice_elevenlabs import synthesize_nudge


def _print_prediction_summary(
    forecast: dict,
    nudge: str | None = None,
    clip_path: str | None = None,
) -> None:
    print("Prediction summary:")
    if forecast["forecast_kind"] == "stale_count_estimate":
        print(f"- estimated_activity_count: {forecast['estimated_activity_count']:,} (model estimate, not observed)")
        print(
            f"- model range: {forecast['model_range_low']:,}-{forecast['model_range_high']:,} rides "
            "(tree spread, not a calibrated prediction interval)"
        )
        print(f"- target date: {forecast['target_date']}")
        print(
            f"- activity data through: {forecast['activity_data_through']} "
            f"({forecast['activity_staleness_days']} days behind; "
            f"{forecast['forecast_horizon_days']}-day forecast lead)"
        )
    else:
        print(
            f"- high_activity_day: {forecast['high_activity_day']} "
            f"({forecast['high_activity_probability']:.2%})"
        )
        print(f"- peak_window: {forecast['peak_window']}")
        if nudge is not None:
            print(f"- nudge: {nudge}")
        if clip_path is not None:
            print(f"- clip: {clip_path}")
    print(f"- model: {forecast['model_used']}")
    print("- metrics saved to results/metrics.json")


def main() -> None:
    for stale_output in (
        RESULTS_DIR / "metrics.json",
        RESULTS_DIR / "tomorrow_prediction.json",
        RESULTS_DIR / "nudge.json",
        RESULTS_DIR / "status.json",
    ):
        stale_output.unlink(missing_ok=True)
    dataset = load_divvy_dataset()
    weather_df = fetch_weather_snapshot(
        city="Chicago",
        latitude=41.8781,
        longitude=-87.6298,
        start_date=dataset["date"].min(),
        end_date=dataset["date"].max(),
    )
    feature_df = build_feature_table(dataset, weather_df)
    try:
        evaluate_models(feature_df)
        forecast = forecast_tomorrow(feature_df, weather_df)
    except ValueError as exc:
        status = {"status": "unavailable", "message": str(exc)}
        (RESULTS_DIR / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Forecast unavailable: {exc}")
        return
    nudge = None
    clip_path = None
    audio_error = None
    if forecast["forecast_kind"] in {"high_activity_classifier", "stale_count_estimate"}:
        nudge = generate_nudge(forecast, weather_df)
        try:
            clip_path = synthesize_nudge(nudge)
        except RuntimeError as exc:
            audio_error = str(exc)
            print(f"Audio generation failed: {audio_error}")
        nudge_payload = {"text": nudge, "audio_path": clip_path}
        if audio_error:
            nudge_payload["audio_error"] = audio_error
        (RESULTS_DIR / "nudge.json").write_text(
            json.dumps(nudge_payload, indent=2),
            encoding="utf-8",
        )
    (RESULTS_DIR / "status.json").write_text(
        json.dumps({"status": "ok"}, indent=2),
        encoding="utf-8",
    )
    _print_prediction_summary(forecast, nudge, clip_path)


if __name__ == "__main__":
    main()
