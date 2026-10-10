from src.dataset import load_divvy_dataset
from src.weather import fetch_weather_snapshot
from src.features import build_feature_table
from src.evaluate import evaluate_models
from src.forecast_tabpfn import forecast_tomorrow
from src.nudge_gemma import generate_nudge
from src.voice_elevenlabs import synthesize_nudge


def main() -> None:
    dataset = load_divvy_dataset()
    weather_df = fetch_weather_snapshot(city="Chicago", latitude=41.8781, longitude=-87.6298)
    feature_df = build_feature_table(dataset, weather_df)
    metrics = evaluate_models(feature_df)
    forecast = forecast_tomorrow(feature_df)
    nudge = generate_nudge(forecast, weather_df)
    clip_path = synthesize_nudge(nudge)
    print("Prediction summary:")
    print(f"- high_activity_day: {forecast['high_activity_day']} ({forecast['high_activity_probability']:.2%})")
    print(f"- peak_window: {forecast['peak_window']}")
    print(f"- nudge: {nudge}")
    print(f"- clip: {clip_path}")
    print("- metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()
