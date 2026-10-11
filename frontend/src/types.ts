export type Score = {
  accuracy: number;
  balanced_accuracy: number;
  f1: number;
};

export type Metrics = {
  evaluation: {
    method: string;
    out_of_sample_days: number;
    labeled_days: number;
    class_counts: Record<string, number>;
  };
  model_used: string;
  tabpfn: Score;
  baselines: Record<string, Score>;
};

type ForecastBase = {
  target_date: string;
  activity_data_through: string;
  activity_staleness_days: number;
  forecast_horizon_days: number;
  model_used: string;
};

export type Forecast =
  | (ForecastBase & {
      forecast_kind: "high_activity_classifier";
      activity_freshness: "fresh";
      high_activity_day: number;
      high_activity_probability: number;
      peak_window: string;
    })
  | (ForecastBase & {
      forecast_kind: "stale_count_estimate";
      activity_freshness: "stale";
      estimated_activity_count: number;
      model_range_low: number;
      model_range_high: number;
      range_method: string;
    });

export type WeatherSuggestion = {
  date: string;
  title: string;
  suggestion: string;
  basis: string;
};

export type DashboardData = {
  city: string;
  state: "ready" | "estimated" | "unavailable" | "evaluation_only" | "not_run" | "running" | "error";
  message: string | null;
  updated_at: string | null;
  activity_through: string | null;
  activity_freshness: {
    status: "fresh" | "stale" | "unavailable";
    days_behind: number | null;
  };
  history: Array<{ date: string; rides: number }>;
  metrics: Metrics | null;
  forecast: Forecast | null;
  nudge: { text: string; audio_path: string | null; audio_error?: string | null } | null;
  weather: {
    date: string;
    temperature_2m_max: number | null;
    temperature_2m_min: number | null;
    precipitation_sum: number | null;
    precipitation_hours: number | null;
    wind_speed_10m_max: number | null;
  } | null;
  weather_suggestion: WeatherSuggestion | null;
  job: {
    state: string;
    message: string | null;
    started_at: string | null;
    finished_at: string | null;
  };
};
