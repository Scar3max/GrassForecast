import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  AudioLines,
  Bike,
  CalendarDays,
  Check,
  ChevronDown,
  CircleHelp,
  CloudSun,
  Compass,
  Leaf,
  LoaderCircle,
  MapPin,
  RefreshCw,
  Sprout,
  Sun,
  Wind,
} from "lucide-react";
import type { DashboardData, Score } from "./types";

const API = import.meta.env.VITE_API_URL ?? "";

const labels: Record<string, string> = {
  same_as_yesterday: "Same as yesterday",
  same_weekday_last_week: "Same weekday last week",
  majority_class: "Majority-class guess",
};

function formatDate(date: string | null | undefined, options?: Intl.DateTimeFormatOptions) {
  if (!date) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    ...(options ?? {}),
  }).format(new Date(`${date.slice(0, 10)}T12:00:00`));
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function scorePercent(score: Score | undefined, key: keyof Score) {
  return score ? `${(score[key] * 100).toFixed(1)}%` : "—";
}

function ActivityChart({ history }: { history: DashboardData["history"] }) {
  const chart = useMemo(() => {
    if (!history.length) return null;
    const width = 760;
    const height = 230;
    const padding = { top: 16, right: 14, bottom: 30, left: 52 };
    const values = history.map((point) => point.rides);
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const range = Math.max(max - min, 1);
    const points = history.map((item, index) => ({
      ...item,
      x: padding.left + (index / Math.max(history.length - 1, 1)) * (width - padding.left - padding.right),
      y: padding.top + ((max - item.rides) / range) * (height - padding.top - padding.bottom),
    }));
    const line = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    const bottom = height - padding.bottom;
    const area = `${line} L ${points.at(-1)?.x ?? padding.left} ${bottom} L ${points[0].x} ${bottom} Z`;
    return { width, height, points, line, area, bottom, max };
  }, [history]);

  if (!chart) {
    return <div className="chart-empty">Activity history will appear here after the first successful data run.</div>;
  }

  const tickCount = 4;
  return (
    <div className="chart-wrap" role="img" aria-label={`Daily bike trips for the latest ${history.length} available days`}>
      <svg viewBox={`0 0 ${chart.width} ${chart.height}`} className="activity-chart" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chart-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#a6c879" stopOpacity=".34" />
            <stop offset="100%" stopColor="#a6c879" stopOpacity="0" />
          </linearGradient>
        </defs>
        {Array.from({ length: tickCount }, (_, index) => {
          const y = 18 + (index / (tickCount - 1)) * (chart.bottom - 18);
          const value = chart.max - (index / (tickCount - 1)) * chart.max;
          return (
            <g key={index}>
              <line x1="52" x2={chart.width - 14} y1={y} y2={y} className="chart-grid" />
              <text x="43" y={y + 4} textAnchor="end" className="chart-label">
                {formatNumber(value / 1000)}k
              </text>
            </g>
          );
        })}
        <path d={chart.area} fill="url(#chart-fill)" />
        <path d={chart.line} className="chart-line" />
        {chart.points.map((point, index) => (
          <circle
            key={`${point.date}-${index}`}
            cx={point.x}
            cy={point.y}
            r={index === chart.points.length - 1 ? 4 : 2.5}
            className={index === chart.points.length - 1 ? "chart-dot chart-dot-current" : "chart-dot"}
          >
            <title>{`${formatDate(point.date, { year: "numeric" })}: ${formatNumber(point.rides)} rides`}</title>
          </circle>
        ))}
        <text x="52" y={chart.height - 7} className="chart-label">{formatDate(history[0].date)}</text>
        <text x={chart.width - 14} y={chart.height - 7} textAnchor="end" className="chart-label">
          {formatDate(history.at(-1)?.date)}
        </text>
      </svg>
    </div>
  );
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [showPrivacy, setShowPrivacy] = useState(false);

  const loadDashboard = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/dashboard`);
      if (!response.ok) throw new Error(`Dashboard service responded with ${response.status}.`);
      const data = (await response.json()) as DashboardData;
      setDashboard(data);
      setRequestError(null);
      setRefreshing(data.state === "running");
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "Could not connect to the local dashboard service.");
      setRefreshing(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  useEffect(() => {
    if (!refreshing) return;
    const timer = window.setInterval(() => void loadDashboard(), 3500);
    return () => window.clearInterval(timer);
  }, [refreshing, loadDashboard]);

  const startRun = async () => {
    setRefreshing(true);
    setRequestError(null);
    try {
      const response = await fetch(`${API}/api/run`, { method: "POST" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Could not start the daily run.");
      await loadDashboard();
    } catch (error) {
      setRefreshing(false);
      setRequestError(error instanceof Error ? error.message : "Could not start the daily pipeline.");
    }
  };

  const metrics = dashboard?.metrics;
  const peakBaseline = useMemo(() => {
    if (!metrics) return null;
    return Object.entries(metrics.baselines).sort((a, b) => b[1].balanced_accuracy - a[1].balanced_accuracy)[0];
  }, [metrics]);

  const liveForecast = dashboard?.state === "ready" && dashboard.forecast?.forecast_kind === "high_activity_classifier"
    ? dashboard.forecast
    : null;
  const staleEstimate = dashboard?.state === "estimated" && dashboard.forecast?.forecast_kind === "stale_count_estimate"
    ? dashboard.forecast
    : null;
  const weatherSuggestion = dashboard?.weather_suggestion;
  const noticeCopy = dashboard?.message ?? "Historical results are shown below; we only show a forecast when the activity data is fresh enough to trust.";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Grass Forecast home">
          <span className="brand-mark"><Sprout size={23} strokeWidth={2.2} /></span>
          <span className="brand-type">grass<span>forecast</span></span>
        </a>

        <div className="side-section-label">YOUR OUTLOOK</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-item nav-item-active" href="#overview"><Compass size={17} /> Overview</a>
          <a className="nav-item" href="#activity"><Bike size={17} /> City activity</a>
          <a className="nav-item" href="#accuracy"><Check size={17} /> Model check</a>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <div className="sidebar-note-icon"><Leaf size={17} /></div>
            <p>A little fresh air can fit into any day.</p>
            <span>No streaks. No pressure.</span>
          </div>
          <button className="privacy-link" onClick={() => setShowPrivacy((value) => !value)}>
            <CircleHelp size={16} /> About data & privacy <ChevronDown size={14} className={showPrivacy ? "rotate" : ""} />
          </button>
          {showPrivacy && (
            <div className="privacy-details">
              Weather requests send Chicago coordinates to Open-Meteo. Forecast summaries go to Gemini only when configured. Voice text goes to ElevenLabs only when enabled.
            </div>
          )}
          <div className="sidebar-foot"><span className="status-pip" /> Local demo · Chicago, IL</div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb">Your city <span>/</span> Daily outlook</div>
          <div className="topbar-right">
            <span className="today-label"><CalendarDays size={15} /> {new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" }).format(new Date())}</span>
            <span className="city-pill"><MapPin size={14} /> Chicago</span>
          </div>
        </header>

        <section className="intro-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> A LITTLE MORE OUT THERE</div>
            <h1>Make room for <em>outside.</em></h1>
            <p className="intro-copy">Fresh-data outlooks and clearly labeled estimates when Divvy data lags.</p>
          </div>
          <button className="refresh-button" onClick={startRun} disabled={refreshing || loading}>
            {refreshing ? <LoaderCircle size={16} className="spin" /> : <RefreshCw size={16} />}
            {refreshing ? "Updating…" : "Refresh outlook"}
          </button>
        </section>

        {requestError && (
          <section className="notice notice-error" role="alert">
            <div className="notice-icon"><CircleHelp size={18} /></div>
            <div><strong>Dashboard service not connected</strong><p>{requestError} Start the local API and frontend using the README steps.</p></div>
          </section>
        )}

        <section className={liveForecast ? "forecast-hero forecast-hero-ready" : "forecast-hero"} aria-live="polite">
          <div className="hero-left">
            <div className="hero-kicker"><Sun size={15} /> {liveForecast ? "TOMORROW’S OUTLOOK" : staleEstimate ? "TOMORROW’S ACTIVITY ESTIMATE" : "TOMORROW’S WEATHER IDEA"}</div>
            {loading ? (
              <div className="hero-loading"><LoaderCircle size={21} className="spin" /> Gathering the latest city signals…</div>
            ) : liveForecast ? (
              <>
                <h2>{liveForecast.high_activity_day ? "A promising day to get out." : "A quieter day on the paths."}</h2>
                <p className="hero-description">The city tends to pick up in the <strong>{liveForecast.peak_window}</strong>. Take it at your own pace.</p>
                <div className="freshness-row"><CalendarDays size={15} /> Activity data is fresh for tomorrow’s outlook through <strong>{formatDate(liveForecast.activity_data_through, { year: "numeric" })}</strong>.</div>
                <div className="hero-facts">
                  <div className="hero-fact"><span>Activity outlook</span><strong>{Math.round(liveForecast.high_activity_probability * 100)}% <small>chance of a busy day</small></strong></div>
                  <div className="hero-fact"><span>Likely peak</span><strong>{liveForecast.peak_window}</strong></div>
                  {dashboard?.weather && <div className="hero-fact"><span>High temperature</span><strong>{dashboard.weather.temperature_2m_max ?? "—"}° <small>Chicago</small></strong></div>}
                </div>
              </>
            ) : staleEstimate ? (
              <>
                <div className="availability-tag"><span /> Estimate · latest activity data is stale</div>
                <h2>{formatNumber(staleEstimate.estimated_activity_count ?? 0)} estimated Divvy rides</h2>
                <p className="hero-description">A model estimate for {formatDate(staleEstimate.target_date)}—not observed trips.</p>
                <div className="hero-facts">
                  <div className="hero-fact"><span>Model spread</span><strong>{formatNumber(staleEstimate.model_range_low ?? 0)}–{formatNumber(staleEstimate.model_range_high ?? 0)} <small>rides</small></strong></div>
                  <div className="hero-fact"><span>Forecast lead</span><strong>{staleEstimate.forecast_horizon_days} <small>days beyond data</small></strong></div>
                </div>
                <p className="weather-caveat">{staleEstimate.range_method}. Activity data is through {formatDate(staleEstimate.activity_data_through, { year: "numeric" })}, {staleEstimate.activity_staleness_days} {staleEstimate.activity_staleness_days === 1 ? "day" : "days"} behind.</p>
              </>
            ) : weatherSuggestion ? (
              <>
                <div className="availability-tag"><span /> Weather-based · not an activity prediction</div>
                <h2>{weatherSuggestion.title}</h2>
                <p className="hero-description">{weatherSuggestion.suggestion}</p>
                <p className="weather-caveat">{weatherSuggestion.basis}</p>
                <div className="freshness-row"><CalendarDays size={15} />
                  {dashboard?.activity_through
                    ? <>No activity estimate available · data through <strong>{formatDate(dashboard.activity_through, { year: "numeric" })}</strong>{dashboard.activity_freshness.days_behind ? ` · ${dashboard.activity_freshness.days_behind} days behind` : ""}</>
                    : "City activity prediction is currently unavailable."}
                </div>
              </>
            ) : (
              <>
                <div className="availability-tag"><span /> Forecast on pause</div>
                <h2>We’re waiting on a fresh city signal.</h2>
                <p className="hero-description">{noticeCopy}</p>
                {dashboard?.activity_through && (
                  <div className="freshness-row"><CalendarDays size={15} /> Latest activity data: <strong>{formatDate(dashboard.activity_through, { year: "numeric" })}</strong></div>
                )}
              </>
            )}
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="sun-disc" />
            <div className="sun-glow" />
            <div className="hill hill-back" />
            <div className="hill hill-front" />
            <div className="art-path" />
            <div className="art-leaf leaf-one"><Leaf size={25} /></div>
            <div className="art-leaf leaf-two"><Leaf size={17} /></div>
            <div className="art-spark spark-one">✳</div>
            <div className="art-spark spark-two">✳</div>
          </div>
        </section>

        {dashboard?.state === "running" && (
          <div className="running-strip"><LoaderCircle size={15} className="spin" /> Refreshing activity and weather, then updating the evaluation. This may take a few minutes.</div>
        )}

        <section className="quick-stats" aria-label="Forecast details">
          <article className="stat-card">
            <div className="stat-icon stat-icon-green"><Bike size={18} /></div>
            <div><span className="stat-label">CITY ACTIVITY</span><strong>{dashboard?.activity_through ? formatDate(dashboard.activity_through, { year: "numeric" }) : "Not loaded"}</strong><small>{dashboard?.activity_freshness.status === "fresh" ? "Fresh for tomorrow’s outlook" : dashboard?.activity_freshness.status === "stale" ? `Stale · ${dashboard.activity_freshness.days_behind} days behind` : "Latest available trip day"}</small></div>
            <ArrowUpRight className="stat-trend" size={16} />
          </article>
          <article className="stat-card">
            <div className="stat-icon stat-icon-sun"><CloudSun size={18} /></div>
            <div><span className="stat-label">WEATHER CHECK</span><strong>{dashboard?.weather?.temperature_2m_max != null ? `${dashboard.weather.temperature_2m_max}° / ${dashboard.weather.temperature_2m_min ?? "—"}°` : "Forecast pending"}</strong><small>{dashboard?.weather?.date ? formatDate(dashboard.weather.date, { year: "numeric" }) : "Open-Meteo"}</small></div>
            <Wind className="stat-trend" size={16} />
          </article>
          <article className="stat-card">
            <div className="stat-icon stat-icon-lavender"><Check size={18} /></div>
            <div><span className="stat-label">MODEL REALITY CHECK</span><strong>{peakBaseline ? `${(peakBaseline[1].balanced_accuracy * 100).toFixed(0)}% baseline` : "In progress"}</strong><small>{peakBaseline ? labels[peakBaseline[0]] : "Walk-forward comparison"}</small></div>
            {metrics?.tabpfn && peakBaseline && (metrics.tabpfn.balanced_accuracy > peakBaseline[1].balanced_accuracy
              ? <ArrowUpRight className="stat-trend stat-trend-up" size={16} />
              : <ArrowDownRight className="stat-trend" size={16} />)}
          </article>
        </section>

        <section className="content-grid">
          <article className="panel activity-panel" id="activity">
            <div className="panel-heading">
              <div>
                <div className="section-eyebrow">THE CITY, LATELY</div>
                <h3>Bike activity</h3>
              </div>
              <span className="period-chip"><span className="legend-dot" /> Daily rides</span>
            </div>
            <p className="panel-copy">A look at recent daily trips in the public Divvy dataset.</p>
            <ActivityChart history={dashboard?.history ?? []} />
            <div className="chart-footer">
              <span><Bike size={14} /> Divvy public trip counts</span>
              <span>{dashboard?.history.length ?? 0} days shown</span>
            </div>
          </article>

          <article className="panel nudge-panel">
            <div className="panel-heading">
              <div>
                <div className="section-eyebrow">{liveForecast || staleEstimate ? "A SMALL NUDGE" : "WEATHER-ONLY CONTEXT"}</div>
                <h3>{liveForecast || staleEstimate ? "Your outside note" : "What this idea uses"}</h3>
              </div>
              <div className="nudge-icon">{liveForecast || staleEstimate ? <AudioLines size={18} /> : <CloudSun size={18} />}</div>
            </div>
            {(liveForecast || staleEstimate) && dashboard?.nudge ? (
              <>
                <div className="quote-mark">“</div>
                <p className="nudge-text">{dashboard.nudge.text}</p>
                {dashboard.nudge.audio_path && (
                  <audio controls className="audio-player" src={`${API}/api/audio?path=${encodeURIComponent(dashboard.nudge.audio_path)}`}>
                    Your browser does not support audio playback.
                  </audio>
                )}
                {!dashboard.nudge.audio_path && dashboard.nudge.audio_error && (
                  <p className="weather-caveat" role="status">Audio unavailable: {dashboard.nudge.audio_error}</p>
                )}
                <div className="nudge-signoff"><span className="mini-leaf"><Leaf size={14} /></span>
                  {staleEstimate
                    ? <>Estimate-based note · data through {formatDate(staleEstimate.activity_data_through, { year: "numeric" })}</>
                    : liveForecast
                      ? <>Generated for {formatDate(liveForecast.target_date)}</>
                      : null}
                </div>
                {staleEstimate && <p className="weather-caveat">Ride count is an estimate, not observed activity; the shown model spread is not a calibrated confidence interval.</p>}
              </>
            ) : (
              <div className="nudge-empty">
                <div className="nudge-empty-art">{weatherSuggestion ? <CloudSun size={24} /> : <Sprout size={24} />}</div>
                <p>{weatherSuggestion
                  ? "This idea uses tomorrow’s daily temperature, precipitation, and wind forecast."
                  : "Your nudge will show up when there’s enough fresh activity data for a reliable forecast."}</p>
                <span>{weatherSuggestion
                  ? "It does not use bike-share activity. Check hourly conditions before choosing when to go."
                  : "No pressure—this dashboard won’t guess when it doesn’t know."}</span>
              </div>
            )}
          </article>
        </section>

        <section className="panel model-panel" id="accuracy">
          <div className="panel-heading model-heading">
            <div>
              <div className="section-eyebrow">HONEST BY DESIGN</div>
              <h3>How the model holds up</h3>
              <p className="panel-copy">Compared on the same time-ordered test days. Higher balanced accuracy is better.</p>
            </div>
            {metrics && <span className="sample-chip">{metrics.evaluation.out_of_sample_days} test days</span>}
          </div>
          {metrics ? (
            <>
              <div className="model-table-wrap">
                <table className="model-table">
                  <thead><tr><th>Approach</th><th>Accuracy</th><th>Balanced</th><th>F1</th><th>vs. strongest baseline</th></tr></thead>
                  <tbody>
                    {[{ name: metrics.model_used, score: metrics.tabpfn, isModel: true }, ...Object.entries(metrics.baselines).map(([name, score]) => ({ name: labels[name] ?? name, score, isModel: false }))].map((row) => {
                      const delta = row.score.balanced_accuracy - (peakBaseline?.[1].balanced_accuracy ?? row.score.balanced_accuracy);
                      return (
                        <tr key={row.name} className={row.isModel ? "model-row" : ""}>
                          <td><span className={row.isModel ? "model-name-dot" : "baseline-name-dot"} />{row.name}{row.isModel && <span className="model-badge">MODEL</span>}</td>
                          <td>{scorePercent(row.score, "accuracy")}</td>
                          <td><strong>{scorePercent(row.score, "balanced_accuracy")}</strong></td>
                          <td>{scorePercent(row.score, "f1")}</td>
                          <td className={row.isModel && delta < 0 ? "delta-muted" : "delta-positive"}>{row.isModel ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts` : row.score.balanced_accuracy === peakBaseline?.[1].balanced_accuracy ? "Best simple baseline" : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="evaluation-footnote">
                <CircleHelp size={15} />
                <span>Walk-forward evaluation · {metrics.evaluation.labeled_days} labeled days · Weather uses archived observations, not historical day-ahead forecasts; scores are preliminary.</span>
              </div>
            </>
          ) : (
            <div className="chart-empty">Evaluation scores will appear after the first successful pipeline run.</div>
          )}
        </section>

        <footer className="page-footer">
          <div><span className="footer-mark"><Sprout size={16} /></span> A little more outside, one forecast at a time.</div>
          <span>Open data · Clear limits · No guilt</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
