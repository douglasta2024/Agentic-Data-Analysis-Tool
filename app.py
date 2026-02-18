"""
Agentic Data Analysis Tool — Streamlit Web App

Two-perspective data analysis:
- Data Scientist (📊): statistical depth, quality checks, correlations, time series diagnostics
- Business Analyst (💼): visual-first, plain-English KPIs, actionable takeaways
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from agents import AnalysisResult, GeminiClient, OrchestratorAgent

# ── Config ──────────────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS: Dict[str, str] = {
    "positive": "#1B9E77",
    "negative": "#D95F02",
    "neutral": "#7570B3",
    "accent": "#E7298A",
}
CHART_CONFIG = {"displaylogo": False}

# Keys reset when a new file is uploaded (api_key is intentionally excluded)
_RESET_ON_UPLOAD = [
    "analysis_run", "results", "synthesis", "gemini_ds", "gemini_ba",
    "dataset_type", "date_column", "pipeline_log", "agent_errors",
    "stationarity_verdicts", "has_seasonality",
]


# ── Session State ────────────────────────────────────────────────────────────────


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "df": None,
        "results": [],
        "synthesis": "",
        "gemini_ds": "",
        "gemini_ba": "",
        "analysis_run": False,
        "gemini_enabled": False,
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "_file_id": "",
        "dataset_type": "",
        "date_column": None,
        "pipeline_log": [],
        "agent_errors": {},
        "stationarity_verdicts": {},
        "has_seasonality": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Feature Gating ────────────────────────────────────────────────────────────


def llm_available() -> bool:
    return bool(st.session_state.get("api_key", ""))


# ── Agent Pipeline ────────────────────────────────────────────────────────────


def run_agents(df: pd.DataFrame, max_trend_cols: int) -> Tuple[str, List[AnalysisResult]]:
    orchestrator = OrchestratorAgent()
    synthesis, results = orchestrator.analyze(
        df, query="Thorough analysis", max_trend_columns=max_trend_cols
    )
    logger.info(f"Agent pipeline complete: {len(results)} agents ran.")
    return synthesis, results


def get_gemini_summary(
    df: pd.DataFrame,
    results: List[AnalysisResult],
    api_key: str,
) -> Tuple[str, str]:
    """Call Gemini for dual-perspective insights. Returns ('', '') on any error."""
    try:
        client = GeminiClient(api_key=api_key)
        return client.summarize(df, results)
    except Exception as e:
        logger.error(f"Gemini summary error: {e}")
        return "", ""


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> Tuple[Any, int, bool, str]:
    st.sidebar.title("Data Analysis Tool")
    st.sidebar.markdown("Upload a CSV, configure settings, then click **Run Analysis**.")

    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    max_trend_cols = st.sidebar.slider("Max trend columns", min_value=1, max_value=10, value=5)

    st.sidebar.divider()

    # API key input — pre-filled from .env if available
    api_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        placeholder="AIza...",
        help="Paste your Gemini API key. The key lives only in this session.",
    )
    st.session_state.api_key = api_key_input.strip()

    if llm_available():
        gemini_on = st.sidebar.toggle("Enable AI Insights", value=False)
    else:
        st.sidebar.caption("Enter a Gemini API key above to enable AI insights.")
        gemini_on = False

    # Dataset type badge (shown after analysis)
    ss = st.session_state
    if ss.dataset_type:
        badge = {"time_series": "🕐 Time Series", "cross_sectional": "📋 Cross-Sectional",
                 "panel": "🗂 Panel"}.get(ss.dataset_type, ss.dataset_type)
        st.sidebar.info(f"Dataset type: **{badge}**")

    # Agent errors expander
    if ss.agent_errors:
        with st.sidebar.expander(f"⚠ Analysis warnings ({len(ss.agent_errors)})"):
            for agent, err in ss.agent_errors.items():
                st.markdown(f"**{agent}**: {err}")

    return uploaded, max_trend_cols, gemini_on, st.session_state.api_key


# ── Data Scientist Tab ────────────────────────────────────────────────────────


def render_ds_tab(
    df: pd.DataFrame,
    results: List[AnalysisResult],
    gemini_insights: str,
) -> None:
    profiler = next((r for r in results if r.agent_name == "DataProfiler"), None)
    quality = next((r for r in results if r.agent_name == "DataQuality"), None)
    eda = next((r for r in results if r.agent_name == "EDA"), None)
    causal = next((r for r in results if r.agent_name == "CausalFlag"), None)
    trend = next((r for r in results if r.agent_name == "TrendAnalysis"), None)
    seasonality = next((r for r in results if r.agent_name == "Seasonality"), None)
    stationarity = next((r for r in results if r.agent_name == "Stationarity"), None)
    autocorr = next((r for r in results if r.agent_name == "Autocorrelation"), None)
    cross_sec = next((r for r in results if r.agent_name == "CrossSectionalSummary"), None)

    # Use the profiler's filtered numeric list so identifier columns are excluded
    _profiler_num_cols: List[str] = (
        profiler.data.get("numeric_columns", []) if profiler and profiler.data else []
    )
    numeric = (
        df[_profiler_num_cols]
        if _profiler_num_cols
        else df.select_dtypes(include=[np.number])
    )

    # ── 1. Dataset Overview ──────────────────────────────────────────────────
    st.subheader("Dataset Overview")
    if profiler and profiler.data:
        d = profiler.data
        shape = d.get("shape", (0, 0))
        mem = d.get("memory_mb", 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{shape[0]:,}")
        c2.metric("Columns", shape[1])
        c3.metric("Numeric", len(d.get("numeric_columns", [])))
        c4.metric("Categorical", len(d.get("categorical_columns", [])))
        c5.metric("Memory", f"{mem:.1f} MB")

        excluded = d.get("excluded_columns", [])
        if excluded:
            st.caption(
                f"**{len(excluded)} column(s) excluded from analysis** (near-unique identifiers): "
                + ", ".join(f"`{c}`" for c in excluded)
            )

    # ── 2. Data Quality ──────────────────────────────────────────────────────
    st.subheader("Data Quality")
    if quality and quality.data:
        qd = quality.data
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Duplicate rows", f"{qd.get('duplicate_rows', 0):,}")
        q2.metric("High-missing cols", qd.get("high_missing_columns_count", 0))
        q3.metric("Constant cols", len(qd.get("constant_columns", [])))
        q4.metric("Cols with outliers", len(qd.get("outlier_summary", {})))

        missing_series = (df.isna().mean() * 100).sort_values(ascending=False).head(10)
        cols_with_missing = missing_series[missing_series > 0]
        if not cols_with_missing.empty:
            fig = px.bar(
                x=cols_with_missing.index.astype(str),
                y=cols_with_missing.values,
                labels={"x": "Column", "y": "Missing %"},
                title="Missing Values by Column",
                color_discrete_sequence=[COLORS["negative"]],
            )
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        else:
            st.success("No missing values detected.")

        outlier_data = qd.get("outlier_summary", {})
        if outlier_data:
            with st.expander(f"IQR outlier details ({len(outlier_data)} columns)"):
                out_df = pd.DataFrame(
                    [{"Column": c, "Outliers": v["count"], "%": v["pct"],
                      "Lower bound": v["lower_bound"], "Upper bound": v["upper_bound"]}
                     for c, v in outlier_data.items()]
                )
                st.dataframe(out_df, use_container_width=True)

    # ── 3. Statistical Summary ───────────────────────────────────────────────
    st.subheader("Statistical Summary")
    if not numeric.empty:
        with st.expander("Descriptive statistics (all numeric columns)", expanded=True):
            st.dataframe(numeric.describe().T.round(4), use_container_width=True)

        skew = (
            numeric.skew(numeric_only=True)
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(5)
        )
        if not skew.empty:
            skew_df = skew.reset_index()
            skew_df.columns = ["Column", "Skewness"]
            fig = px.bar(
                skew_df, x="Column", y="Skewness",
                title="Top 5 Most Skewed Columns",
                color="Skewness",
                color_continuous_scale=["#1B9E77", "#f7f7f7", "#D95F02"],
                color_continuous_midpoint=0,
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # ── 4. Distributions ─────────────────────────────────────────────────────
    st.subheader("Distributions")
    num_cols_4 = numeric.columns.tolist()[:4]
    if num_cols_4:
        col_a, col_b = st.columns(2)
        for i, col in enumerate(num_cols_4):
            fig = px.histogram(
                df, x=col, nbins=30, title=f"Distribution of {col}",
                color_discrete_sequence=[COLORS["neutral"]],
            )
            fig.update_layout(height=270, showlegend=False)
            (col_a if i % 2 == 0 else col_b).plotly_chart(
                fig, use_container_width=True, config=CHART_CONFIG
            )

    # ── 5. Correlation Heatmap ───────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    if eda and eda.data and eda.data.get("correlation_matrix"):
        corr_df = pd.DataFrame(eda.data["correlation_matrix"])
        if not corr_df.empty and corr_df.shape[1] >= 2:
            fig = px.imshow(
                corr_df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Pearson Correlation Matrix", aspect="auto",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            top_corrs = eda.data.get("top_correlations", [])
            if top_corrs:
                with st.expander("Top correlations"):
                    st.dataframe(
                        pd.DataFrame(top_corrs).rename(
                            columns={"a": "Column A", "b": "Column B", "r": "Pearson r"}
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

    # ── 6. Causal Flags ──────────────────────────────────────────────────────
    if causal and causal.data:
        warnings = causal.data.get("causal_warnings", [])
        strong_warnings = [w for w in warnings if w.get("level") == "strong"]
        if strong_warnings:
            with st.expander(f"Causal interpretation warnings ({len(strong_warnings)} strong)"):
                for w in strong_warnings:
                    st.warning(w["message"])
        if causal.data.get("has_binary_treatment_column"):
            col = causal.data.get("potential_treatment_col")
            st.info(
                f"Binary column '{col}' detected — possible A/B test or treatment/control structure."
            )

    # ── 7. Cross-Sectional Segment Analysis ──────────────────────────────────
    if cross_sec and cross_sec.data:
        top_group = cross_sec.data.get("top_grouping_column")
        if top_group:
            st.subheader("Segment Analysis")
            st.caption(f"Most discriminating grouping column: **{top_group}**")
            segs = cross_sec.data.get("segment_summaries", {}).get(top_group, {})
            if segs:
                num_col = list(segs.keys())[0] if segs else None
                if num_col:
                    seg_rows = segs.get(num_col, {})
                    seg_df = pd.DataFrame.from_dict(seg_rows, orient="index").round(4)
                    seg_df.index.name = top_group
                    with st.expander(f"Segment stats: {num_col} by {top_group}"):
                        st.dataframe(seg_df.reset_index(), use_container_width=True)

    # ── 8. Time Series: Trend Analysis ───────────────────────────────────────
    if trend and trend.data and trend.data.get("trend_metrics"):
        st.subheader("Trend Analysis")
        date_col = trend.data["date_column"]
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col)
        rw = trend.data.get("rolling_window", 7)
        x_dates = work[date_col]

        for metric in trend.data["trend_metrics"][:5]:
            col = metric["column"]
            direction = metric["direction"]
            pct = metric["pct_change"]
            slope = metric["slope"]
            p_val = metric.get("p_value", 1.0)
            r2 = metric.get("r_squared", 0.0)
            has_break = metric.get("structural_break", False)
            color = (
                COLORS["positive"] if direction == "upward"
                else COLORS["negative"] if direction == "downward"
                else COLORS["neutral"]
            )

            y_series = work[col].astype(float)
            x_num = np.arange(len(work), dtype=float)

            fig = go.Figure()

            # Trace 1 — raw time series (thin, semi-transparent)
            fig.add_trace(go.Scatter(
                x=x_dates, y=y_series,
                mode="lines", name=col,
                line={"color": color, "width": 1.2},
                opacity=0.6,
            ))

            # Trace 2 — rolling average (the smoothed signal)
            if metric.get("rolling_mean") and len(metric["rolling_mean"]) == len(work):
                fig.add_trace(go.Scatter(
                    x=x_dates, y=metric["rolling_mean"],
                    mode="lines", name=f"{rw}-period avg",
                    line={"color": color, "width": 2.5},
                ))

            # Trace 3 — OLS trend line (straight line showing direction)
            x_mean = x_num.mean()
            intercept = float(y_series.mean()) - slope * x_mean
            ols_y = slope * x_num + intercept
            fig.add_trace(go.Scatter(
                x=x_dates, y=ols_y,
                mode="lines", name=f"OLS trend (R²={r2:.2f})",
                line={"color": "#333333", "dash": "dot", "width": 1.5},
            ))

            # Structural break — vertical line at the midpoint
            if has_break and len(work) >= 4:
                break_date = x_dates.iloc[len(work) // 2]
                fig.add_vline(
                    x=str(break_date),
                    line_dash="dash", line_color=COLORS["negative"], opacity=0.7,
                    annotation_text="Structural break",
                    annotation_position="top right",
                    annotation_font_color=COLORS["negative"],
                )

            sig_label = "significant" if p_val < 0.05 else "not significant"
            fig.update_layout(
                title=f"{col} — {direction.title()} Trend ({pct:+.1f}%)",
                height=340,
                xaxis_title="Date",
                yaxis_title=col,
                legend={"orientation": "h", "y": -0.15},
                margin={"t": 50, "b": 60},
            )
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            st.caption(
                f"OLS slope: {slope:.4f} | p-value: {p_val:.3f} ({sig_label}) | R²: {r2:.3f}"
                + (" | ⚠ Structural break detected at midpoint" if has_break else "")
            )

    # ── 9. Seasonality ───────────────────────────────────────────────────────
    if seasonality and seasonality.data:
        seasonal_class = seasonality.data.get("seasonal_classification", {})
        seasonal_cols = [c for c, v in seasonal_class.items() if v in ("seasonal", "weakly_seasonal")]
        if seasonal_cols:
            st.subheader("Seasonality")
            date_col_s: Optional[str] = (
                profiler.data.get("date_column") if profiler and profiler.data else None
            )
            if date_col_s:
                work_s = df.copy()
                work_s[date_col_s] = pd.to_datetime(work_s[date_col_s], errors="coerce")
                work_s = work_s.dropna(subset=[date_col_s]).sort_values(date_col_s)

                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                work_s["_month_label"] = pd.Categorical(
                    work_s[date_col_s].dt.strftime("%b"), categories=month_order, ordered=True
                )
                work_s["_dow_label"] = pd.Categorical(
                    work_s[date_col_s].dt.strftime("%a"), categories=dow_order, ordered=True
                )

                for col in seasonal_cols[:3]:
                    cls = seasonal_class.get(col, "")
                    st.markdown(f"**{col}** — {cls.replace('_', ' ')}")
                    c_month, c_dow = st.columns(2)
                    with c_month:
                        fig = px.box(
                            work_s, x="_month_label", y=col,
                            title="Distribution by Month",
                            color_discrete_sequence=[COLORS["neutral"]],
                            labels={"_month_label": "Month", col: col},
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
                    with c_dow:
                        fig = px.box(
                            work_s, x="_dow_label", y=col,
                            title="Distribution by Day of Week",
                            color_discrete_sequence=[COLORS["accent"]],
                            labels={"_dow_label": "Day", col: col},
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # ── 10. Stationarity ─────────────────────────────────────────────────────
    if stationarity and stationarity.data:
        verdicts = stationarity.data.get("stationarity_verdict", {})
        transforms = stationarity.data.get("recommended_transforms", {})
        if verdicts:
            st.subheader("Stationarity (ADF + KPSS)")
            verdict_df = pd.DataFrame([
                {"Column": col, "Verdict": v, "Recommended transform": transforms.get(col, "—")}
                for col, v in verdicts.items()
            ])
            st.dataframe(verdict_df, use_container_width=True)

    # ── 11. Autocorrelation (ACF / PACF / Lag Plot) ───────────────────────────
    if autocorr and autocorr.data:
        acf_vals = autocorr.data.get("acf_values", {})
        pacf_vals = autocorr.data.get("pacf_values", {})
        lag_data = autocorr.data.get("lag_data", {})
        sig_lags = autocorr.data.get("significant_lags", {})
        conf = autocorr.data.get("confidence_band", 0.2)
        arima_hints = autocorr.data.get("arima_hints", {})
        if acf_vals:
            st.subheader("Autocorrelation (ACF / PACF / Lag Plot)")
            available_cols = list(acf_vals.keys())
            col_to_show = st.selectbox(
                "Column", available_cols, key="autocorr_col_select"
            ) if len(available_cols) > 1 else available_cols[0]

            # Row 1 — ACF and PACF side by side
            acf_col, pacf_col = st.columns(2)
            acf_series = acf_vals[col_to_show]
            lags = list(range(len(acf_series)))

            fig_acf = go.Figure(go.Bar(
                x=lags, y=acf_series,
                marker_color=[
                    COLORS["negative"] if abs(v) > conf else COLORS["neutral"]
                    for v in acf_series
                ],
                name="ACF",
            ))
            fig_acf.add_hline(y=conf, line_dash="dash", line_color="red", opacity=0.5)
            fig_acf.add_hline(y=-conf, line_dash="dash", line_color="red", opacity=0.5)
            fig_acf.update_layout(
                title=f"ACF — {col_to_show}",
                xaxis_title="Lag", yaxis_title="Autocorrelation",
                height=280, showlegend=False,
            )
            acf_col.plotly_chart(fig_acf, use_container_width=True, config=CHART_CONFIG)

            if col_to_show in pacf_vals:
                pacf_series = pacf_vals[col_to_show]
                p_lags = list(range(len(pacf_series)))
                fig_pacf = go.Figure(go.Bar(
                    x=p_lags, y=pacf_series,
                    marker_color=[
                        COLORS["accent"] if abs(v) > conf else COLORS["neutral"]
                        for v in pacf_series
                    ],
                    name="PACF",
                ))
                fig_pacf.add_hline(y=conf, line_dash="dash", line_color="red", opacity=0.5)
                fig_pacf.add_hline(y=-conf, line_dash="dash", line_color="red", opacity=0.5)
                fig_pacf.update_layout(
                    title=f"PACF — {col_to_show}",
                    xaxis_title="Lag", yaxis_title="Partial Autocorrelation",
                    height=280, showlegend=False,
                )
                pacf_col.plotly_chart(fig_pacf, use_container_width=True, config=CHART_CONFIG)

            # Significant lags summary
            col_sig = sig_lags.get(col_to_show, [])
            if col_sig:
                st.caption(f"Significant autocorrelation at lags: {col_sig}")
            else:
                st.caption("No significant autocorrelation detected at standard lags (1, 7, 14, 30).")

            # Row 2 — Lag plot (Y_t vs Y_{t-1}): framework §6 requirement
            if col_to_show in lag_data:
                ld = lag_data[col_to_show]
                y_curr = ld.get("y", [])
                y_lag1 = ld.get("y_lag1", [])
                if y_curr and y_lag1:
                    fig_lag = px.scatter(
                        x=y_lag1, y=y_curr,
                        labels={"x": f"{col_to_show} (t-1)", "y": f"{col_to_show} (t)"},
                        title=f"Lag Plot — {col_to_show} (Y_t vs Y_{{t-1}})",
                        color_discrete_sequence=[COLORS["neutral"]],
                        opacity=0.5,
                    )
                    fig_lag.update_traces(marker_size=4)
                    fig_lag.update_layout(height=340)
                    st.plotly_chart(fig_lag, use_container_width=True, config=CHART_CONFIG)
                    st.caption(
                        "A diagonal cluster → strong positive autocorrelation. "
                        "A circular cloud → no autocorrelation. "
                        "An ellipse → moderate autocorrelation."
                    )

            if arima_hints and col_to_show in arima_hints:
                with st.expander("ARIMA order hint (rule-of-thumb)"):
                    st.markdown(arima_hints[col_to_show])

    # ── 12. AI Technical Insights ────────────────────────────────────────────
    if gemini_insights:
        st.subheader("AI Technical Insights")
        st.markdown(gemini_insights)


# ── Business Analyst Tab ──────────────────────────────────────────────────────


def render_ba_tab(
    df: pd.DataFrame,
    results: List[AnalysisResult],
    synthesis: str,
    gemini_summary: str,
) -> None:
    quality = next((r for r in results if r.agent_name == "DataQuality"), None)
    trend = next((r for r in results if r.agent_name == "TrendAnalysis"), None)
    eda = next((r for r in results if r.agent_name == "EDA"), None)
    cross_sec = next((r for r in results if r.agent_name == "CrossSectionalSummary"), None)
    synthesis_r = next((r for r in results if r.agent_name == "Synthesis"), None)

    trend_metrics: List[Dict[str, Any]] = (
        trend.data.get("trend_metrics", []) if trend and trend.data else []
    )
    date_col_name: Optional[str] = (
        trend.data.get("date_column") if trend and trend.data else None
    )
    dupes = quality.data.get("duplicate_rows", 0) if quality and quality.data else 0
    dupe_pct = (dupes / len(df) * 100) if len(df) > 0 else 0.0

    date_range = "N/A"
    if date_col_name:
        parsed = pd.to_datetime(df[date_col_name], errors="coerce").dropna()
        if not parsed.empty:
            date_range = (
                f"{parsed.min().strftime('%b %Y')} – {parsed.max().strftime('%b %Y')}"
            )

    # ── 1. KPI Cards ─────────────────────────────────────────────────────────
    st.subheader("At a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Data fields", df.shape[1])
    c3.metric("Trends detected", len(trend_metrics))
    c4.metric("Duplicate rate", f"{dupe_pct:.1f}%")
    if date_range != "N/A":
        st.caption(f"Date range: {date_range}")

    # ── 2. Plain-English Trends ───────────────────────────────────────────────
    if trend_metrics:
        st.subheader("What Is Trending")
        for m in trend_metrics:
            col = m["column"]
            direction = m["direction"]
            pct = m["pct_change"]
            p_val = m.get("p_value", 1.0)
            if p_val >= 0.05:
                st.info(f"**{col}** shows a {direction} movement ({pct:+.1f}%), but the trend is not statistically significant.")
            elif direction == "upward":
                st.success(f"**{col}** grew by {pct:+.1f}% over the observed period.")
            elif direction == "downward":
                st.warning(f"**{col}** declined by {abs(pct):.1f}% over the observed period.")
            else:
                st.info(f"**{col}** remained relatively stable ({pct:+.1f}% change).")

    else:
        st.info("No time-based trends were detected in this dataset.")

    # ── 3. Visual Trend Charts (clean, no slope annotations) ─────────────────
    if trend_metrics and date_col_name:
        st.subheader("Key Trends Over Time")
        work = df.copy()
        work[date_col_name] = pd.to_datetime(work[date_col_name], errors="coerce")
        work = work.dropna(subset=[date_col_name]).sort_values(date_col_name)
        for metric in trend_metrics[:3]:
            col = metric["column"]
            color = (
                COLORS["positive"] if metric["direction"] == "upward"
                else COLORS["negative"] if metric["direction"] == "downward"
                else COLORS["neutral"]
            )
            fig = px.line(
                work, x=date_col_name, y=col,
                title=f"{col} Over Time",
                color_discrete_sequence=[color],
            )
            fig.update_layout(height=280, xaxis_title="Date", yaxis_title=col, showlegend=False)
            fig.update_traces(line_width=2.2)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # ── 4. Segment Highlights (cross-sectional) ───────────────────────────────
    if cross_sec and cross_sec.data and cross_sec.data.get("top_grouping_column"):
        top_group = cross_sec.data["top_grouping_column"]
        segs = cross_sec.data.get("segment_summaries", {}).get(top_group, {})
        if segs:
            st.subheader(f"Segment Breakdown: {top_group}")
            num_col = list(segs.keys())[0]
            seg_rows = segs.get(num_col, {})
            if seg_rows:
                cats = list(seg_rows.keys())
                means = [seg_rows[c].get("mean", 0) for c in cats]
                fig = px.bar(
                    x=cats, y=means,
                    labels={"x": top_group, "y": f"Mean {num_col}"},
                    title=f"Average {num_col} by {top_group}",
                    color_discrete_sequence=[COLORS["neutral"]],
                )
                fig.update_layout(height=300, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # ── 5. Key Relationships ──────────────────────────────────────────────────
    if eda and eda.data and eda.data.get("top_correlations"):
        st.subheader("Key Relationships in the Data")
        for c in eda.data["top_correlations"][:5]:
            r = c["r"]
            a, b = c["a"], c["b"]
            abs_r = abs(r)
            strength = "strongly" if abs_r >= 0.7 else "moderately" if abs_r >= 0.4 else "weakly"
            direction_word = "together" if r > 0 else "inversely"
            st.markdown(f"- **{a}** and **{b}** move {strength} {direction_word} (r = {r:.2f})")

    # ── 6. Summary & Takeaways ────────────────────────────────────────────────
    st.subheader("Summary & Takeaways")
    if gemini_summary:
        st.markdown(gemini_summary)
    elif synthesis_r and synthesis_r.data:
        bullets: List[str] = synthesis_r.data.get("business_bullets", [])
        for bullet in bullets:
            st.markdown(f"- {bullet}" if not bullet.startswith("-") else bullet)
        action = synthesis_r.data.get("recommended_action", "")
        if action:
            st.info(f"**Recommended action:** {action}")
    else:
        # Fallback to raw synthesis text
        for line in synthesis.strip().split("\n"):
            line = line.strip()
            if line:
                st.markdown(line if line.startswith(("-", "*", "#")) else f"- {line}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Data Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    ss = st.session_state
    uploaded, max_trend_cols, gemini_on, api_key = render_sidebar()
    ss.gemini_enabled = gemini_on

    # ── File upload / reset ───────────────────────────────────────────────────
    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if ss._file_id != file_id:
            ss._file_id = file_id
            for key in _RESET_ON_UPLOAD:
                ss[key] = [] if isinstance(ss[key], list) else ({} if isinstance(ss[key], dict) else False if isinstance(ss[key], bool) else "")
            ss.df = None
            try:
                ss.df = pd.read_csv(uploaded)
                logger.info(f"Loaded '{uploaded.name}': {ss.df.shape}")
            except Exception as e:
                logger.error(f"CSV load failed: {e}")
                st.error("Could not read the file. Make sure it is a valid CSV.")
    elif uploaded is None and ss._file_id:
        ss._file_id = ""
        ss.df = None
        for key in _RESET_ON_UPLOAD:
            ss[key] = [] if isinstance(ss[key], list) else ({} if isinstance(ss[key], dict) else False if isinstance(ss[key], bool) else "")

    run_button = st.sidebar.button("Run Analysis", type="primary", disabled=(ss.df is None))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_button and ss.df is not None:
        with st.spinner("Running analysis agents..."):
            try:
                synthesis, results = run_agents(ss.df, max_trend_cols)
                ss.synthesis = synthesis
                ss.results = results
                ss.analysis_run = True

                # Extract routing metadata for sidebar badge and TS rendering
                profiler_result = next((r for r in results if r.agent_name == "DataProfiler"), None)
                if profiler_result and profiler_result.data:
                    ss.dataset_type = profiler_result.data.get("dataset_type", "")
                    ss.date_column = profiler_result.data.get("date_column")

                # Collect agent errors
                ss.agent_errors = {
                    r.agent_name: r.error
                    for r in results
                    if r.error
                }

                # Seasonality flag
                seas_result = next((r for r in results if r.agent_name == "Seasonality"), None)
                if seas_result and seas_result.data:
                    sc = seas_result.data.get("seasonal_classification", {})
                    ss.has_seasonality = any(v == "seasonal" for v in sc.values())

                # Stationarity verdicts
                stat_result = next((r for r in results if r.agent_name == "Stationarity"), None)
                if stat_result and stat_result.data:
                    ss.stationarity_verdicts = stat_result.data.get("stationarity_verdict", {})

            except Exception as e:
                logger.error(f"Agent pipeline failed: {e}")
                st.error(f"Analysis failed: {e}")

        if ss.analysis_run and gemini_on and api_key:
            with st.spinner("Generating AI insights with Gemini..."):
                ds_text, ba_text = get_gemini_summary(ss.df, ss.results, api_key)
                ss.gemini_ds = ds_text
                ss.gemini_ba = ba_text
                if not ds_text and not ba_text:
                    st.warning("AI insights unavailable — using rule-based analysis instead.")
        else:
            ss.gemini_ds = ""
            ss.gemini_ba = ""

    # ── Page content ──────────────────────────────────────────────────────────
    st.title("Data Analysis Tool")

    if ss.df is None:
        st.info(
            "Upload a CSV file using the sidebar to get started. "
            "Then click **Run Analysis** to explore your data from two perspectives."
        )
        return

    if not ss.analysis_run:
        fname = uploaded.name if uploaded else "your file"
        st.info(
            f"**{fname}** loaded — {ss.df.shape[0]:,} rows × {ss.df.shape[1]} columns. "
            "Click **Run Analysis** in the sidebar to begin."
        )
        with st.expander("Preview data (first 20 rows)"):
            st.dataframe(ss.df.head(20), use_container_width=True)
        return

    ds_tab, ba_tab = st.tabs(["📊 Data Scientist", "💼 Business Analyst"])

    with ds_tab:
        try:
            render_ds_tab(ss.df, ss.results, ss.gemini_ds)
        except Exception as e:
            logger.error(f"DS tab render error: {e}")
            st.error(f"Error rendering Data Scientist view: {e}")

    with ba_tab:
        try:
            render_ba_tab(ss.df, ss.results, ss.synthesis, ss.gemini_ba)
        except Exception as e:
            logger.error(f"BA tab render error: {e}")
            st.error(f"Error rendering Business Analyst view: {e}")


if __name__ == "__main__":
    main()
