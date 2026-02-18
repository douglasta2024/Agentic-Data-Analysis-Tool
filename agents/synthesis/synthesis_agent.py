"""
SynthesisAgent — Part I §12 Communication

Always runs last. Reads all upstream context dicts and produces
two structured output streams:
- technical_bullets: for the Data Scientist tab
- business_bullets: for the Business Analyst tab

Also formats the Gemini prompt when LLM is enabled.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """Part I §12 — Communication: synthesise all agent findings into two perspectives."""

    name: str = "Synthesis"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        technical: List[str] = []
        business: List[str] = []

        profiler = context.get("DataProfiler", {})
        quality = context.get("DataQuality", {})
        eda = context.get("EDA", {})
        causal = context.get("CausalFlag", {})
        trend = context.get("TrendAnalysis", {})
        seasonality = context.get("Seasonality", {})
        stationarity = context.get("Stationarity", {})
        autocorr = context.get("Autocorrelation", {})
        cross_sec = context.get("CrossSectionalSummary", {})

        dataset_type = profiler.get("dataset_type", "cross_sectional")

        # ── Technical bullets ────────────────────────────────────────────────
        shape = profiler.get("shape", (0, 0))
        technical.append(
            f"Dataset: {shape[0]:,} rows × {shape[1]} columns ({dataset_type.replace('_', ' ')})."
        )

        # Quality
        dupes = quality.get("duplicate_rows", 0)
        high_missing_ct = quality.get("high_missing_columns_count", 0)
        if dupes > 0:
            technical.append(f"Data quality: {dupes:,} duplicate rows detected.")
        if high_missing_ct > 0:
            technical.append(f"Data quality: {high_missing_ct} column(s) exceed 40% missing threshold.")
        outlier_ct = len(quality.get("outlier_summary", {}))
        if outlier_ct > 0:
            technical.append(f"Outliers (IQR): {outlier_ct} column(s) contain outliers.")

        # Correlations
        top_corr = eda.get("top_correlations", [])
        if top_corr:
            best = top_corr[0]
            technical.append(
                f"Strongest correlation: {best['a']} vs {best['b']} (r={best['r']:.3f})."
            )

        # Causal warnings
        causal_warnings = causal.get("causal_warnings", [])
        n_strong = sum(1 for w in causal_warnings if w.get("level") == "strong")
        if n_strong > 0:
            technical.append(
                f"Causal: {n_strong} strong correlation pair(s) — correlation ≠ causation."
            )

        # Trend
        trend_metrics = trend.get("trend_metrics", [])
        if trend_metrics:
            up = sum(1 for m in trend_metrics if m["direction"] == "upward")
            down = sum(1 for m in trend_metrics if m["direction"] == "downward")
            technical.append(f"Trends: {up} upward, {down} downward across {len(trend_metrics)} columns.")
            breaks = [m["column"] for m in trend_metrics if m.get("structural_break")]
            if breaks:
                technical.append(f"Structural breaks detected in: {', '.join(breaks)}.")

        # Stationarity
        verdicts = stationarity.get("stationarity_verdict", {})
        if verdicts:
            n_stat = sum(1 for v in verdicts.values() if v == "stationary")
            technical.append(f"Stationarity: {n_stat}/{len(verdicts)} series are stationary.")
            non_stat = [c for c, v in verdicts.items() if v != "stationary"]
            if non_stat:
                technical.append(f"Non-stationary series: {', '.join(non_stat[:4])}.")

        # Seasonality
        seasonal_class = seasonality.get("seasonal_classification", {})
        if seasonal_class:
            n_seasonal = sum(1 for v in seasonal_class.values() if v == "seasonal")
            if n_seasonal > 0:
                technical.append(
                    f"Seasonality: {n_seasonal}/{len(seasonal_class)} series show seasonal patterns."
                )

        # Autocorrelation
        sig_lags = autocorr.get("significant_lags", {})
        if sig_lags:
            autocorr_cols = [c for c, lags in sig_lags.items() if lags]
            if autocorr_cols:
                technical.append(
                    f"Significant autocorrelation detected in: {', '.join(autocorr_cols[:3])}."
                )

        # Cross-sectional
        top_group = cross_sec.get("top_grouping_column")
        if top_group:
            technical.append(
                f"Segment analysis: '{top_group}' is the most discriminating grouping column."
            )

        # ── Business bullets ─────────────────────────────────────────────────
        business.append(
            f"Your dataset contains {shape[0]:,} records across {shape[1]} fields."
        )

        if dupes > 0:
            dupe_pct = dupes / shape[0] * 100 if shape[0] > 0 else 0
            business.append(
                f"⚠ {dupes:,} duplicate rows found ({dupe_pct:.1f}% of data) — consider deduplication."
            )

        if trend_metrics:
            strongest = max(trend_metrics, key=lambda m: abs(m["pct_change"]), default=None)
            if strongest:
                direction_word = "grew" if strongest["direction"] == "upward" else "declined"
                pct = abs(strongest["pct_change"])
                business.append(
                    f"{strongest['column']} {direction_word} by {pct:.1f}% over the observed period."
                )

        if top_group:
            business.append(
                f"'{top_group}' has the biggest impact on your numeric metrics — "
                "a natural lens for further investigation."
            )

        if top_corr:
            best = top_corr[0]
            business.append(
                f"{best['a']} and {best['b']} are closely linked (r={best['r']:.2f}) — "
                "movements in one often accompany movements in the other."
            )

        # Recommended action
        recommended_action = _recommend_action(
            dataset_type, trend_metrics, verdicts, n_strong if causal_warnings else 0
        )
        business.append(f"Recommended action: {recommended_action}")

        logger.info(
            f"Synthesis: {len(technical)} technical bullets, {len(business)} business bullets"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(technical),
            data={
                "technical_bullets": technical,
                "business_bullets": business,
                "recommended_action": recommended_action,
                "top_concern": technical[1] if len(technical) > 1 else "",
            },
        )


def _recommend_action(
    dataset_type: str,
    trend_metrics: List[Dict[str, Any]],
    verdicts: Dict[str, str],
    n_strong_corr: int,
) -> str:
    if dataset_type in ("time_series", "panel"):
        non_stat = [c for c, v in verdicts.items() if v != "stationary"]
        if non_stat:
            return (
                f"Difference or detrend {', '.join(non_stat[:2])} before modelling — "
                "non-stationary series can produce spurious relationships."
            )
        downward_cols = [m["column"] for m in trend_metrics if m["direction"] == "downward"]
        if downward_cols:
            return (
                f"Investigate the downward trend in {downward_cols[0]} — "
                "identify whether this reflects a structural issue or seasonal trough."
            )
        return "Monitor the identified trends over the next reporting period."
    else:
        if n_strong_corr > 0:
            return (
                "Validate the strong correlations through domain expertise before using them "
                "in decision-making — they may reflect confounding rather than causation."
            )
        return "Use the segment analysis to identify the highest-value groups for targeted action."
