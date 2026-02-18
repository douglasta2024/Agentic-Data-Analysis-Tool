"""
TrendAnalysisAgent — Part I §4 + Part III §2-3

Enforces temporal ordering (Part III Rule 1), decomposes structure,
computes OLS trend, rolling averages, structural break detection,
and percent change per numeric column.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_DEFAULT_MAX_COLS = 5
_DIRECTION_THRESHOLD = 10.0   # % change needed to classify as up/down vs flat
_PVALUE_SIGNIFICANCE = 0.05


def _infer_rolling_window(n_rows: int, date_col: pd.Series) -> int:
    """Infer a sensible rolling average window based on date frequency."""
    try:
        diffs = date_col.dropna().sort_values().diff().dropna()
        median_diff = diffs.median()
        if pd.isnull(median_diff):
            return 7
        days = median_diff.days if hasattr(median_diff, "days") else 1
        if days <= 1:
            return 7    # daily data → 7-day window
        elif days <= 7:
            return 4    # weekly data → 4-week window
        else:
            return 3    # monthly+ data → 3-period window
    except Exception:
        return 7


class TrendAnalysisAgent(BaseAgent):
    """Part III §2-3 — Trend Analysis: OLS slope, rolling average, structural break."""

    name: str = "TrendAnalysis"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        profiler_data = context.get("DataProfiler", {})
        date_col: Optional[str] = profiler_data.get("date_column")
        numeric_cols: List[str] = profiler_data.get("numeric_columns", [])
        max_cols: int = int(context.get("max_trend_columns", _DEFAULT_MAX_COLS))

        if not date_col or not numeric_cols:
            return AnalysisResult(
                agent_name=self.name,
                findings="No date column or numeric columns available for trend analysis.",
                data={"date_column": date_col, "trend_metrics": [], "rolling_window": 0},
            )

        # df is pre-sorted by the orchestrator (Part III Rule 1)
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])

        if work.empty:
            return AnalysisResult(
                agent_name=self.name,
                findings="Date column could not be parsed — no records remain.",
                data={"date_column": date_col, "trend_metrics": [], "rolling_window": 0},
            )

        rolling_window = _infer_rolling_window(len(work), work[date_col])

        # Select columns with highest variance for trend analysis
        candidate_cols = (
            work[numeric_cols]
            .var(numeric_only=True)
            .sort_values(ascending=False)
            .index.tolist()[:max_cols]
        )

        trend_metrics: List[Dict[str, Any]] = []
        findings: List[str] = [
            f"Date column: {date_col}  |  {len(work):,} records  |  Rolling window: {rolling_window}",
        ]
        x = np.arange(len(work), dtype=float)

        for col in candidate_cols:
            series = work[col].astype(float).interpolate(limit_direction="both")
            if series.isna().all() or len(series) < 4:
                continue

            # OLS slope and p-value
            slope, intercept, r_val, p_val, _ = scipy_stats.linregress(x, series.values)
            slope = float(slope)
            p_val = float(p_val)

            start = float(series.iloc[0]) if series.iloc[0] != 0 else 1e-9
            end = float(series.iloc[-1])
            pct_change = ((end - start) / abs(start)) * 100

            if abs(pct_change) > _DIRECTION_THRESHOLD and p_val < _PVALUE_SIGNIFICANCE:
                direction = "upward" if pct_change > 0 else "downward"
            else:
                direction = "stable"

            # Rolling mean for chart rendering
            rolling_mean = (
                series.rolling(window=rolling_window, min_periods=1)
                .mean()
                .round(4)
                .tolist()
            )

            # Structural break: compare first half mean vs second half mean
            mid = len(series) // 2
            first_half = series.iloc[:mid].dropna()
            second_half = series.iloc[mid:].dropna()
            structural_break = False
            if len(first_half) >= 3 and len(second_half) >= 3:
                _, break_p = scipy_stats.ttest_ind(first_half, second_half, equal_var=False)
                structural_break = bool(break_p < _PVALUE_SIGNIFICANCE)

            metric = {
                "column": col,
                "slope": round(slope, 6),
                "p_value": round(p_val, 4),
                "r_squared": round(r_val**2, 4),
                "start": round(start, 3),
                "end": round(end, 3),
                "pct_change": round(pct_change, 2),
                "direction": direction,
                "rolling_mean": rolling_mean,
                "structural_break": structural_break,
            }
            trend_metrics.append(metric)
            sig = " (significant)" if p_val < _PVALUE_SIGNIFICANCE else " (not significant)"
            findings.append(
                f"  {col}: {direction} ({pct_change:+.1f}%){sig}, "
                f"slope={slope:.4f}, R²={r_val**2:.3f}"
            )

        logger.info(f"TrendAnalysis: {len(trend_metrics)} columns analysed")

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "date_column": date_col,
                "trend_metrics": trend_metrics,
                "rolling_window": rolling_window,
            },
        )
