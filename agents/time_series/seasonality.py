"""
SeasonalityAgent — Part III §4 Seasonality Detection

Computes monthly and day-of-week means, evaluates ACF at seasonal lags,
and classifies each series as seasonal / weakly_seasonal / not_seasonal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_SEASONAL_ACF_THRESHOLD = 0.30
_WEAK_SEASONAL_ACF_THRESHOLD = 0.15
_SEASONAL_LAGS = {
    "weekly": 7,
    "monthly": 12,
    "quarterly": 4,
}


class SeasonalityAgent(BaseAgent):
    """Part III §4 — Seasonality: monthly/DOW means, ACF at seasonal lags."""

    name: str = "Seasonality"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        profiler_data = context.get("DataProfiler", {})
        date_col: Optional[str] = profiler_data.get("date_column")
        numeric_cols: List[str] = profiler_data.get("numeric_columns", [])

        if not date_col or not numeric_cols:
            return AnalysisResult(
                agent_name=self.name,
                findings="No date column — seasonality analysis skipped.",
                data={},
            )

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col)

        work["_month"] = work[date_col].dt.month
        work["_dow"] = work[date_col].dt.dayofweek  # 0=Monday

        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

        monthly_means: Dict[str, Dict[str, float]] = {}
        dow_means: Dict[str, Dict[str, float]] = {}
        acf_at_seasonal_lags: Dict[str, Dict[str, float]] = {}
        seasonal_classification: Dict[str, str] = {}

        findings: List[str] = []

        for col in numeric_cols[:8]:  # limit for performance
            series = work[col].astype(float)
            if series.isna().all():
                continue

            # Monthly means
            m_means = work.groupby("_month")[col].mean().dropna()
            monthly_means[col] = {
                month_names.get(int(k), str(k)): round(float(v), 4)
                for k, v in m_means.items()
            }

            # Day-of-week means
            d_means = work.groupby("_dow")[col].mean().dropna()
            dow_means[col] = {
                dow_names.get(int(k), str(k)): round(float(v), 4)
                for k, v in d_means.items()
            }

            # ACF at seasonal lags using pandas .autocorr()
            series_clean = series.interpolate(limit_direction="both").dropna()
            acf_vals: Dict[str, float] = {}
            for lag_name, lag in _SEASONAL_LAGS.items():
                if len(series_clean) > lag + 1:
                    acf_val = float(series_clean.autocorr(lag=lag))
                    acf_vals[lag_name] = round(acf_val if not np.isnan(acf_val) else 0.0, 4)
                else:
                    acf_vals[lag_name] = 0.0
            acf_at_seasonal_lags[col] = acf_vals

            # Classification based on strongest seasonal ACF
            max_acf = max(abs(v) for v in acf_vals.values()) if acf_vals else 0.0
            if max_acf >= _SEASONAL_ACF_THRESHOLD:
                classification = "seasonal"
            elif max_acf >= _WEAK_SEASONAL_ACF_THRESHOLD:
                classification = "weakly_seasonal"
            else:
                classification = "not_seasonal"
            seasonal_classification[col] = classification

            findings.append(
                f"  {col}: {classification} (max seasonal ACF={max_acf:.3f})"
            )

        n_seasonal = sum(1 for v in seasonal_classification.values() if v == "seasonal")
        findings.insert(0, f"Seasonality analysis: {n_seasonal}/{len(seasonal_classification)} columns classified seasonal.")

        logger.info(f"Seasonality: {n_seasonal} seasonal columns of {len(seasonal_classification)}")

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "seasonal_classification": seasonal_classification,
                "monthly_means": monthly_means,
                "dow_means": dow_means,
                "acf_at_seasonal_lags": acf_at_seasonal_lags,
            },
        )
