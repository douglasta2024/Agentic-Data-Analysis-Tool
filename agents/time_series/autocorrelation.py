"""
AutocorrelationAgent — Part III §6 Autocorrelation Examination

Computes ACF and PACF up to 40 lags, generates lag-plot data,
flags significant autocorrelation, and provides an ARIMA order hint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_MAX_LAGS = 40
_SIGNIFICANCE_LAGS = [1, 7, 14, 30]
_MAX_COLS = 5  # limit for performance
_CONFIDENCE_LEVEL = 0.95


def _confidence_band(n: int) -> float:
    """95% confidence band for ACF: ±1.96 / sqrt(n)."""
    return 1.96 / np.sqrt(n) if n > 0 else 0.2


def _arima_hint(acf_vals: List[float], pacf_vals: List[float], conf: float) -> str:
    """
    Rule-of-thumb ARIMA order hint.
    - ACF cuts off at lag q → MA(q) component
    - PACF cuts off at lag p → AR(p) component
    """
    # Find where ACF first becomes insignificant
    q = 0
    for i, v in enumerate(acf_vals[1:], start=1):
        if abs(v) > conf:
            q = i
        else:
            break

    # Find where PACF first becomes insignificant
    p = 0
    for i, v in enumerate(pacf_vals[1:], start=1):
        if abs(v) > conf:
            p = i
        else:
            break

    return f"Suggested ARIMA order: ({p}, d, {q}) — verify with information criteria."


class AutocorrelationAgent(BaseAgent):
    """Part III §6 — Autocorrelation: ACF, PACF, lag plots, ARIMA hint."""

    name: str = "Autocorrelation"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        try:
            from statsmodels.tsa.stattools import acf, pacf
        except ImportError:
            return AnalysisResult(
                agent_name=self.name,
                findings="statsmodels not installed — autocorrelation analysis skipped. Run: pip install statsmodels",
                data={},
                error="statsmodels not available",
            )

        profiler_data = context.get("DataProfiler", {})
        numeric_cols: List[str] = profiler_data.get("numeric_columns", [])[:_MAX_COLS]
        date_col: Optional[str] = profiler_data.get("date_column")

        if not numeric_cols:
            return AnalysisResult(
                agent_name=self.name,
                findings="No numeric columns for autocorrelation analysis.",
                data={},
            )

        acf_values: Dict[str, List[float]] = {}
        pacf_values: Dict[str, List[float]] = {}
        lag_data: Dict[str, Dict[str, List[float]]] = {}
        significant_lags: Dict[str, List[int]] = {}
        arima_hints: Dict[str, str] = {}
        findings: List[str] = []

        for col in numeric_cols:
            series = df[col].astype(float).interpolate(limit_direction="both").dropna()
            n = len(series)
            if n < 15:
                continue

            n_lags = min(_MAX_LAGS, n // 2 - 1)
            conf = _confidence_band(n)

            try:
                acf_arr = acf(series.values, nlags=n_lags, fft=True)
                acf_values[col] = [round(float(v), 4) for v in acf_arr]
            except Exception as e:
                logger.warning(f"ACF failed for {col}: {e}")
                acf_values[col] = []
                continue

            try:
                pacf_arr = pacf(series.values, nlags=n_lags, method="yw")
                pacf_values[col] = [round(float(v), 4) for v in pacf_arr]
            except Exception as e:
                logger.warning(f"PACF failed for {col}: {e}")
                pacf_values[col] = [0.0] * len(acf_values[col])

            # Lag plot data (y[t] vs y[t-1])
            lag_data[col] = {
                "y": series.iloc[1:].round(4).tolist(),
                "y_lag1": series.iloc[:-1].round(4).tolist(),
            }

            # Significant lags at common check points
            sig_lags = [
                lag for lag in _SIGNIFICANCE_LAGS
                if lag < len(acf_values[col]) and abs(acf_values[col][lag]) > conf
            ]
            significant_lags[col] = sig_lags

            # ARIMA hint
            arima_hints[col] = _arima_hint(acf_values[col], pacf_values[col], conf)

            findings.append(
                f"  {col}: significant autocorr at lags {sig_lags if sig_lags else 'none'}. "
                f"{arima_hints[col]}"
            )

        findings.insert(0, f"Autocorrelation analysis: {len(acf_values)} columns processed.")

        logger.info(f"Autocorrelation: {len(acf_values)} columns processed")

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "acf_values": acf_values,
                "pacf_values": pacf_values,
                "lag_data": lag_data,
                "significant_lags": significant_lags,
                "arima_hints": arima_hints,
                "confidence_band": round(_confidence_band(len(df)), 4),
            },
        )
