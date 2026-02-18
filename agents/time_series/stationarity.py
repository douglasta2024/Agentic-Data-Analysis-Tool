"""
StationarityAgent — Part III §5 Stationarity Check

Runs ADF and KPSS tests on each numeric column.
Interprets the joint result into one of four cases and recommends
a differencing order and/or log transform.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_PVALUE_SIGNIFICANCE = 0.05
_MAX_COLS = 10  # limit for performance


def _interpret_joint(adf_stationary: bool, kpss_stationary: bool) -> str:
    """
    Joint ADF + KPSS interpretation.

    ADF H0: unit root (non-stationary). Reject H0 → stationary.
    KPSS H0: stationary. Reject H0 → non-stationary.

    Case 1: ADF rejects AND KPSS does not reject → stationary
    Case 2: ADF does not reject AND KPSS rejects → unit root (non-stationary)
    Case 3: ADF rejects AND KPSS rejects → trend-stationary (trend present)
    Case 4: ADF does not reject AND KPSS does not reject → difference-stationary
    """
    if adf_stationary and kpss_stationary:
        return "stationary"
    elif not adf_stationary and not kpss_stationary:
        return "unit_root"
    elif adf_stationary and not kpss_stationary:
        return "trend_stationary"
    else:
        return "difference_stationary"


class StationarityAgent(BaseAgent):
    """Part III §5 — Stationarity: ADF + KPSS tests, differencing recommendations."""

    name: str = "Stationarity"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            return AnalysisResult(
                agent_name=self.name,
                findings="statsmodels not installed — stationarity tests skipped. Run: pip install statsmodels",
                data={},
                error="statsmodels not available",
            )

        profiler_data = context.get("DataProfiler", {})
        numeric_cols: List[str] = profiler_data.get("numeric_columns", [])[:_MAX_COLS]
        date_col: Optional[str] = profiler_data.get("date_column")

        if not numeric_cols:
            return AnalysisResult(
                agent_name=self.name,
                findings="No numeric columns for stationarity tests.",
                data={},
            )

        adf_results: Dict[str, Any] = {}
        kpss_results: Dict[str, Any] = {}
        stationarity_verdict: Dict[str, str] = {}
        recommended_transforms: Dict[str, str] = {}
        findings: List[str] = []

        for col in numeric_cols:
            series = df[col].astype(float).interpolate(limit_direction="both").dropna()
            if len(series) < 10:
                continue

            # ADF test
            try:
                adf_stat, adf_p, adf_lags, *_ = adfuller(series.values, autolag="AIC")
                adf_stationary = bool(adf_p < _PVALUE_SIGNIFICANCE)
                adf_results[col] = {
                    "statistic": round(float(adf_stat), 4),
                    "p_value": round(float(adf_p), 4),
                    "lags": int(adf_lags),
                    "is_stationary": adf_stationary,
                }
            except Exception as e:
                logger.warning(f"ADF test failed for {col}: {e}")
                adf_results[col] = {"error": str(e)}
                adf_stationary = False

            # KPSS test
            try:
                kpss_stat, kpss_p, kpss_lags, _ = kpss(series.values, regression="c", nlags="auto")
                kpss_stationary = bool(kpss_p > _PVALUE_SIGNIFICANCE)  # fail to reject H0 = stationary
                kpss_results[col] = {
                    "statistic": round(float(kpss_stat), 4),
                    "p_value": round(float(kpss_p), 4),
                    "lags": int(kpss_lags),
                    "is_stationary": kpss_stationary,
                }
            except Exception as e:
                logger.warning(f"KPSS test failed for {col}: {e}")
                kpss_results[col] = {"error": str(e)}
                kpss_stationary = True  # assume stationary on error

            verdict = _interpret_joint(adf_stationary, kpss_stationary)
            stationarity_verdict[col] = verdict

            # Recommend transform
            if verdict == "stationary":
                recommended_transforms[col] = "none"
            elif verdict == "unit_root":
                recommended_transforms[col] = "first_difference"
            elif verdict == "trend_stationary":
                recommended_transforms[col] = "detrend"
            else:  # difference_stationary
                recommended_transforms[col] = "first_difference"

            # Check if log transform could help (variance scaling with mean)
            if series.min() > 0:
                cv = float(series.std() / series.mean()) if series.mean() != 0 else 0.0
                if cv > 0.5 and verdict != "stationary":
                    recommended_transforms[col] += " + log_transform"

            findings.append(
                f"  {col}: {verdict} — recommend {recommended_transforms[col]}"
            )

        n_stationary = sum(1 for v in stationarity_verdict.values() if v == "stationary")
        findings.insert(
            0,
            f"Stationarity: {n_stationary}/{len(stationarity_verdict)} columns stationary."
        )

        logger.info(
            f"Stationarity: {n_stationary}/{len(stationarity_verdict)} stationary"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "adf_results": adf_results,
                "kpss_results": kpss_results,
                "stationarity_verdict": stationarity_verdict,
                "recommended_transforms": recommended_transforms,
            },
        )
