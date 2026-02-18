"""
CausalFlagAgent — Part I §10 Causal vs Predictive Clarification

Lightweight rule-based agent. Flags strong correlations with the reminder that
correlation ≠ causation, and checks for potential A/B test structure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_STRONG_CORR_THRESHOLD = 0.70
_MODERATE_CORR_THRESHOLD = 0.40


class CausalFlagAgent(BaseAgent):
    """Part I §10 — Causal vs Predictive: flag correlation pairs, detect A/B signals."""

    name: str = "CausalFlag"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        eda_data = context.get("EDA", {})
        top_correlations: List[Dict[str, Any]] = eda_data.get("top_correlations", [])

        # Pull filtered column lists from DataProfiler to exclude identifier-like columns
        profiler_data = context.get("DataProfiler", {})
        analysis_cols: List[str] = (
            profiler_data.get("numeric_columns", [])
            + profiler_data.get("categorical_columns", [])
        )
        if not analysis_cols:
            analysis_cols = df.columns.tolist()

        findings: List[str] = []
        causal_warnings: List[Dict[str, Any]] = []

        # ── Flag strong correlations ─────────────────────────────────────────
        for pair in top_correlations:
            r = abs(pair["r"])
            if r >= _STRONG_CORR_THRESHOLD:
                warning = {
                    "a": pair["a"],
                    "b": pair["b"],
                    "r": pair["r"],
                    "level": "strong",
                    "message": (
                        f"{pair['a']} and {pair['b']} are strongly correlated (r={pair['r']:.2f}). "
                        "This does not imply causation — a confounder or reversed causality may explain this."
                    ),
                }
                causal_warnings.append(warning)
                findings.append(f"  STRONG: {pair['a']} vs {pair['b']} r={pair['r']:.2f}")
            elif r >= _MODERATE_CORR_THRESHOLD:
                warning = {
                    "a": pair["a"],
                    "b": pair["b"],
                    "r": pair["r"],
                    "level": "moderate",
                    "message": (
                        f"{pair['a']} and {pair['b']} have a moderate correlation (r={pair['r']:.2f}). "
                        "Interpret directionally, not causally."
                    ),
                }
                causal_warnings.append(warning)

        if causal_warnings:
            n_strong = sum(1 for w in causal_warnings if w["level"] == "strong")
            findings.insert(0, f"Causal interpretation warnings: {n_strong} strong, "
                              f"{len(causal_warnings) - n_strong} moderate correlation pairs.")
        else:
            findings.append("No strong correlations detected — causal risk is low.")

        # ── Detect binary treatment column (A/B test signal) ─────────────────
        has_binary_treatment = False
        potential_treatment_col: str | None = None
        for col in analysis_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                # Check for classic A/B, 0/1, control/treatment patterns
                vals_lower = set(str(v).lower() for v in unique_vals)
                is_ab = vals_lower <= {"a", "b", "0", "1", "control", "treatment", "yes", "no",
                                       "true", "false"}
                if is_ab:
                    has_binary_treatment = True
                    potential_treatment_col = col
                    findings.append(
                        f"Binary column detected: '{col}' ({list(unique_vals)}) — "
                        "possible A/B test or treatment/control structure. "
                        "If causal inference is the goal, use difference-in-differences or matching, "
                        "not plain correlation."
                    )
                    break

        logger.info(
            f"CausalFlag: {len(causal_warnings)} warnings, "
            f"binary_treatment={has_binary_treatment}"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings) if findings else "No causal concerns flagged.",
            data={
                "causal_warnings": causal_warnings,
                "has_binary_treatment_column": has_binary_treatment,
                "potential_treatment_col": potential_treatment_col,
            },
        )
