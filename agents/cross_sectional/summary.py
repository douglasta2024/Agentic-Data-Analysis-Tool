"""
CrossSectionalSummaryAgent — Part I §4 Segment-Level Comparisons

Groups numeric columns by categorical columns, computes segment statistics,
and uses one-way ANOVA to identify which categorical explains the most variance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_MAX_CATEGORIES = 20   # skip high-cardinality columns as grouping keys
_TOP_SEGMENTS = 5      # top/bottom segments to surface in findings


class CrossSectionalSummaryAgent(BaseAgent):
    """Part I §4 — Segment comparisons: group stats, ANOVA, top/bottom performers."""

    name: str = "CrossSectionalSummary"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        profiler_data = context.get("DataProfiler", {})
        numeric_cols: List[str] = profiler_data.get("numeric_columns", [])
        categorical_cols: List[str] = profiler_data.get("categorical_columns", [])

        if not numeric_cols or not categorical_cols:
            return AnalysisResult(
                agent_name=self.name,
                findings="No numeric or categorical columns available for segment analysis.",
                data={},
            )

        # Filter categoricals by cardinality
        usable_cats = [
            c for c in categorical_cols
            if 2 <= df[c].nunique(dropna=True) <= _MAX_CATEGORIES
        ]
        if not usable_cats:
            return AnalysisResult(
                agent_name=self.name,
                findings="No categorical columns with 2–20 unique values found for segment grouping.",
                data={},
            )

        findings: List[str] = []
        segment_summaries: Dict[str, Any] = {}
        anova_results: Dict[str, Dict[str, float]] = {}
        top_grouping_column: Optional[str] = None
        best_f_stat = 0.0

        try:
            from scipy.stats import f_oneway
            has_scipy = True
        except ImportError:
            logger.warning("scipy not available; ANOVA skipped.")
            has_scipy = False

        for cat_col in usable_cats[:5]:  # limit for performance
            cat_stats: Dict[str, Any] = {}

            for num_col in numeric_cols[:6]:
                grouped = df.groupby(cat_col)[num_col].agg(
                    mean="mean", std="std", count="count", median="median"
                ).round(4)
                cat_stats[num_col] = grouped.to_dict(orient="index")

                # ANOVA: does this category explain variance in this numeric?
                if has_scipy:
                    groups = [
                        grp.dropna().values
                        for _, grp in df.groupby(cat_col)[num_col]
                        if len(grp.dropna()) >= 2
                    ]
                    if len(groups) >= 2:
                        try:
                            f_stat, p_val = f_oneway(*groups)
                            key = f"{cat_col}→{num_col}"
                            anova_results[key] = {
                                "f_stat": round(float(f_stat), 4),
                                "p_value": round(float(p_val), 4),
                            }
                            if float(f_stat) > best_f_stat:
                                best_f_stat = float(f_stat)
                                top_grouping_column = cat_col
                        except Exception:
                            pass

            segment_summaries[cat_col] = cat_stats

        # Build findings for the most informative grouping
        if top_grouping_column and numeric_cols:
            findings.append(
                f"Most discriminating grouping column: '{top_grouping_column}' "
                f"(highest ANOVA F-stat={best_f_stat:.2f})"
            )
            primary_num = numeric_cols[0]
            if top_grouping_column in segment_summaries:
                col_stats = segment_summaries[top_grouping_column].get(primary_num, {})
                sorted_segments = sorted(
                    col_stats.items(),
                    key=lambda x: x[1].get("mean", 0),
                    reverse=True,
                )
                top = sorted_segments[:_TOP_SEGMENTS]
                bottom = sorted_segments[-_TOP_SEGMENTS:]
                findings.append(f"Top segments by {primary_num} mean:")
                for seg, stats in top:
                    findings.append(
                        f"  - {seg}: mean={stats.get('mean', 'N/A'):.4f}, "
                        f"n={int(stats.get('count', 0)):,}"
                    )
                findings.append(f"Bottom segments by {primary_num} mean:")
                for seg, stats in bottom:
                    findings.append(
                        f"  - {seg}: mean={stats.get('mean', 'N/A'):.4f}, "
                        f"n={int(stats.get('count', 0)):,}"
                    )
        else:
            findings.append(f"Segment analysis completed for {len(usable_cats)} categorical columns.")

        logger.info(
            f"CrossSectionalSummary: {len(usable_cats)} grouping cols, "
            f"top_col={top_grouping_column}"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "segment_summaries": segment_summaries,
                "anova_results": anova_results,
                "top_grouping_column": top_grouping_column,
            },
        )
