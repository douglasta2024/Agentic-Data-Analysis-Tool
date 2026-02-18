"""
EDAAgent — Part I §4 Exploratory Data Analysis

Univariate: distributions, skewness, kurtosis, cardinality.
Bivariate: Pearson + Spearman correlations, mutual information pairs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_TOP_CORR_PAIRS = 10
_MI_MAX_COLS = 15       # limit mutual info to avoid slow computation on wide datasets


class EDAAgent(BaseAgent):
    """Part I §4 — EDA: distributions, correlations, mutual information."""

    name: str = "EDA"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        # Pull filtered column lists from DataProfiler to exclude identifier-like columns
        profiler_data = context.get("DataProfiler", {})
        numeric_col_list: List[str] = profiler_data.get(
            "numeric_columns", df.select_dtypes(include=[np.number]).columns.tolist()
        )
        categorical_col_list: List[str] = profiler_data.get(
            "categorical_columns",
            df.select_dtypes(include=["object", "category"]).columns.tolist(),
        )
        numeric = df[numeric_col_list] if numeric_col_list else df.select_dtypes(include=[np.number])
        findings: List[str] = []

        # ── Univariate stats ─────────────────────────────────────────────────
        univariate_stats: Dict[str, Any] = {}
        if not numeric.empty:
            desc = numeric.describe().T
            skew = numeric.skew(numeric_only=True)
            kurt = numeric.kurtosis(numeric_only=True)

            most_skewed = skew.abs().sort_values(ascending=False).head(5)
            findings.append("Most skewed columns:")
            for col in most_skewed.index:
                findings.append(f"  - {col}: skew={skew[col]:.2f}, kurt={kurt[col]:.2f}")

            univariate_stats = {
                col: {
                    "mean": round(float(desc.loc[col, "mean"]), 4) if col in desc.index else None,
                    "std": round(float(desc.loc[col, "std"]), 4) if col in desc.index else None,
                    "min": round(float(desc.loc[col, "min"]), 4) if col in desc.index else None,
                    "max": round(float(desc.loc[col, "max"]), 4) if col in desc.index else None,
                    "skewness": round(float(skew[col]), 4) if col in skew.index else None,
                    "kurtosis": round(float(kurt[col]), 4) if col in kurt.index else None,
                }
                for col in numeric.columns
            }

        # ── Pearson correlation matrix ───────────────────────────────────────
        correlation_matrix: Dict[str, Any] = {}
        top_correlations: List[Dict[str, Any]] = []
        spearman_matrix: Dict[str, Any] = {}

        if numeric.shape[1] >= 2:
            pearson = numeric.corr(method="pearson", numeric_only=True)
            spearman = numeric.corr(method="spearman", numeric_only=True)

            correlation_matrix = pearson.to_dict()
            spearman_matrix = spearman.to_dict()

            pairs: List[Tuple[str, str, float]] = []
            cols = list(pearson.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], float(pearson.iloc[i, j])))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            top_correlations = [
                {"a": a, "b": b, "r": round(v, 3)}
                for a, b, v in pairs[:_TOP_CORR_PAIRS]
            ]
            findings.append("Top absolute Pearson correlations:")
            for item in top_correlations[:5]:
                findings.append(f"  - {item['a']} vs {item['b']}: r={item['r']:.3f}")

        # ── Mutual information ───────────────────────────────────────────────
        mi_pairs: List[Dict[str, Any]] = []
        if numeric.shape[1] >= 2:
            try:
                from sklearn.feature_selection import mutual_info_regression

                mi_cols = numeric.columns.tolist()[:_MI_MAX_COLS]
                mi_df = numeric[mi_cols].dropna()
                if len(mi_df) >= 10 and len(mi_cols) >= 2:
                    mi_results: List[Tuple[str, str, float]] = []
                    for target_col in mi_cols:
                        feature_cols = [c for c in mi_cols if c != target_col]
                        X = mi_df[feature_cols].values
                        y = mi_df[target_col].values
                        scores = mutual_info_regression(X, y, random_state=42)
                        for feat_col, score in zip(feature_cols, scores):
                            mi_results.append((feat_col, target_col, float(score)))

                    mi_results.sort(key=lambda x: x[2], reverse=True)
                    seen: set = set()
                    for f, t, s in mi_results:
                        key = tuple(sorted([f, t]))
                        if key not in seen:
                            seen.add(key)
                            mi_pairs.append({"a": f, "b": t, "mi": round(s, 4)})
                        if len(mi_pairs) >= _TOP_CORR_PAIRS:
                            break
            except ImportError:
                logger.warning("scikit-learn not available; skipping mutual information.")

        # ── Categorical cardinality (analysis columns only) ──────────────────
        cat_cardinality: Dict[str, int] = {
            col: int(df[col].nunique(dropna=True))
            for col in categorical_col_list
            if col in df.columns
        }
        if cat_cardinality:
            top_cat = sorted(cat_cardinality.items(), key=lambda x: x[1])[:5]
            findings.append("Categorical column cardinalities (lowest):")
            for col, n in top_cat:
                findings.append(f"  - {col}: {n} unique values")

        logger.info(
            f"EDA: {len(top_correlations)} correlation pairs, "
            f"{len(mi_pairs)} MI pairs, {len(univariate_stats)} numeric cols"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "univariate_stats": univariate_stats,
                "correlation_matrix": correlation_matrix,
                "spearman_matrix": spearman_matrix,
                "top_correlations": top_correlations,
                "mutual_info_pairs": mi_pairs,
                "categorical_cardinality": cat_cardinality,
            },
        )
