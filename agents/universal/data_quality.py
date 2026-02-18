"""
DataQualityAgent — Part I §3 Data Quality Assessment

Checks: missing values, duplicates, constant columns, high-cardinality categoricals,
IQR outlier flags, and inconsistent string encodings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent

logger = logging.getLogger(__name__)

_HIGH_MISSING_THRESHOLD = 40.0   # % missing to flag a column
_HIGH_CARDINALITY_RATIO = 0.05   # unique / total > this → high-cardinality


class DataQualityAgent(BaseAgent):
    """Part I §3 — Data Quality: missingness, duplicates, outliers, encoding issues."""

    name: str = "DataQuality"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        findings: List[str] = []

        # Pull filtered column lists from DataProfiler to exclude identifier-like columns
        profiler_data = context.get("DataProfiler", {})
        numeric_cols: List[str] = profiler_data.get(
            "numeric_columns", df.select_dtypes(include=[np.number]).columns.tolist()
        )
        categorical_cols: List[str] = profiler_data.get(
            "categorical_columns",
            df.select_dtypes(include=["object", "category"]).columns.tolist(),
        )

        # ── Duplicates ──────────────────────────────────────────────────────
        duplicate_rows = int(df.duplicated().sum())
        findings.append(f"Duplicate rows: {duplicate_rows:,}")

        # ── Missing values ───────────────────────────────────────────────────
        missing_pct = df.isna().mean() * 100
        high_missing_cols = missing_pct[missing_pct > _HIGH_MISSING_THRESHOLD]
        high_missing_count = int(len(high_missing_cols))
        findings.append(f"Columns with >{_HIGH_MISSING_THRESHOLD:.0f}% missing: {high_missing_count}")
        if not high_missing_cols.empty:
            for col, pct in high_missing_cols.sort_values(ascending=False).items():
                findings.append(f"  - {col}: {pct:.1f}%")

        # ── Constant columns ─────────────────────────────────────────────────
        constant_cols = [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ]
        findings.append(f"Constant columns: {len(constant_cols)}")
        if constant_cols:
            findings.extend([f"  - {c}" for c in constant_cols[:6]])

        # ── High-cardinality categoricals (within analysis columns only) ─────
        high_cardinality: List[Tuple[str, int]] = []
        for col in categorical_cols:
            n_unique = int(df[col].nunique(dropna=True))
            threshold = max(50, len(df) * _HIGH_CARDINALITY_RATIO)
            if n_unique > threshold:
                high_cardinality.append((col, n_unique))
        if high_cardinality:
            findings.append("High-cardinality categorical columns:")
            for col, n in high_cardinality[:6]:
                findings.append(f"  - {col}: {n:,} unique values")

        # ── IQR outlier flags (analysis columns only) ─────────────────────────
        outlier_summary: Dict[str, Dict[str, Any]] = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lo) | (series > hi)).sum())
            if n_out > 0:
                outlier_summary[col] = {
                    "count": n_out,
                    "pct": round(n_out / len(series) * 100, 2),
                    "lower_bound": round(lo, 4),
                    "upper_bound": round(hi, 4),
                }
        if outlier_summary:
            top = sorted(outlier_summary.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            findings.append("Columns with IQR outliers (top 5):")
            for col, info in top:
                findings.append(f"  - {col}: {info['count']:,} outliers ({info['pct']:.1f}%)")

        # ── Inconsistent string encodings (analysis columns only) ─────────────
        inconsistent: List[str] = []
        object_cols = [c for c in categorical_cols if c in df.columns and df[c].dtype == object]
        for col in object_cols:
            vals = df[col].dropna().astype(str)
            lowered = set(vals.str.lower().unique())
            original = set(vals.unique())
            if len(original) > len(lowered):
                inconsistent.append(col)
        if inconsistent:
            findings.append(f"Possible case inconsistencies in: {', '.join(inconsistent[:5])}")

        logger.info(
            f"DataQuality: {duplicate_rows} dupes, {high_missing_count} high-missing cols, "
            f"{len(outlier_summary)} cols with outliers"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings),
            data={
                "duplicate_rows": duplicate_rows,
                "high_missing_columns": high_missing_cols.round(2).to_dict(),
                "high_missing_columns_count": high_missing_count,
                "constant_columns": constant_cols,
                "high_cardinality": [{"column": c, "unique_values": n} for c, n in high_cardinality],
                "outlier_summary": outlier_summary,
                "inconsistent_encodings": inconsistent,
            },
        )
