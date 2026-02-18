"""
DataProfilerAgent — Part I §2 Data Understanding

Runs first. Classifies the dataset as time_series, cross_sectional, or panel.
The `dataset_type` and `date_column` outputs drive the orchestrator's routing decision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base import AnalysisResult, BaseAgent, DatasetType

logger = logging.getLogger(__name__)

# Minimum fraction of values parseable as dates to accept a column as temporal
_DATE_PARSE_THRESHOLD = 0.70
# Column name fragments that suggest a datetime column
_DATE_KEYWORDS = ("date", "time", "period", "year", "month", "week", "day", "timestamp")
# If unique values / total rows exceeds this, the column is treated as an identifier and excluded
# Applies to: all categorical columns, and integer-typed numeric columns
# Float columns are exempt (continuous measurements are naturally near-unique)
_ID_CARDINALITY_RATIO = 0.95


def _filter_id_columns(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    date_col: Optional[str],
) -> tuple[List[str], List[str], List[str]]:
    """
    Remove identifier-like columns from analysis lists.

    A column is considered an identifier when its unique value ratio exceeds
    _ID_CARDINALITY_RATIO. Rules:
    - Categorical/object: always apply the ratio check.
    - Numeric integer dtypes: apply the ratio check (sequential IDs stored as int).
    - Numeric float dtypes: exempt — continuous measurements are naturally near-unique.
    - The detected date column is never excluded.

    Returns (filtered_numeric, filtered_categorical, excluded_columns).
    """
    n = max(len(df), 1)
    excluded: List[str] = []

    filtered_cat = []
    for col in categorical_cols:
        if col == date_col:
            filtered_cat.append(col)
            continue
        ratio = df[col].nunique(dropna=True) / n
        if ratio > _ID_CARDINALITY_RATIO:
            excluded.append(col)
            logger.info(f"DataProfiler: excluded high-cardinality column '{col}' (ratio={ratio:.2f})")
        else:
            filtered_cat.append(col)

    filtered_num = []
    for col in numeric_cols:
        if col == date_col:
            filtered_num.append(col)
            continue
        # Only apply ratio filter to integer-typed columns
        if pd.api.types.is_integer_dtype(df[col]):
            ratio = df[col].nunique(dropna=True) / n
            if ratio > _ID_CARDINALITY_RATIO:
                excluded.append(col)
                logger.info(f"DataProfiler: excluded integer ID column '{col}' (ratio={ratio:.2f})")
                continue
        filtered_num.append(col)

    return filtered_num, filtered_cat, excluded


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the name of the most-likely datetime column, or None.

    Preference order:
    1. Columns whose name contains a date keyword AND parse successfully.
    2. Object columns that parse as dates at > _DATE_PARSE_THRESHOLD ratio.
    """
    candidates: List[str] = [
        c for c in df.columns if any(kw in c.lower() for kw in _DATE_KEYWORDS)
    ]
    # Add object columns not already in candidates as secondary candidates
    for c in df.select_dtypes(include=["object"]).columns:
        if c not in candidates:
            candidates.append(c)

    best_col: Optional[str] = None
    best_ratio = 0.0
    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_col and best_ratio >= _DATE_PARSE_THRESHOLD:
        return best_col
    return None


def _classify_dataset(df: pd.DataFrame, date_col: Optional[str]) -> DatasetType:
    """
    Classify the dataset type using the detected date column.

    Rules:
    - No date column → CROSS_SECTIONAL
    - Date column exists + unique values == total rows → TIME_SERIES
    - Date column exists + duplicate dates → PANEL (entity × time)
    - Date column with low cardinality (≤5 unique) → treated as CROSS_SECTIONAL category
    """
    if date_col is None:
        return DatasetType.CROSS_SECTIONAL

    parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
    n_unique = parsed.nunique()

    if n_unique <= 5:
        # Treat year/quarter columns as categorical grouping, not time series
        return DatasetType.CROSS_SECTIONAL

    if n_unique == len(parsed):
        return DatasetType.TIME_SERIES

    # Duplicate dates → panel structure
    return DatasetType.PANEL


class DataProfilerAgent(BaseAgent):
    """Part I §2 — Data Understanding: structure, types, granularity, dataset classification."""

    name: str = "DataProfiler"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols: List[str] = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols: List[str] = df.select_dtypes(
            include=["datetime64"]
        ).columns.tolist()
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024**2))

        date_col = _detect_date_column(df)
        dataset_type = _classify_dataset(df, date_col)

        # Remove identifier-like columns — they carry no analytical value
        numeric_cols, categorical_cols, excluded_cols = _filter_id_columns(
            df, numeric_cols, categorical_cols, date_col
        )

        missing_pct = (df.isna().mean() * 100).round(2)
        top_missing: Dict[str, float] = (
            missing_pct[missing_pct > 0]
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        findings_lines = [
            f"Rows: {len(df):,}  |  Columns: {df.shape[1]}",
            f"Numeric: {len(numeric_cols)}  |  Categorical: {len(categorical_cols)}  |  Datetime: {len(datetime_cols)}",
            f"Memory: {mem_mb:.2f} MB",
            f"Dataset type: {dataset_type.value}",
        ]
        if date_col:
            findings_lines.append(f"Detected date column: {date_col}")
        if excluded_cols:
            findings_lines.append(
                f"Excluded {len(excluded_cols)} identifier-like column(s) from analysis: "
                + ", ".join(excluded_cols)
            )
        if top_missing:
            findings_lines.append("Columns with missing values:")
            for col, pct in top_missing.items():
                findings_lines.append(f"  - {col}: {pct:.1f}% missing")
        else:
            findings_lines.append("No missing values detected.")

        logger.info(
            f"DataProfiler: {dataset_type.value}, date_col={date_col}, "
            f"shape={df.shape}, excluded={excluded_cols}"
        )

        return AnalysisResult(
            agent_name=self.name,
            findings="\n".join(findings_lines),
            data={
                "shape": df.shape,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols,
                "excluded_columns": excluded_cols,
                "dataset_type": dataset_type.value,
                "date_column": date_col,
                "memory_mb": round(mem_mb, 2),
                "top_missing": top_missing,
            },
        )
