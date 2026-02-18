"""
Base types for the multi-agent analysis system.

All agents inherit BaseAgent and return AnalysisResult.
The DatasetType enum is used by DataProfilerAgent to classify the dataset
and by the OrchestratorAgent to route the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class DatasetType(str, Enum):
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    PANEL = "panel"


@dataclass
class AnalysisResult:
    agent_name: str
    findings: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base for all analysis agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        """
        Run the agent on `df`.

        Parameters
        ----------
        df:
            The dataset to analyse. Time-series agents receive a date-sorted copy.
        context:
            Accumulates outputs from all previously-run agents, keyed by agent_name.
            Read prior results via context["AgentName"]["key"].
            Do not modify context directly — the orchestrator manages it.
        """
        ...
