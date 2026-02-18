"""
OrchestratorAgent — Two-phase routing pipeline.

Phase 1 (Universal): DataProfilerAgent → DataQualityAgent → EDAAgent → CausalFlagAgent
Phase 2 (Routed):
  - time_series/panel: sort by date → TrendAnalysisAgent → SeasonalityAgent
                                     → StationarityAgent → AutocorrelationAgent
  - cross_sectional:  CrossSectionalSummaryAgent
Phase 3 (Always last): SynthesisAgent

Each agent is wrapped in try/except. Failure returns a placeholder AnalysisResult
and the pipeline continues. No single agent can crash the whole run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from agents.base import AnalysisResult, BaseAgent, DatasetType
from agents.cross_sectional.summary import CrossSectionalSummaryAgent
from agents.synthesis.synthesis_agent import SynthesisAgent
from agents.time_series.autocorrelation import AutocorrelationAgent
from agents.time_series.seasonality import SeasonalityAgent
from agents.time_series.stationarity import StationarityAgent
from agents.time_series.trend_analysis import TrendAnalysisAgent
from agents.universal.causal_flag import CausalFlagAgent
from agents.universal.data_profiler import DataProfilerAgent
from agents.universal.data_quality import DataQualityAgent
from agents.universal.eda import EDAAgent

logger = logging.getLogger(__name__)

_UNIVERSAL_AGENTS: List[Type[BaseAgent]] = [
    DataProfilerAgent,
    DataQualityAgent,
    EDAAgent,
    CausalFlagAgent,
]
_TIME_SERIES_AGENTS: List[Type[BaseAgent]] = [
    TrendAnalysisAgent,
    SeasonalityAgent,
    StationarityAgent,
    AutocorrelationAgent,
]
_CROSS_SECTIONAL_AGENTS: List[Type[BaseAgent]] = [
    CrossSectionalSummaryAgent,
]


class OrchestratorAgent:
    """Coordinates the full agent pipeline with dataset-type routing."""

    def analyze(
        self,
        df: pd.DataFrame,
        query: str = "Thorough analysis",
        max_trend_columns: int = 5,
    ) -> Tuple[str, List[AnalysisResult]]:
        """
        Run all agents and return (synthesis_text, all_results).

        synthesis_text is the plain-text version of Synthesis findings,
        suitable for the Business Analyst fallback when LLM is disabled.
        """
        all_results: List[AnalysisResult] = []
        context: Dict[str, Any] = {
            "query": query,
            "max_trend_columns": max_trend_columns,
        }

        # ── Phase 1: Universal ───────────────────────────────────────────────
        for AgentClass in _UNIVERSAL_AGENTS:
            result = self._run_agent(AgentClass, df, context)
            all_results.append(result)
            if not result.error:
                context[result.agent_name] = result.data

        # ── Phase 2: Routed ──────────────────────────────────────────────────
        dataset_type_str: str = context.get("DataProfiler", {}).get(
            "dataset_type", DatasetType.CROSS_SECTIONAL.value
        )
        date_col = context.get("DataProfiler", {}).get("date_column")

        if dataset_type_str in (DatasetType.TIME_SERIES.value, DatasetType.PANEL.value) and date_col:
            # Enforce Part III Rule 1: never shuffle time — sort before dispatching
            df_sorted = df.copy()
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
            df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)
            logger.info(
                f"Orchestrator: routing to TIME SERIES pipeline (date_col={date_col})"
            )
            for AgentClass in _TIME_SERIES_AGENTS:
                result = self._run_agent(AgentClass, df_sorted, context)
                all_results.append(result)
                if not result.error:
                    context[result.agent_name] = result.data
        else:
            logger.info("Orchestrator: routing to CROSS-SECTIONAL pipeline")
            for AgentClass in _CROSS_SECTIONAL_AGENTS:
                result = self._run_agent(AgentClass, df, context)
                all_results.append(result)
                if not result.error:
                    context[result.agent_name] = result.data

        # ── Phase 3: Synthesis (always last) ─────────────────────────────────
        synthesis_result = self._run_agent(SynthesisAgent, df, context)
        all_results.append(synthesis_result)

        # Build synthesis text for rule-based fallback
        synthesis_data = synthesis_result.data if not synthesis_result.error else {}
        business_bullets: List[str] = synthesis_data.get("business_bullets", [])
        synthesis_text = "\n".join(business_bullets) if business_bullets else synthesis_result.findings

        return synthesis_text, all_results

    def _run_agent(
        self,
        AgentClass: Type[BaseAgent],
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        agent = AgentClass()
        logger.info(f"Running agent: {agent.name}")
        try:
            result = agent.analyze(df, context)
            logger.info(f"Agent {agent.name} completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Agent {agent.name} failed: {e}")
            return AnalysisResult(
                agent_name=agent.name,
                findings=f"Agent failed: {e}",
                data={},
                error=str(e),
            )
