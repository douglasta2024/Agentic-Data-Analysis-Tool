"""
agents — Multi-agent data analysis system.

Public surface used by app.py:
    from agents import OrchestratorAgent, AnalysisResult
    from agents import GeminiClient  # optional LLM synthesis
"""

from agents.base import AnalysisResult, BaseAgent, DatasetType
from agents.llm_client import GeminiClient
from agents.orchestrator import OrchestratorAgent

__all__ = [
    "AnalysisResult",
    "BaseAgent",
    "DatasetType",
    "GeminiClient",
    "OrchestratorAgent",
]
