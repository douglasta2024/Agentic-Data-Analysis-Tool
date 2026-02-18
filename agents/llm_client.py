"""
GeminiClient — LLM synthesis utility (not a BaseAgent).

Accepts the API key as a constructor argument — never reads os.getenv internally.
The key comes from st.session_state.api_key, supplied by the UI.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

from agents.base import AnalysisResult

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.0-flash"
_MAX_FINDINGS_CHARS = 8000  # truncate very long agent findings for the prompt


class GeminiClient:
    """Wrapper around the Gemini API for dual-perspective analysis synthesis."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty.")
        try:
            import google.genai as genai
            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini insights. "
                "Run: pip install google-genai"
            ) from exc

    def summarize(
        self,
        df: pd.DataFrame,
        results: List[AnalysisResult],
    ) -> Tuple[str, str]:
        """
        Generate dual-perspective insights.

        Returns
        -------
        (data_scientist_text, business_analyst_text)
        Both are empty strings on any error.
        """
        findings_text = "\n\n".join(
            f"=== {r.agent_name} ===\n{r.findings}"
            for r in results
            if not r.error
        )
        # Truncate to avoid token limit issues
        if len(findings_text) > _MAX_FINDINGS_CHARS:
            findings_text = findings_text[:_MAX_FINDINGS_CHARS] + "\n[...truncated]"

        prompt = (
            f"You are analyzing a dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.\n"
            f"Columns: {', '.join(df.columns.tolist()[:20])}\n\n"
            f"Agent findings:\n{findings_text}\n\n"
            "Provide two sections clearly marked with these exact headers:\n\n"
            "[DATA_SCIENTIST]\n"
            "Write 3-5 technical findings a data scientist would care about. "
            "Use precise language. Reference statistical patterns, data quality issues, "
            "distributional properties, correlation strengths, stationarity, and trend significance. "
            "Be specific with numbers where available.\n\n"
            "[BUSINESS_ANALYST]\n"
            "Write 2-3 plain-English takeaways a business analyst would care about, "
            "then give 1 recommended action. No statistical jargon. "
            "Focus on what changed, what stands out, and what to do about it."
        )

        try:
            response = self._client.models.generate_content(
                model=_MODEL,
                contents=prompt,
            )
            text: str = response.text

            ds_section = ""
            ba_section = ""
            if "[DATA_SCIENTIST]" in text and "[BUSINESS_ANALYST]" in text:
                ds_section = (
                    text.split("[DATA_SCIENTIST]")[1]
                    .split("[BUSINESS_ANALYST]")[0]
                    .strip()
                )
                ba_section = text.split("[BUSINESS_ANALYST]")[1].strip()
            else:
                ba_section = text.strip()

            logger.info("GeminiClient: summary generated successfully.")
            return ds_section, ba_section

        except Exception as e:
            logger.error(f"GeminiClient API error: {e}")
            return "", ""
