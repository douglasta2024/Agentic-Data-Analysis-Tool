# Agentic Data Analysis Tool

Streamlit app that runs a small, routed agent pipeline to analyze a CSV from two perspectives:

- Data Scientist: statistical depth (quality checks, EDA, time-series diagnostics when applicable)
- Business Analyst: visual-first KPIs and plain-English takeaways (with optional Gemini-powered insights)

![Example Image](images/example.png)

## Quickstart

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV in the sidebar and click **Run Analysis**.

## Optional: Gemini AI Insights

AI insights are disabled unless you provide a Gemini API key.

- Recommended: set `GEMINI_API_KEY` in `.env` (or paste it into the UI field)
- The key is kept in Streamlit session state and is not written by the app

`.env` example:

```text
GEMINI_API_KEY=your_key_here
```

## Important Files

- `app.py`: Streamlit UI entry point (upload, settings, dual tabs, pipeline trigger, optional Gemini summary)
- `agents/`: analysis pipeline (BaseAgent + routed agents + synthesis)
- `agents/orchestrator.py`: routes Universal vs Time Series vs Cross-Sectional agents; wraps each agent in try/except
- `agents/base.py`: `BaseAgent`, `AnalysisResult`, and `DatasetType`
- `agents/llm_client.py`: `GeminiClient` integration used by the UI for dual-perspective summaries
- `data_science_framework.md`: the decision-oriented framework that the agents are designed to implement
- `requirements.txt`: Python dependencies
- `.env`: local environment variables (do not commit secrets)
- `analysis_outputs/`: generated artifacts/outputs from analyses (if your workflow writes files there)
- `visa_stock_data.csv`: example time-series dataset for validation
- `amazon_sales_dataset.csv`: example cross-sectional dataset for validation
- `ollama_test.py`: local/experimental script (not part of the Streamlit app path)
- `CLAUDE.md`: project conventions and architectural notes for AI-assisted development

## Agent Pipeline (High Level)

1. Universal agents always run first (profiling, quality, EDA, causal flags)
2. Orchestrator routes based on dataset type inferred by `DataProfilerAgent`
   - `time_series` / `panel`: sort by date column, then run trend/seasonality/stationarity/autocorrelation agents
   - `cross_sectional`: run cross-sectional summary agent
3. Synthesis runs last and produces rule-based business bullets (used as a fallback when Gemini is disabled)

## Repo Layout

```
agents/
  base.py
  orchestrator.py
  llm_client.py
  universal/
  time_series/
  cross_sectional/
  synthesis/
app.py
requirements.txt
data_science_framework.md
```
