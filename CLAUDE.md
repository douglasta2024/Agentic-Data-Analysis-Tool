# CLAUDE.md — Agentic Data Analysis Tool

Project standards and conventions for AI-assisted development on this codebase.

---

## Common Bash Commands

```bash
# Start the Streamlit web app
streamlit run app.py

# Install all dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
agents/
    __init__.py               # Public exports: OrchestratorAgent, AnalysisResult
    base.py                   # BaseAgent (ABC), AnalysisResult (@dataclass), DatasetType enum
    orchestrator.py           # Two-phase routing orchestrator
    llm_client.py             # GeminiClient(api_key) — not a BaseAgent
    universal/                # Always-run agents (Part I framework)
        data_profiler.py      # DataProfilerAgent
        data_quality.py       # DataQualityAgent
        eda.py                # EDAAgent
        causal_flag.py        # CausalFlagAgent
    cross_sectional/          # Run when dataset_type == "cross_sectional"
        summary.py            # CrossSectionalSummaryAgent
    time_series/              # Run when dataset_type == "time_series" | "panel"
        trend_analysis.py     # TrendAnalysisAgent
        seasonality.py        # SeasonalityAgent
        stationarity.py       # StationarityAgent
        autocorrelation.py    # AutocorrelationAgent
    synthesis/                # Always-run, always last
        synthesis_agent.py    # SynthesisAgent
app.py                        # Streamlit UI entry point
requirements.txt
data_science_framework.md     # Framework governing agent design decisions
```

---

## Code Style Conventions

- **Language**: Python 3.10+
- **Style**: PEP 8
- **Line length**: 100 characters max
- **Type hints**: Required on all function signatures
- **Imports order**: stdlib → third-party → local (blank line between groups)
- **Data structures**: Use `@dataclass` for structured results — see `AnalysisResult` in `agents/base.py`
- **Agent pattern**: All agents inherit `BaseAgent` from `agents/base.py`, implement `analyze(df: pd.DataFrame, context: Dict[str, Any]) -> AnalysisResult`
- **DatasetType**: Enum in `agents/base.py` — `TIME_SERIES`, `CROSS_SECTIONAL`, `PANEL`
- **Constants**: No magic numbers — use named constants or config dicts at module level
- **No print statements**: Use the `logging` module for all diagnostic output

### Example agent signature
```python
class TrendAnalysisAgent(BaseAgent):
    name: str = "TrendAnalysis"

    def analyze(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> AnalysisResult:
        date_col: str = context["DataProfiler"]["date_column"]
        ...
```

### Context Dict Pattern
Agents communicate via a shared `context` dict accumulated by the orchestrator.
Each agent reads prior results as `context["AgentName"]["key"]` — never import or
instantiate other agent classes directly.

---

## Agent Architecture

### Two-Pipeline Routing

The orchestrator runs agents in three phases:

**Phase 1 — Universal (always):** DataProfilerAgent → DataQualityAgent → EDAAgent → CausalFlagAgent

**Phase 2 — Routed** (based on `DataProfiler.data["dataset_type"]`):
- `time_series` / `panel`: DataFrame is sorted by date column first (Part III Rule 1 — never shuffle time), then: TrendAnalysisAgent → SeasonalityAgent → StationarityAgent → AutocorrelationAgent
- `cross_sectional`: CrossSectionalSummaryAgent

**Phase 3 — Always last:** SynthesisAgent

### Framework → Agent Mapping

| Framework Section | Agent |
|---|---|
| Part I §2 Data Understanding | DataProfilerAgent |
| Part I §3 Data Quality | DataQualityAgent |
| Part I §4 EDA | EDAAgent |
| Part I §10 Causal vs Predictive | CausalFlagAgent |
| Part I §4 Segment comparisons | CrossSectionalSummaryAgent |
| Part I §4 + Part III §2-3 Trend | TrendAnalysisAgent |
| Part III §4 Seasonality | SeasonalityAgent |
| Part III §5 Stationarity | StationarityAgent |
| Part III §6 Autocorrelation | AutocorrelationAgent |
| Part I §12 Communication | SynthesisAgent |

---

## UI & Content Design Guidelines (Streamlit)

### Layout
- Use `st.tabs()` for the dual-perspective split — never radio buttons or sidebar pages
- `st.expander()` for secondary detail that would clutter the primary view
- `st.columns()` for side-by-side KPI metrics
- Keep sidebar to configuration only (upload, API key, settings, toggles)

### Data Scientist Tab (dense, technical)
- `st.dataframe()` for tabular data with full precision
- Plotly charts with axis labels, slope annotations, and r-values
- Show raw numbers alongside visualizations
- Time-series datasets: render stationarity verdicts, ACF/PACF bar charts, seasonality boxplots

### Business Analyst Tab (spacious, visual-first)
- `st.metric()` for KPI cards — always include a delta where applicable
- Plain-English chart titles (e.g., "Revenue Over Time", not "trend_Adj Close")
- Hide raw statistical tables — surface interpreted findings only
- Use `st.info()` / `st.success()` / `st.warning()` callout boxes for key takeaways

### Color Scheme
| Purpose | Hex |
|---------|-----|
| Positive / growth | `#1B9E77` (green) |
| Warning / negative | `#D95F02` (orange) |
| Neutral / default | `#7570B3` (purple) |
| Secondary accent | `#E7298A` (pink) |

### Chart Standards
- All chart titles must be plain English — no raw column name dumps
- Use `plotly.express` (`px`) for standard charts; `plotly.graph_objects` (`go`) only for custom layouts
- Set `height` explicitly on all charts for consistent layout
- Remove Plotly logo watermark: `config={"displaylogo": False}`

---

## State Management

Use `st.session_state` (aliased as `ss = st.session_state`) for all persistent data.

### Canonical Session State Keys
| Key | Type | Reset on upload? | Description |
|-----|------|-----------------|-------------|
| `ss.df` | `pd.DataFrame` | Yes | Loaded CSV |
| `ss.results` | `List[AnalysisResult]` | Yes | All agent outputs |
| `ss.synthesis` | `str` | Yes | Rule-based synthesis text |
| `ss.gemini_ds` | `str` | Yes | Gemini DS insights (or `""`) |
| `ss.gemini_ba` | `str` | Yes | Gemini BA summary (or `""`) |
| `ss.analysis_run` | `bool` | Yes | Whether pipeline has run |
| `ss.gemini_enabled` | `bool` | No | Whether LLM toggle is on |
| `ss.api_key` | `str` | **No** | Gemini API key (UI input or .env) |
| `ss.dataset_type` | `str` | Yes | `"time_series"` / `"cross_sectional"` / `"panel"` |
| `ss.date_column` | `Optional[str]` | Yes | Detected date column name |
| `ss.pipeline_log` | `List[str]` | Yes | Ordered agent names that ran successfully |
| `ss.agent_errors` | `Dict[str, str]` | Yes | `agent_name → error message` |
| `ss.stationarity_verdicts` | `Dict[str, str]` | Yes | Column → stationarity verdict |
| `ss.has_seasonality` | `bool` | Yes | True if any series classified seasonal |

### Rules
- Always guard agent pipeline calls with `if run_button:` to avoid recomputing on every rerun
- Initialize all keys at app startup with defaults to prevent `KeyError`
- Reset all analysis keys (except `api_key`) when a new file is uploaded
- Never cache agent results in `@st.cache_data` — store in session state

---

## API Key Handling

The Gemini API key can come from two sources (UI input takes priority):

```python
# In render_sidebar()
env_key = os.getenv("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Gemini API Key", value=env_key, type="password", placeholder="AIza..."
)
ss.api_key = api_key_input.strip()

if ss.api_key:
    gemini_on = st.sidebar.toggle("Enable AI Insights", value=False)
else:
    st.sidebar.caption("Enter a Gemini API key to enable AI insights.")
    gemini_on = False
```

- `GeminiClient` in `agents/llm_client.py` takes `api_key` as a constructor argument
- The client never calls `os.getenv` internally
- The key is never written to disk — lives in `st.session_state` only
- `.env` file is still supported as a convenience (pre-fills the input field)

---

## Logging

```python
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
```

### Log Levels
| Level | When to use |
|-------|------------|
| `INFO` | Agent started/completed, file loaded, pipeline routed |
| `WARNING` | Missing date column, empty result, partial data, seasonality inconclusive |
| `ERROR` | Gemini API failure, file parsing error, stationarity test failure |
| `DEBUG` | Verbose intermediate values — remove before committing |

### User-Facing Errors (Streamlit)
- `st.error("message")` — unrecoverable errors (file parsing failed, no data)
- `st.warning("message")` — degraded state (Gemini unavailable, agent failed)
- `st.info("message")` — neutral status, instructions, missing analysis conditions
- Never expose raw exception tracebacks to the user

---

## Error Handling

### File Upload
```python
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    logger.error(f"CSV load failed: {e}")
    st.error("Could not read the file. Make sure it is a valid CSV.")
    st.stop()
```

### Agent Pipeline
- Each agent call is individually wrapped in `try/except` inside the orchestrator
- On failure: log the error, return `AnalysisResult(error=str(e), data={})` and continue
- Display `ss.agent_errors` as a collapsible "Analysis Warnings" expander in the sidebar
- Pipeline never stops due to a single agent failure

### Gemini API
```python
try:
    response = client.models.generate_content(...)
except Exception as e:
    logger.error(f"Gemini API error: {e}")
    st.warning("AI insights unavailable. Showing rule-based analysis.")
    return "", ""  # SynthesisAgent fallback renders instead
```

### Top-Level Guard
Wrap each tab's render function in `try/except` to prevent one tab's crash from breaking the other.

---

## Feature Gating

AI insights require:
1. `ss.api_key` is non-empty (from UI input or pre-filled from `.env`)
2. User has toggled on "Enable AI Insights" in the sidebar

```python
def llm_available() -> bool:
    return bool(st.session_state.get("api_key", ""))
```

Time series agents gate on:
- `ss.dataset_type in ("time_series", "panel")` — checked before rendering TS sections in the DS tab

---

## Debugging

### In-App Debug Panel (remove before committing)
```python
with st.expander("Debug: Session State"):
    st.write(st.session_state)
```

### Verbose Logging
```python
logging.basicConfig(level=logging.DEBUG)  # temporary, remove before committing
```

### Test Datasets
Always validate changes against both sample datasets:
- `visa_stock_data.csv` — time-series, numeric-heavy, has date column → activates TS pipeline
- `amazon_sales_dataset.csv` — categorical-heavy, no date column → activates cross-sectional pipeline

### Common Issues
| Symptom | Likely cause |
|---------|-------------|
| App reruns unexpectedly | State mutation without `st.session_state` |
| Charts not updating | Analysis keys not reset on new upload |
| Gemini toggle missing | `ss.api_key` is empty — check UI input or `.env` |
| TS agents not running | `DataProfilerAgent` classified dataset as cross-sectional; inspect `ss.dataset_type` |
| Stationarity test error | `statsmodels` not installed — run `pip install statsmodels` |
| Mutual info import error | `scikit-learn` not installed — run `pip install scikit-learn` |

---

## Dependencies

```
streamlit
plotly
pandas
numpy
statsmodels
scikit-learn
scipy
python-dotenv
google-genai
```
