# Data Science Analytical Framework

*A Structured Decision-Oriented Approach*

------------------------------------------------------------------------

# Part I --- Standard Data Science Thought Process

## 1. Problem Definition (Before Touching Data)

**Primary Question:**\
What decision will this analysis inform?

Clarify: - Target variable (if predictive) - Business objective -
Constraints (time, compute, interpretability, regulatory) - Success
metric (RMSE, AUC, lift, revenue impact, etc.) - Unit of analysis (user,
transaction, session, product, etc.)

> If the objective is unclear, downstream modeling is invalid.

------------------------------------------------------------------------

## 2. Data Understanding

### A. Data Structure

-   Shape (rows, columns)
-   Data types
-   Granularity
-   Panel vs cross-sectional vs time series
-   Hierarchical structure

### B. Data Generating Process (DGP)

-   How was the data collected?
-   Observational or experimental?
-   Sampling bias?
-   Survivorship bias?
-   Logging changes over time?
-   Missingness mechanism (MCAR, MAR, MNAR)?

> Understanding the DGP is more important than choosing an algorithm.

------------------------------------------------------------------------

## 3. Data Quality Assessment

Systematically check: - Missing values (pattern + mechanism) - Outliers
(error vs signal) - Duplicates - Inconsistent encodings - Target
leakage - Label noise - Internal consistency checks

If data quality is poor, modeling is premature.

------------------------------------------------------------------------

## 4. Exploratory Data Analysis (EDA)

EDA is structured and hypothesis-driven.

### Univariate

-   Distribution
-   Skewness / kurtosis
-   Cardinality

### Bivariate / Multivariate

-   Correlation (linear and rank-based)
-   Mutual information
-   Interaction effects
-   Segment-level comparisons

EDA should generate or test hypotheses --- not just produce plots.

------------------------------------------------------------------------

## 5. Feature Engineering

Feature creation must: - Reflect domain logic - Avoid leakage - Respect
temporal ordering - Be computed inside training folds

------------------------------------------------------------------------

## 6. Modeling Strategy

Model selection depends on: - Data size - Dimensionality -
Interpretability constraints - Overfitting risk - Latency requirements

Always: 1. Start with a baseline model 2. Compare against naive
benchmarks 3. Increase complexity only if justified

------------------------------------------------------------------------

## 7. Validation Framework

Choose based on data structure: - Random split (IID assumption valid) -
Time-based split (if temporal) - Group-based split (avoid leakage) -
Cross-validation

Never trust a single split.

------------------------------------------------------------------------

## 8. Evaluation Metrics

Metrics must match business objectives.

Examples: - RMSE → magnitude error - MAE → robustness to outliers - AUC
→ ranking quality - Precision/Recall → imbalance-sensitive - Calibration
→ probability reliability - Uplift → treatment effect - Business KPIs →
revenue, cost, churn

------------------------------------------------------------------------

## 9. Error Analysis

Inspect: - Failure cases - Subgroup performance - Distribution shift -
High-leverage observations - Segment-level degradation

------------------------------------------------------------------------

## 10. Causal vs Predictive Clarification

If causal: - Correlation is insufficient - Must define identification
strategy - Randomized experiments - Instrumental variables - Matching -
Difference-in-differences - Regression discontinuity

------------------------------------------------------------------------

## 11. Robustness & Sensitivity Checks

Test: - Feature removal - Hyperparameter variation - Alternate model
families - Different time windows - Random seeds

------------------------------------------------------------------------

## 12. Communication

Deliver: - Clear problem restatement - Data limitations - Assumptions
made - Quantified uncertainty - Business interpretation - Risk
assessment

------------------------------------------------------------------------

# Part II --- Internal Validation Mental Checklist

## Phase 1 --- Problem Integrity

-   Is the question decision-aligned?
-   Predictive vs causal clarified?

## Phase 2 --- Data Generating Process

-   Do I understand how this data was produced?
-   Are rows truly independent?
-   Is temporal integrity preserved?

## Phase 3 --- Data Quality

-   Missingness mechanism understood?
-   Outliers examined?
-   Target constructed correctly?

## Phase 4 --- Modeling Validity

-   Does this beat a naive baseline?
-   Overfitting controlled?
-   Metric aligned with business cost?

## Phase 5 --- Robustness

-   Are results stable?
-   Subgroup performance checked?
-   Feature importance sanity-checked?

## Phase 6 --- Interpretability & Risk

-   Can I explain this simply?
-   Is uncertainty quantified?
-   Business impact simulated?

------------------------------------------------------------------------

# Part III --- Time Series Analytical Framework

## 1. Respect Temporal Order

-   Never shuffle time
-   Always use forward validation
-   Avoid future information leakage

------------------------------------------------------------------------

## 2. Identify Structure

Decompose conceptually:

Y_t = Trend + Seasonality + Residual

------------------------------------------------------------------------

## 3. Trend Analysis

-   Raw time plot
-   Rolling averages
-   Structural break detection

------------------------------------------------------------------------

## 4. Seasonality Detection

-   Seasonal subseries plots
-   ACF analysis
-   Boxplots by month/day-of-week

------------------------------------------------------------------------

## 5. Stationarity Check

-   ADF test
-   KPSS test
-   Differencing if needed
-   Log transforms if variance scales with mean

------------------------------------------------------------------------

## 6. Autocorrelation Examination

-   ACF
-   PACF
-   Lag plots

------------------------------------------------------------------------

## 7. Baseline Models First

-   Naive forecast
-   Seasonal naive
-   Moving average
-   Linear trend

------------------------------------------------------------------------

## 8. Proper Validation

-   Rolling-origin evaluation
-   Expanding window validation
-   Walk-forward testing

Metrics: - MAE - RMSE - MAPE (with caution)

------------------------------------------------------------------------

## 9. Residual Diagnostics

-   Residual over time
-   Residual ACF
-   Bias detection
-   White noise verification

------------------------------------------------------------------------

## 10. Drift Monitoring

-   Forecast error trends
-   Seasonal pattern changes
-   Distribution shifts

------------------------------------------------------------------------

# Final Integrity Check

1.  What assumption, if wrong, invalidates everything?
2.  Could this be leakage?
3.  Could this be selection bias?
4.  Is this reproducible?
5.  Would this replicate next quarter?

------------------------------------------------------------------------

# Core Philosophy

Most data science failures come from invalid assumptions --- not weak
algorithms.
