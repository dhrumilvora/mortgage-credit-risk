# Mortgage Credit Risk Prediction

A configuration-driven Python framework for building, training, and evaluating an origination-time mortgage credit-risk model from Freddie Mac Single-Family Loan-Level data.

The baseline model estimates the probability that a mortgage reaches 90+ days past due (DPD), or REO acquisition, during its first 24 months of observable performance history.

## What is implemented

- Ingestion and schema validation for Freddie Mac origination and monthly-performance files.
- Canonical Parquet datasets and vintage-specific loan-level modelling inputs.
- Leakage-safe target construction: performance data defines the outcome but is not used as an origination-time predictor.
- Sentinel-value normalization, a DTI-missingness indicator, and configurable feature engineering (currently credit-score bands).
- Multi-vintage development and out-of-time (OOT) population loading.
- Reproducible, stratified development/validation splitting.
- Train-only fitted preprocessing: median imputation for numeric fields, constant imputation plus one-hot encoding for categorical fields.
- Logistic-regression training, versioned model/preprocessor artifacts, and training metadata.
- Validation and OOT evaluation with JSON, Excel, and chart outputs.

## Target and cohort

`ever_90dpd_24m` is positive when, within the first 24 months of performance history, a loan has either:

- a numeric delinquency status of at least `3` (approximately 90+ DPD); or
- an `RA` (REO acquisition) status.

To make the outcome observable, a loan must first appear at loan age 0 or 1. Loans with an unobserved early period, or an unobservable early termination other than a voluntary payoff, are excluded. A voluntary payoff (`ZBC = 01`) without prior serious delinquency is treated as a non-event.

This is a serious-delinquency / credit-deterioration target, not a realized-loss model. It is related to probability of default (PD), not directly to loss given default (LGD) or exposure at default (EAD).

## Pipeline

```text
Raw Freddie Mac data
  -> canonical origination and performance Parquet
  -> origination/performance preprocessing
  -> loan-month master dataset
  -> 24-month target and vintage-specific model input
  -> development/OOT loading and development split
  -> fitted preprocessing and logistic-regression training
  -> model artifacts and validation/OOT evaluation reports
```

The package entry point runs every stage; each stage can be skipped through configuration:

```python
from pathlib import Path
from credit_risk import run_pipeline

run_pipeline(Path("."))
```

Before a full run, configure the data locations and enabled stages in `config/catalog/base.yml` and `config/parameters/base.yml`. The default configuration deliberately skips ingestion, modelling, and evaluation, so a run can reuse existing artifacts or execute only selected stages.

## Baseline predictors

The predictor set is controlled in `parameters.modelling.features`. It includes borrower credit and capacity, leverage, mortgage structure, program, channel, and state fields, plus configured engineered features. The baseline configuration uses credit-score bands in place of raw credit score.

Performance-history variables are reserved for target construction. Future loan-month models may add point-in-time behavioural features, provided they are available as of the observation date.

## Outputs

Configured paths are rooted at `data/` by default.

| Output | Purpose |
|---|---|
| `03_processed/.../model-input.parquet` | One loan per row with origination-time features and target |
| `04_model_split/*.parquet` | Persisted training, validation, and OOT populations |
| `05_artifacts/<version>/<algorithm>/` | Model, fitted preprocessor, training metadata, and training configuration |
| `05_artifacts/model_evaluation/<version>/<algorithm>/<dataset>/` | Evaluation JSON, Excel workbook, and ROC, KS, decile, and calibration charts |
| `06_reporting/data_quality/pipeline_qc.xlsx` | Data-quality workbook |

## Evaluation

For enabled validation and OOT datasets, the framework produces classification metrics, ROC-AUC, PR-AUC, KS, Brier score, log loss, confusion matrices, risk deciles, calibration tables, and diagnostic charts. The classification threshold, risk deciles, calibration bins, model version, and evaluation mode are all configuration-driven.

## Project documentation

- [Project flow](docs/project_flow.md) — architecture, stages, configuration, and artifacts.
- [Modelling methodology](docs/modelling-methodology.md) — target, cohort, features, preprocessing, training, and evaluation decisions.
- [Credit-risk reference guide](docs/credit-risk-reference-guide.md) — terminology and Freddie Mac field guidance.

## Current limitations

The current trained algorithm is logistic regression. Model calibration, challenger models, explainability, stability analysis, and production monitoring remain future work. Data and model outputs are local project artifacts and should be independently validated before any credit decisioning use.
