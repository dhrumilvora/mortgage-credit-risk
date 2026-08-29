# Mortgage Credit Risk Prediction

A configuration-driven Python framework for training and evaluating mortgage credit-risk models with Freddie Mac Single-Family Loan-Level data.

This repository documentation covers **V2**, the point-in-time behavioural model. It predicts whether a loan will become seriously delinquent during the 12 months after each configured observation age. V3 is being updated separately and is not described here.

## V2 model

- Behavioural approach with PySpark preprocessing.
- Observation ages: 6 and 12 months.
- Target: `future_90dpd_12m`.
- Train vintages: 2015–2018; validation: 2019–2020; OOT: 2021–2022.
- Configured algorithm: XGBoost, with SHAP enabled.

`future_90dpd_12m` is positive if numeric delinquency reaches `>= 3` (about 90+ DPD) or REO acquisition (`RA`) occurs from the month after the observation point through the following 12 months. It is a PD-like serious-delinquency outcome, not an LGD, EAD, or realised-loss model.

## Leakage boundary

Each model row is `loan_id × observation_age`. A loan requires an exact performance observation at that age, no serious delinquency through that point, and no termination at that point. Predictors may use origination data and performance history through the observation month only; later months construct the target only.

```text
Origination + history through observation age  -> predictors
Months observation age + 1 through +12         -> future target only
```

An incomplete forward window is excluded unless a serious-delinquency event occurs or a voluntary payoff (`ZBC = 01`) establishes a non-event.

## Pipeline

```text
Raw Freddie Mac files
  -> canonical origination and performance Parquet
  -> loan-month master dataset
  -> behavioural features and forward target
  -> chronological train, validation, and OOT populations
  -> fitted preprocessing and configured model
  -> versioned artifacts, evaluation reports, and SHAP outputs
```

```python
from pathlib import Path
from credit_risk import run_pipeline

run_pipeline(Path("."))
```

Stage switches in `config/parameters/base.yml` control ingestion, preprocessing, reporting, modelling, and evaluation. The checked-in configuration skips ingestion and preprocessing, so compatible model-input data must already exist.

## Features and outputs

The behavioural contract in `config/parameters/behavioral.yml` combines origination characteristics with point-in-time loan state, strictly historical delinquency measures, and UPB trajectory. The preprocessor is fitted only on training data: numeric fields use median imputation; categorical fields use `Unknown` imputation and one-hot encoding.

| Output | Purpose |
|---|---|
| `03_processed/behavioral/.../model-input.parquet` | Point-in-time model population |
| `04_model_split/*.parquet` | Train, validation, and OOT populations |
| `05_artifacts/<version>/<algorithm>/` | Model, preprocessor, configuration, and metadata |
| `05_artifacts/<version>/<algorithm>/model_evaluation/<dataset>/` | Metrics, charts, threshold summary, and SHAP outputs |

Evaluation includes threshold metrics, ROC-AUC, PR-AUC, KS, Brier score, log loss, deciles, calibration, and diagnostic charts. Model outputs require independent validation before any credit-decisioning use.

## Documentation

- [Project flow](docs/project_flow.md)
- [Modelling methodology](docs/modelling-methodology.md)
- [Credit-risk reference guide](docs/credit-risk-reference-guide.md)
- [V1 to V2 rationale](docs/V1_to_V2_detailed_rationale.md)
