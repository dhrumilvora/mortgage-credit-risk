# Mortgage Credit Risk — V2

V2 is a configuration-driven, point-in-time behavioural credit-risk pipeline for Freddie Mac Single-Family Loan-Level data. It builds loan observations at defined ages, trains a model using information available at each observation point, and evaluates the risk of serious delinquency over the next 12 months.

V3 is maintained separately and is not covered by this README.

## What V2 predicts

The active target is `future_90dpd_12m`.

For an eligible loan observed at age *t*, the target is positive when either of the following occurs between ages *t + 1* and *t + 12*:

- numeric delinquency status reaches `>= 3` (approximately 90+ DPD); or
- the loan reaches REO acquisition status (`RA`).

This is a PD-like serious-delinquency outcome. It is not a realised-loss, LGD, EAD, or expected-loss model.

## Current configuration

| Setting | Active value |
|---|---|
| Modelling approach | Behavioural / point-in-time |
| Processing engine | PySpark |
| Observation ages | 6 and 12 months |
| Prediction horizon | 12 months |
| Training vintages | 2015–2018 |
| Chronological validation vintages | 2019–2020 |
| Out-of-time vintages | 2021–2022 |
| Supported V2 models | Logistic regression and XGBoost |
| Active algorithm | XGBoost |
| SHAP configuration | Enabled; generated only by the Pandas evaluation path |

The active configuration skips raw-file ingestion but enables preprocessing, modelling, and evaluation. Canonical origination and performance datasets must therefore already be available in the configured intermediate paths before a run.

## Leakage boundary

Each V2 record has grain `loan_id × observation_age`. A loan is eligible only when it has an exact monthly observation at the configured age, has not already experienced serious delinquency, and has not terminated at that observation point.

```text
Origination data + performance history through age t  -> predictors
Performance from age t + 1 through age t + 12        -> target only
```

The forward outcome is observable when the full horizon is available, a serious-delinquency event occurs, or a voluntary payoff/maturity (`ZBC = 01`) occurs before the horizon ends. Other incomplete early exits without an event are excluded.

## V2 features

The feature contract is defined in [config/parameters/behavioral.yml](config/parameters/behavioral.yml). It includes:

- origination risk, leverage, and structure: credit score, DTI, LTV/CLTV, UPB, MI, and borrower count;
- origination categorical attributes: occupancy, property type, loan purpose, channel, programme flags, and state;
- point-in-time loan state: current UPB and rate, estimated LTV, loan age, and remaining term;
- current and lifetime behavioural history: current DPD, maximum DPD to date, delinquency-month count, and delinquency recency;
- recent behavioural windows: 3- and 6-month counts of 30+/60+ DPD, maximum DPD, and delinquency months; and
- post-origination trajectory: percentage UPB change from origination.

All behavioural fields are constructed using information available at or before the observation month.

## Pipeline

```text
Canonical origination + performance Parquet
  -> loan-month master dataset
  -> eligible behavioural risk sets at ages 6 and 12
  -> leakage-safe V2 features + forward 12-month target
  -> chronological training, validation, and OOT populations
  -> training-only preprocessing + configured V2 model
  -> metrics, charts, threshold search, and (Pandas-only) SHAP outputs
```

Run the pipeline from the project root:

```python
from pathlib import Path
from credit_risk import run_pipeline

run_pipeline(Path("."))
```

The notebook [notebooks/main.ipynb](notebooks/main.ipynb) provides a documented V2 runbook with configuration inspection and artifact review.

## Prerequisites

1. Install the project dependencies and activate the project environment.
2. Configure Java and PySpark for the selected local Spark setup.
3. Ensure canonical source data exists for all configured vintages, because raw ingestion is disabled.
4. Review `config/parameters/base.yml` and `config/parameters/behavioral.yml` before running.

## Outputs

All paths are rooted at `data/` by default.

| Location | Contents |
|---|---|
| `03_processed/behavioral/.../model-input.parquet` | V2 point-in-time feature/target population |
| `04_model_split/` | Training, validation, and OOT populations |
| `05_artifacts/<version>/<algorithm>/` | Model, preprocessor, configuration, and training metadata |
| `05_artifacts/<version>/<algorithm>/model_evaluation/<dataset>/` | Evaluation metrics, workbook, charts, threshold summary, and Pandas-only SHAP artifacts |
| `06_reporting/data_quality/` | Optional data-quality reporting artifacts |

Evaluation includes classification metrics, ROC-AUC, PR-AUC, KS, Brier score, log loss, risk deciles, calibration tables, charts, and configured threshold selection.

## Model support

V2 documents two supported modelling choices:

- **Logistic regression** for an interpretable baseline.
- **XGBoost** for the active nonlinear model.

Select the model with `parameters.modelling.algorithm`; the current value is `xgboost`. The PySpark XGBoost implementation is the active training path.

SHAP is currently implemented in the Pandas evaluation path. When the selected engine is PySpark, evaluation logs that SHAP is skipped even when `evaluation.shap.enabled` is true. XGBoost SHAP values, when generated through the Pandas path, are on the raw-margin (log-odds) scale.

## Documentation

- [V2 project flow](docs/project_flow.md)
- [V2 modelling methodology](docs/modelling-methodology.md)
- [Credit-risk reference guide](docs/credit-risk-reference-guide.md)
- [V1 to V2 model-evolution rationale](docs/V1_to_V2_detailed_rationale.md)

## Important note

This is a research framework. Model outputs, calibration, stability, fairness, monitoring, and operational controls require independent validation before any credit-decisioning use.
