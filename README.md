# Mortgage Credit Risk Modelling

A configuration-driven **point-in-time behavioural mortgage credit-risk modelling framework** built on Freddie Mac Single-Family Loan-Level data and PySpark.

The project focuses on predicting a serious-delinquency outcome from information available at a defined observation point in a mortgage's life, while maintaining a strict temporal boundary between predictors and future outcomes.

> **Current active modelling path:** GAM with configurable nonlinear spline effects, controlled interactions, and macroeconomic features.

## What the model predicts

The active target is:

```text
future_90dpd
```

For an eligible loan observed at age `t`, the target is positive when, during `t + 1` through `t + 12`, either:

- numeric delinquency status reaches `>= 3` (approximately 90+ DPD), or
- the loan reaches REO acquisition status (`RA`).

This is a **PD-like serious-delinquency outcome**. It is not an LGD, EAD, realised-loss, or expected-loss model.

## Current configuration

| Setting | Active configuration |
|---|---|
| Modelling approach | Behavioural / point-in-time |
| Engine | PySpark |
| Observation ages | 2, 4, 6, 8, 10, 12 months |
| Prediction horizon | 12 months |
| Training vintages | 2015–2020 |
| Test vintage | 2021 |
| Out-of-time vintage | 2022 |
| Active algorithm | GAM |
| GAM spline degree | 3 |
| GAM knots | 6 quantile-based knots |
| GAM interactions | Enabled |
| Macro features | Enabled |
| SHAP | Disabled in active evaluation configuration |
| Model version | `gam_macro_weighted_semi_quarterly` |

The checked-in configuration currently skips both raw ingestion and preprocessing and expects canonical/intermediate model-input data to already exist.

## Leakage boundary

Each model observation has grain:

```text
loan_id × observation_age
```

Predictors may use information available **at or before** the observation age.

The forward outcome uses:

```text
observation_age + 1 ... observation_age + 12
```

Conceptually:

```text
Performance history through t  ──► FEATURES
Performance after t            ──► TARGET ONLY
```

Loans with prior serious delinquency or an invalid observation state are excluded according to the configured eligibility rules.

Voluntary payoff/maturity (`ZBC = 01`) is treated as a non-event when no prior serious event has occurred; other incomplete early exits are handled according to the target construction rules.

## Feature contract

The behavioural configuration includes:

### Origination risk and structure
- credit score
- original DTI
- original LTV / CLTV
- original UPB
- mortgage insurance percentage
- borrower count
- loan/programme characteristics

### Point-in-time loan state
- current UPB
- current interest rate
- estimated LTV
- calculated loan age
- remaining months to legal maturity
- balance composition

### Lifetime behavioural history
- maximum DPD to date
- delinquency months to date
- months since last delinquency
- historical modification/payment-deferral/assistance/disaster indicators

### Recent behavioural windows
- 3-month and 12-month delinquency counts
- maximum recent DPD
- modification/payment-deferral/assistance/disaster counts
- rate-step activity

### Delinquency trajectory
- current delinquency streak
- maximum recent delinquency streak
- months since recent 30/60 DPD
- DPD trend
- DPD acceleration
- delinquency-intensity change
- DPD-severity change
- delinquency episode count
- relapse-after-current indicator

### Loan trajectory
- UPB percentage change from origination
- interest-rate change from origination

### Macroeconomic variables
- unemployment rate
- 30-year mortgage rate
- purchase-only HPI
- Federal Funds rate

### Categorical variables
Property type, occupancy, loan purpose, channel, programme indicators, state, modification/payment-deferral indicators, disaster delinquency, and borrower-assistance-plan indicators.

The source of truth for the feature contract is `config/parameters/behavioral.yml`.

## GAM architecture

The active GAM uses a **linear-by-default** representation.

Selected numerical variables are eligible for spline transformation. Spline knots are learned from training data using quantiles. If a feature cannot support the requested spline basis because of insufficient valid/unique values, the implementation falls back to a linear representation.

The active spline configuration is:

```yaml
degree: 3
num_knots: 6
method: quantile
```

The GAM also supports explicitly configured interactions.

```text
Main effects
    ├── linear numerical effects
    ├── spline numerical effects
    └── categorical effects

Interactions
    ├── numeric × numeric
    └── numeric × categorical

                 ↓
          VectorAssembler
                 ↓
        Logistic regression
```

Interactions are deliberately controlled rather than generated exhaustively.

## Temporal evaluation

The active configuration separates:

```text
2015–2020  → training population with configured validation split
2021       → test
2022       → out-of-time evaluation
```

The configured validation split is chronological/yearly within the training population.

The 2022 OOT population is held out from model and preprocessing fitting.

## Pipeline

```text
Canonical origination + performance data
              ↓
        loan-month master
              ↓
     behavioural risk sets
              ↓
 leakage-safe feature/target construction
              ↓
 chronological train / validation / test / OOT
              ↓
     training-only preprocessing
              ↓
          GAM training
              ↓
      persisted model artifacts
              ↓
 evaluation + threshold analysis
```

Run from the project root:

```python
from pathlib import Path
from credit_risk import run_pipeline

run_pipeline(Path("."))
```

Review `config/parameters/base.yml` and `config/parameters/behavioral.yml` before execution.

## Outputs

The pipeline writes versioned artefacts under `data/`, including:

- model-input Parquet data
- train/validation/test/OOT populations
- fitted model and preprocessing artefacts
- configuration and training metadata
- evaluation metrics and reports
- risk-decile analysis
- calibration tables and charts
- threshold-selection outputs
- configured data-quality artefacts

## Documentation

- [Project flow](docs/project_flow.md)
- [Modelling methodology](docs/modelling-methodology.md)
- [GAM interaction architecture](docs/gam_interactions.md)
- [Credit-risk reference guide](docs/credit-risk-reference-guide.md)
- [V1 → V2 detailed rationale](docs/V1_to_V2_detailed_rationale.md)
- [Experimental trajectory-conditioned PD idea](docs/potentially%20in%20future/trajectory_conditioned_dynamic_mortgage_pd_idea.md)

## Research status

This repository is a research/model-development framework. Discrimination, calibration, temporal stability, fairness, monitoring, independent validation, and production governance would be required before operational credit-decisioning use.
