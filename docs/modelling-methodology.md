# Mortgage Credit Risk Modelling Methodology

## Objective

V2 estimates point-in-time serious-delinquency risk. At each eligible observation age, `future_90dpd_12m` is one when a loan reaches numeric delinquency `>= 3` (about 90+ DPD) or REO acquisition (`RA`) during the following 12 months. This is a PD-like outcome, not a realised-loss, LGD, EAD, or expected-loss estimate. V3 is maintained separately and is outside this methodology.

## Risk set and outcome

The model grain is `loan_id × observation_age`; the active ages are 6 and 12 months. A loan must have an exact performance record at the observation age, no serious delinquency through that month, and no termination at that month.

The target window is `observation_age + 1` through `observation_age + 12`.

| Forward-window condition | Treatment |
|---|---|
| Serious delinquency or `RA` | Event |
| Complete horizon with no event | Non-event |
| Voluntary payoff/maturity (`ZBC = 01`) with no event | Non-event |
| Other early exit or incomplete horizon with no event | Excluded |

## Features and leakage control

Features must be available no later than the observation age. The active contract includes origination risk and capacity, leverage and collateral, mortgage structure, programme and geography, point-in-time loan state, delinquency history, and UPB trajectory.

Examples include credit score, DTI, LTV/CLTV, UPB, current UPB, current interest rate, estimated LTV, loan age, remaining term, current/max DPD, delinquency months to date, months since last delinquency, and percentage UPB change from origination. `loan_id` is used for joins and group-safe splitting, never as a predictor.

Configured sentinels are normalised before modelling: `999` for DTI/LTV/CLTV and `9` for first-time-homebuyer status are treated as missing; `original_dti_missing` is retained as a feature.

## Validation design

The default chronological split trains on 2015–2018 and validates on 2019–2020; 2021–2022 remain out of time. OOT data is never used to fit preprocessing or the model. Random behavioural splits are loan-grouped and verify that no loan appears in both partitions.

## Preprocessing and training

The preprocessor is fit only on training data and reused unchanged for validation and OOT scoring.

| Feature type | Treatment |
|---|---|
| Numeric | Median imputation |
| Categorical | `Unknown` imputation, one-hot encoding, unseen categories ignored |
| Engineered | Pass through |

V2 supports logistic regression as its baseline and XGBoost as its nonlinear model. The active configuration trains XGBoost. Versioned artifacts preserve the model, preprocessor, training metadata, and exact training configuration.

## Evaluation and explainability

Validation and OOT reports include threshold metrics, ROC-AUC, PR-AUC, KS, Brier score, log loss, confusion counts, deciles, calibration tables, and charts. Configured threshold search evaluates candidate thresholds against the selected metric.

SHAP output preserves sampled transformed features, SHAP values, global importance, and metadata when evaluation uses the Pandas engine. The current PySpark evaluation path skips SHAP. XGBoost contributions, when generated, are on the raw-margin (log-odds) scale, not calibrated-PD changes.

## Limitations

This is a horizon-specific point-in-time model, not a full survival or competing-risks model. Calibration, stability analysis, model comparison, fairness review, monitoring, and independent validation remain necessary before deployment or credit decisioning.
