# Mortgage Credit Risk Modelling Methodology

## Objective

The baseline model estimates the probability of serious mortgage delinquency using only information available at origination. Its binary target is `ever_90dpd_24m`: whether a loan reaches numeric delinquency status `>= 3` (approximately 90+ DPD) or REO acquisition (`RA`) within its first 24 months of observable performance history.

The target describes credit deterioration, not realized economic loss. It is best viewed as a PD-like outcome; it does not estimate LGD, EAD, or expected loss.

## Cohort and outcome observability

The first available performance observation must occur at loan age 0 or 1. Later entry cannot rule out unobserved early delinquency and is excluded. The observation window is the first 24 months of performance history.

Early terminations are handled as follows:

| Condition | Treatment |
|---|---|
| Serious delinquency observed before termination | Event |
| Voluntary payoff / maturity (`ZBC = 01`) without prior event | Non-event |
| Other termination before a fully observable outcome, without prior event | Excluded / censored |

This avoids assigning a non-event to a loan whose full outcome cannot be established. Prepayment is a competing event and could be modelled explicitly in a future survival or competing-risks extension.

## Predictor eligibility

Eligibility is assessed separately from feature engineering. A field must be available at origination, plausibly relevant, sufficiently usable, stable enough for its intended use, and free of leakage risk. Identifiers, post-origination information, constants, and unavailable variables are excluded.

The configured baseline predictors cover:

- borrower risk and capacity: credit score, DTI, number of borrowers, first-time-homebuyer flag;
- leverage and collateral: LTV, CLTV, mortgage-insurance percentage, property type, occupancy;
- mortgage structure: original UPB, interest rate, term, purpose, and channel;
- program and geography: super-conforming and HARP flags, plus property state.

`loan_id` is retained for joins and reconciliation but never used as a predictor. Geographic or institution-level fields with high cardinality require stability review before inclusion.

## Data preparation and feature engineering

The pipeline builds a master loan-month dataset only to construct the target. The final modelling population is formed by joining that target back to preprocessed origination data, which preserves the origination-time leakage boundary.

Known sentinels are normalized before modelling. The implemented origination preprocessing converts sentinel values for `original_dti`, `original_ltv`, `original_cltv`, and `first_time_homebuyer_flag`, and adds `original_dti_missing`.

Feature transformations are configuration-driven. The current configuration creates `credit_score_bins` using these left-inclusive bands:

| Band | Range |
|---|---|
| `<580` | 0–579 |
| `580-619` | 580–619 |
| `620-659` | 620–659 |
| `660-699` | 660–699 |
| `700-739` | 700–739 |
| `740-779` | 740–779 |
| `780+` | 780–<850 |

The baseline modelling configuration uses the binned credit-score feature rather than raw `credit_score`. It also creates binned LTV and CLTV representations, retains raw DTI, and adds `original_upb_log` alongside raw original UPB. A DTI binning configuration is available but is not part of the active model feature list.

## Development and OOT design

Vintage-specific model inputs are persisted independently. Configured development vintages are concatenated into one population and retain their vintage metadata. Configured OOT vintages are loaded separately and remain outside model fitting.

The development population is split into training and validation populations. With the default configuration, the validation population is 20% and the split is stratified on `ever_90dpd_24m`; the random state is fixed at 42. Both split membership and OOT inputs are persisted for reproducibility.

## Fitted preprocessing

Preprocessing is fitted on training data only, then reused to transform validation and OOT populations. This prevents validation or future-population distributions from affecting fitted transformations.

| Feature type | Current treatment |
|---|---|
| Numeric | Median imputation |
| Categorical | Fill missing with `Unknown`; one-hot encode; ignore unseen categories at scoring |
| Engineered | Pass through |

The fitted `ColumnTransformer` is saved alongside the trained model and must be used for scoring.

## Model training

The implemented baseline algorithm is scikit-learn logistic regression. Its penalty, regularization strength, solver, iteration limit, class weight, and random state are configuration-driven. The active configuration uses `class_weight: balanced`. The pipeline validates non-empty inputs and class availability before fitting.

Versioned model artifacts include:

- trained model;
- fitted preprocessor;
- training metadata (training/validation rows, transformed feature count, and training event rate);
- persisted modelling configuration.

Class weighting, threshold selection, probability calibration, and challenger models are model-governance decisions to be selected and validated rather than assumed by the target-construction process. In particular, class weighting changes the effective class prevalence seen during fitting: raw probabilities from the active balanced model should be treated as ranking scores, not calibrated PDs, until a calibration step is validated.

## Evaluation

Evaluation can score the validation population, OOT population, or both. It generates probabilities, applies the configured threshold, and reports:

- accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, log loss, and KS;
- confusion-matrix counts;
- credit-risk metrics and risk deciles;
- calibration tables using configured probability bins;
- ROC, KS, risk-decile, and calibration charts.

Results are persisted as JSON, an Excel workbook, and PNG charts. When an existing artifact is evaluated, the saved training feature contract is used to select scoring inputs. For rare events, ROC-AUC alone is insufficient; PR-AUC, decile concentration, calibration, and threshold-specific outcomes should be reviewed together.

## Limitations and next work

The baseline is an origination-time, binary 24-month model. Future work includes probability calibration, challenger algorithms, coefficient/feature explainability, stability monitoring across vintages and segments, and point-in-time loan-month models with strictly lagged behavioural variables. Any deployment should include independent validation, fair-lending review, monitoring, and controls appropriate to the intended use.
