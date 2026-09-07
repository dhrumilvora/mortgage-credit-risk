# Mortgage Credit Risk Modelling Methodology

## 1. Objective

The active model estimates **point-in-time serious-delinquency risk** for mortgages.

At an eligible observation age `t`, `future_90dpd_12m` is one when a serious event occurs during the following 12 months:

```text
t + 1 ... t + 12
```

A serious event is defined as numeric delinquency status `>= 3` or REO acquisition status `RA`.

This is a PD-like outcome rather than an LGD, EAD, realised-loss, or expected-loss estimate.

---

## 2. Observation grain

The modelling grain is:

```text
loan_id × observation_age
```

The active observation ages are:

```text
2, 4, 6, 8, 10, 12 months
```

An observation must satisfy the configured eligibility conditions, including the existence of the required performance observation and absence of a prior serious event at the observation point.

---

## 3. Target construction

The forward target uses only information after the observation point.

| Condition after observation | Treatment |
|---|---|
| Numeric DPD `>= 3` | Event |
| REO acquisition `RA` | Event |
| Full 12-month horizon with no event | Non-event |
| Voluntary payoff/maturity `ZBC = 01` with no prior event | Non-event |
| Other incomplete early exit without event | Excluded |

This produces a horizon-specific behavioural target.

---

## 4. Leakage control

The central modelling rule is:

```text
FEATURES = information available through t
TARGET   = outcome observed after t
```

Features therefore use current and historical information only.

Examples include current DPD, lifetime delinquency history, recent behavioural windows, current UPB/rate, estimated LTV, loan age, and trajectory variables.

Future performance records are not allowed to enter the predictor population.

---

## 5. Feature engineering

The active feature contract is configuration-driven.

### Origination

Origination risk, affordability, leverage, collateral, loan structure, borrower count, mortgage insurance, programme and geographic variables are retained where configured.

### Current state

The model uses point-in-time state such as:

- current UPB
- current interest rate
- estimated LTV
- loan age
- remaining maturity

### Lifetime behaviour

Examples:

- maximum DPD to date
- delinquency months to date
- months since last delinquency
- historical assistance/modification/payment-deferral indicators

### Recent behaviour

Configured recent windows include 3-month and 12-month behavioural summaries.

These include delinquency counts/severity, modifications, payment deferrals, borrower assistance, disaster delinquency, and rate-step activity.

### Trajectory

The active configuration also includes:

- DPD trend
- DPD acceleration
- delinquency-intensity change
- DPD-severity change
- delinquency streaks
- episode count
- relapse behaviour
- UPB change from origination
- rate change from origination

### Macro

The active GAM configuration includes nonlinear treatment of:

- unemployment rate
- 30-year mortgage rate
- purchase-only HPI
- Federal Funds rate

---

## 6. Preprocessing

The preprocessing state is learned from training data and then reused for later populations.

Typical treatment is:

| Feature type | Treatment |
|---|---|
| Numeric | Median imputation |
| Categorical | `Unknown` imputation + categorical encoding |
| Engineered | Passed through according to feature configuration |

Freddie Mac sentinel handling is performed before modelling. Examples include treating `999` values for DTI/LTV/CLTV as missing and retaining an explicit `original_dti_missing` indicator.

Identifiers such as `loan_id` are used for joins/grouping and are not model predictors.

---

## 7. GAM specification

The active algorithm is a Generalized Additive Model implemented through the project's Spark-compatible GAM pipeline.

The conceptual model is:

```text
logit(PD)
  = intercept
  + main effects
  + configured interaction effects
```

Numerical variables are linear by default.

Selected variables are spline-eligible:

```yaml
degree: 3
num_knots: 6
```

Knots are based on training-data quantiles.

The implementation supports a safe fallback:

```text
spline requested
      ↓
sufficient valid / unique knots?
      ├── yes → spline
      └── no  → linear
```

The representation chosen during fitting is persisted and reused during scoring.

---

## 8. Interactions

Interactions are explicitly configured rather than exhaustively generated.

Supported forms are:

```text
numeric × numeric
numeric × categorical
```

Numeric × numeric interactions use tensor-product spline representations.

Numeric × categorical interactions use varying-effect representations.

Categorical × categorical interactions are not part of the controlled GAM interaction design.

The detailed design is documented in `docs/gam_interactions.md`.

---

## 9. Temporal split

The active configuration uses:

```text
Training vintages : 2015–2020
Test vintage      : 2021
OOT vintage       : 2022
```

A 20% chronological/yearly validation split is configured within the training population.

No OOT data is used to fit preprocessing, spline state, interaction state, or model parameters.

---

## 10. Model fitting

The GAM ultimately estimates logistic-regression coefficients over the assembled main-effect and interaction basis.

Conceptually:

```text
raw features
     ↓
training-fitted preparation
     ↓
spline / categorical representations
     ↓
interaction representations
     ↓
VectorAssembler
     ↓
logistic regression
```

The resulting fitted pipeline contains the transformation state required to reproduce scoring.

---

## 11. Evaluation

The evaluation framework supports:

- ROC-AUC
- PR-AUC
- KS
- Brier score
- log loss
- confusion counts
- risk deciles
- calibration analysis
- threshold search
- validation/test/OOT comparison

The active threshold search is configuration-driven rather than fixed to a single arbitrary cutoff.

Risk ranking is assessed alongside probability quality because a useful credit-risk model must be evaluated on both discrimination and calibration.

---

## 12. Model versioning

The current active model version is:

```text
gam_macro_weighted_semi_quarterly
```

Model, preprocessing, configuration, and training metadata are versioned under the configured artefact path.

---

## 13. Limitations

The active model is a **12-month point-in-time behavioural model**. It is not a full survival/competing-risks framework.

Further work should independently assess:

- temporal stability
- calibration stability
- segment stability
- model comparison
- feature redundancy
- fairness/governance
- monitoring
- independent validation
- production controls
