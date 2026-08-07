# Mortgage Credit Risk Modelling Methodology

## Baseline Modelling Objective

The baseline model estimates, using information available at mortgage
origination, whether a loan will experience serious delinquency within
its first 24 months.

The repository currently implements:

- canonical dataset construction,
- data-quality reporting,
- loan-level modelling-dataset construction,
- configurable multi-vintage development-population loading,
- stratified training/validation splitting,
- persistence of training and validation populations,
- and integration of the modelling-data preparation stage into the
  end-to-end pipeline.

Fitted preprocessing, model fitting, model validation, calibration,
explainability, and independent out-of-time evaluation remain subsequent
modelling phases.

The baseline target is:

`ever_90dpd_24m`

A positive event is defined as:

- numeric Current Loan Delinquency Status >= 03, representing
  90+ Days Past Due (DPD), or
- Real Estate Owned (REO) Acquisition status (`RA`)

within the 24-month performance window.

The target is intentionally described as a serious-delinquency target
rather than a generic "default" target.


## Performance Window

The baseline outcome horizon is 24 months from the beginning of the
loan's performance history.

Alternative horizons such as 12 and 36 months may be evaluated in
future model versions.


## Cohort Eligibility

Loans must have sufficiently complete early performance history for the
24-month outcome to be observable.

For the baseline cohort, the first available performance observation
must occur at Loan Age 0 or Loan Age 1.

Loans whose first performance observation occurs at Loan Age 2 or later
are excluded from the baseline modelling cohort because earlier
delinquency events cannot be ruled out.

### Empirical Validation

In the 2015 50,000-loan sample:

| First Observed Loan Age | Loans | Observed Serious Delinquencies | Observed Event Rate |
|---|---:|---:|---:|
| 0 | 43,842 | 319 | 0.728% |
| 1 | 3,439 | 24 | 0.698% |
| 2–5 | 1,235 | 3 | 0.243% |
| 6–12 | 49 | 0 | 0.000% |
| 13–24 | 94 | 0 | 0.000% |
| 25+ | 1,341 | Not observable | N/A |

The sharp decline in observed event rates among later-entry loans is
consistent with incomplete observation of the beginning of the
performance window.

The baseline cohort therefore requires:

`first_loan_age <= 1`

This retains 47,281 of the original 50,000 loans (94.56%).


## Early Termination

Loans may terminate before completing the full 24-month performance
window.

In the exploratory 2015 sample, 7,591 loans stopped being observed
before Loan Age 24.

The termination distribution was:

| Zero Balance Code | Count |
|---|---:|
| 01 | 7,547 |
| 02 | 3 |
| 03 | 4 |
| 09 | 6 |
| 15 | 1 |
| 96 | 29 |
| Missing | 1 |

### Voluntary Prepayment

Loans terminating through Zero Balance Code (ZBC) `01` are treated as
non-events in the baseline binary model if serious delinquency was not
observed before termination.

Prepayment is recognized as a competing event. A future extension may
model prepayment and serious delinquency using survival or
competing-risks methods.

### Adverse Terminations

All 13 early terminations with Zero Balance Code (ZBC) `02`, `03`, or
`09` in the exploratory sample had already experienced serious
delinquency.

Therefore, these Zero Balance Codes do not currently add positive cases
beyond the delinquency-based target definition.

Zero Balance Code (ZBC) is used as a target-validation and
termination-classification field rather than being directly incorporated
into `ever_90dpd_24m`.

### Special or Unexplained Terminations

If serious delinquency is observed before termination, the loan remains
a positive event regardless of the subsequent termination code.

Otherwise, loans that terminate before completion of the outcome window
without a voluntary payoff (`01`) are excluded from the baseline cohort.
This includes special, adverse, unexplained, and missing zero-balance states
unless serious delinquency was already observed.

This avoids assigning a negative outcome when the full performance
window cannot be confidently observed.


## Final Cohort Reconciliation

Application of the finalized cohort and outcome-observability rules to
the 50,000-loan development sample produced the following population:

| Cohort Stage | Loans | Loans Removed |
|---|---:|---:|
| Initial sample | 50,000 | — |
| First performance observation at Loan Age 0 or 1 | 47,281 | 2,719 |
| Observable 24-month outcome | 47,255 | 26 |
| **Final modelling cohort** | **47,255** | **2,745 total** |

The 26 loans removed during the outcome-observability stage terminated
before completion of the 24-month performance window with Zero Balance
Code (ZBC) `96` and had no previously observed serious-delinquency
event.

Because the complete 24-month outcome for these loans cannot be
established, they are treated as censored and excluded rather than
being assigned to the non-event population.


## Final Target Distribution

The final modelling cohort contains:

| Target | Definition | Loans | Share |
|---|---|---:|---:|
| 0 | No observed 90+ DPD within the defined observable outcome | 46,912 | 99.274% |
| 1 | 90+ DPD / REO Acquisition within 24 months | 343 | 0.726% |
| **Total** | | **47,255** | **100.000%** |

The resulting event rate demonstrates substantial class imbalance.

Treatment of this imbalance will be evaluated during model development
rather than assumed during target construction.


# Origination Feature Eligibility

## Feature Eligibility Principle

The baseline model is designed to estimate serious-delinquency risk
using information available at mortgage origination.

However, availability at origination alone is not sufficient for a
variable to be included as a predictor.

Each candidate field is evaluated across the following dimensions:

1. **Prediction-time availability** — the information must be known at
   or before the intended prediction point.
2. **Credit-risk relevance** — there should be a plausible borrower,
   collateral, loan-structure, program, or geographic relationship with
   mortgage credit risk.
3. **Data quality** — the field must contain sufficient usable
   information after accounting for missing and sentinel values.
4. **Discriminatory potential** — constant or effectively unavailable
   fields cannot contribute information to the model.
5. **Stability and generalizability** — variables that may primarily
   capture institution-, geography-, program-, or vintage-specific
   effects require additional validation before baseline inclusion.
6. **Leakage risk** — identifiers and information arising after
   origination are not permitted as predictors.

Feature eligibility is intentionally separated from feature engineering.

**Feature eligibility** determines what information the model is allowed
to use.

**Feature engineering** determines how eligible information is cleaned,
represented, transformed, and encoded.


## Baseline Candidate Features

The initial baseline contains **17 eligible origination predictors**.

### Borrower Credit Quality and Repayment Capacity

| Feature | Credit-Risk Rationale |
|---|---|
| `credit_score` | Historical borrower credit quality |
| `original_dti` | Debt burden relative to borrower income |
| `number_of_borrowers` | Borrower structure and potential repayment capacity |
| `first_time_homebuyer_flag` | Borrower/homeownership profile |

### Collateral and Leverage

| Feature | Credit-Risk Rationale |
|---|---|
| `original_ltv` | Borrower equity and first-lien leverage |
| `original_cltv` | Combined property leverage |
| `mi_percentage` | Mortgage Insurance (MI) coverage and high-LTV loan structure |
| `property_type` | Collateral/property characteristics |
| `occupancy_status` | Principal residence, investment property, or second-home status |

### Loan Structure

| Feature | Credit-Risk Rationale |
|---|---|
| `original_upb` | Original mortgage principal/exposure |
| `original_interest_rate` | Mortgage pricing and scheduled payment burden |
| `original_loan_term` | Contractual repayment horizon |
| `loan_purpose` | Purchase, cash-out refinance, or no-cash-out refinance |
| `channel` | Origination/acquisition channel |
| `super_conforming_flag` | Loan structure relative to applicable conforming limits |

### Program

| Feature | Credit-Risk Rationale |
|---|---|
| `harp_indicator` | Home Affordable Refinance Program (HARP) status |

### Geography

| Feature | Credit-Risk Rationale |
|---|---|
| `property_state` | Broad geographic housing and economic environment |

These variables are **candidate predictors**, not guaranteed final model
features.

Predictive contribution, redundancy, stability, and out-of-time
performance will be evaluated during model development.


## Challenger Features

Several variables may contain predictive information but are
intentionally excluded from the first baseline specification.

| Feature | Reason for Challenger Status |
|---|---|
| `postal_code` | High-cardinality geographic variable with potential geographic memorization risk |
| `msa` | High-cardinality geography with approximately 11.5% missingness |
| `seller_name` | May capture institution-specific origination effects that can change over time |
| `special_eligibility_program` | Extremely sparse in the development population |

These variables may subsequently be introduced into challenger models.

A challenger feature should only be promoted if it demonstrates
meaningful incremental discriminatory power without materially weakening
stability, interpretability, or out-of-time generalization.


## Fields Excluded from Predictive Modelling

### Identifiers

| Feature | Reason |
|---|---|
| `loan_id` | Unique mortgage identifier; retained for joins and traceability only |
| `pre_harp_loan_id` | Identifier of the pre-HARP mortgage rather than a borrower risk characteristic |

Identifiers are retained in the analytical dataset where required for
lineage, reconciliation, and quality control but are never included in
the model feature matrix.


### Constant or Unavailable Fields

The following fields contain no usable discriminatory variation in the
current modelling cohort:

| Feature | Observed Condition |
|---|---|
| `vantage_score_4` | `9999` for 100% of the final cohort |
| `amortization_type` | `FRM` for 100% of the final cohort |
| `prepayment_penalty_flag` | `N` for 100% of the final cohort |
| `property_valuation_method` | `7` for 100% of the final cohort |
| `interest_only_indicator` | `N` for 100% of the final cohort |

These exclusions are **population-specific**.

A field being excluded because it is constant in the current development
population does not imply that the field is conceptually irrelevant to
mortgage credit risk in other vintages or portfolios.


### Time and Redundant Fields

| Feature | Treatment |
|---|---|
| `first_payment_date` | Retained for cohort, vintage, temporal analysis, and validation design rather than used as a raw baseline predictor |
| `maturity_date` | Excluded from the baseline because it is largely determined by origination timing and original loan term |


## Sentinel and Missing-Value Considerations

Raw null counts alone are insufficient for assessing missing information
because several origination fields use special sentinel values.

Important values identified during the eligibility review include:

| Feature | Sentinel | Interpretation |
|---|---:|---|
| `original_dti` | `999` | Unavailable / non-usable Debt-to-Income Ratio (DTI) |
| `original_ltv` | `999` | Unavailable Loan-to-Value Ratio (LTV) information |
| `original_cltv` | `999` | Unavailable Combined Loan-to-Value Ratio (CLTV) information |
| `vantage_score_4` | `9999` | Unavailable / non-usable score in the current cohort |
| `first_time_homebuyer_flag` | `9` | Unknown / unavailable |

For example, `original_dti` contains no conventional null values, but
3,899 of 47,255 loans (8.25%) contain the `999` sentinel.

Sentinel values are therefore explicitly normalized during feature
engineering rather than treated as genuine numerical observations.


## Cross-Variable Domain Validation

Feature review included checks based on mortgage-domain relationships
rather than relying exclusively on univariate distributions.

### Loan-to-Value Ratio and Combined Loan-to-Value Ratio Consistency

Within the final modelling cohort:

| Relationship | Loans |
|---|---:|
| `original_cltv < original_ltv` | 0 |
| `original_cltv = original_ltv` | 44,457 |
| `original_cltv > original_ltv` | 2,798 |

The expected relationship:

`CLTV >= LTV`

therefore holds throughout the current cohort.

Approximately 94.1% of loans have equal Loan-to-Value Ratio (LTV) and
Combined Loan-to-Value Ratio (CLTV).

Consequently, the two features may contain substantial redundant
information.

Both remain eligible at this stage. Redundancy will be evaluated during
model development rather than resolved through an arbitrary
pre-modelling exclusion.


### Loan-to-Value Ratio and Mortgage Insurance

Mortgage Insurance (MI) exhibits a strong structural relationship with
Original Loan-to-Value Ratio (LTV).

Mortgage Insurance is uncommon at or below approximately 80% LTV and
becomes substantially more prevalent above that level.

This relationship provides a useful domain-consistency check but also
demonstrates that predictive associations must not automatically be
interpreted causally.

For example, a higher delinquency rate among loans with Mortgage
Insurance could partly reflect the higher leverage characteristics of
loans requiring Mortgage Insurance rather than an adverse causal effect
of insurance itself.


## Feature Governance Categories

Origination fields are classified into five functional groups:

| Category | Purpose |
|---|---|
| **Baseline** | Eligible for the initial model specification |
| **Challenger** | Potentially useful but requires additional stability/generalization evidence |
| **Non-Predictive** | Retained for lineage or operational purposes but prohibited as model predictors |
| **Constant / Unavailable** | Contains no usable discriminatory information in the current cohort |
| **Time / Validation** | Retained primarily for temporal analysis, cohort construction, or validation design |

These classifications are implemented in:

`src/credit_risk/features/eligibility.py`

The implementation acts as the programmatic representation of the
feature-governance decisions documented here.


## Feature Engineering Boundary

Completion of the Origination Feature Eligibility Review establishes
the information set available to the baseline model.

Sentinel normalization and the DTI missingness indicator are implemented
during data construction.

The remaining fitted feature-engineering steps belong to the modelling
stage and are performed only after the development population has been
split into training and validation populations.

This phase will address:

- missing-value imputation,
- categorical representation,
- numerical transformations where justified,
- derived features where economically meaningful,
- feature-pipeline reproducibility,
- train/validation consistency,
- and automated modelling-stage quality checks.

No transformation should change the economic meaning of an origination
field without an explicit methodological justification.


# Development and Validation Strategy

## Validation Objective

Model validation must estimate how well the serious-delinquency model
generalizes beyond the observations used to estimate model parameters.

The validation design must also account for the low event rate in the
development population.

The final 2015 modelling cohort contains:

| Population | Loans | Serious-Delinquency Events | Event Rate |
|---|---:|---:|---:|
| Development cohort | 47,255 | 343 | 0.726% |

Because serious delinquency is rare, unnecessarily fragmenting the current
sample would materially reduce the number of positive observations available
for model estimation and validation.


## Temporal Coverage of the Development Sample

`first_payment_date` was evaluated as a potential basis for temporal
validation.

Observed first-payment dates technically range from February 2015 to May
2017. However, the distribution is highly concentrated between March 2015
and February 2016.

Observations after this period are extremely sparse and therefore do not
provide a sufficiently sized independent population for meaningful
out-of-time model evaluation.

A small temporal tail from the same development extract will therefore not
be treated as a true out-of-time test population.


## Development Population Construction

Model development is not structurally restricted to a single Freddie Mac
vintage.

Each data-pipeline execution constructs and persists a loan-level
`model_input` dataset for an individual configured vintage.

The modelling layer can subsequently load one or more persisted
vintage-specific modelling datasets and combine them into a single
development population.

Conceptually:

`Vintage Model Inputs -> Combine Development Vintages -> Development Population`

Development vintages are controlled through modelling configuration.

For example:

```yaml
modelling:
  vintages_train:
    - 2015
```

The same modelling infrastructure can subsequently support:

```yaml
modelling:
  vintages_train:
    - 2015
    - 2016
    - 2017
```

without requiring the underlying data-construction pipeline to process
multiple vintages simultaneously.

The data-construction and modelling configurations therefore have different
responsibilities:

- the configured data vintage identifies the vintage being constructed by
  the current data-pipeline execution;
- `vintages_train` identifies the persisted vintage populations consumed
  for model development.

A `vintage` field is added when vintage-specific model inputs are loaded.

This field is retained as modelling metadata rather than as a baseline
predictor.

It supports:

- population reconciliation,
- vintage-level diagnostics,
- stability analysis,
- model-performance analysis by vintage,
- and future temporal validation.

All combined modelling vintages are required to have compatible modelling
schemas.

Loan identifiers are also validated when vintages are combined so that the
same mortgage cannot silently appear more than once in the development
population.


## Development Split

The combined development population is divided into:

- **Training population**
- **Validation population**

using a configurable development split.

The baseline configuration uses an approximately **80% / 20% stratified
random split**.

Stratification is performed on:

`ever_90dpd_24m`

so that the rare-event proportion is approximately preserved in both
populations.

The modelling configuration controls:

- validation-population size,
- random state,
- and whether target stratification is enabled.

A fixed random state makes the population assignment reproducible for an
unchanged input population and configuration.

Before splitting, the implementation validates that:

- the target column exists,
- the target contains no missing values,
- the target contains at least two classes,
- and the configured validation fraction lies strictly between zero and one.

The training population is used for:

- fitting preprocessing parameters,
- fitting model parameters,
- feature-development decisions,
- hyperparameter selection,
- and model estimation.

The validation population is held separate from model fitting and is used
to evaluate model discrimination and generalization during development.


## Development Split Persistence

The generated training and validation populations are persisted as separate
modelling artifacts.

Conceptually:

`Development Population -> Development Split -> Train + Validation`

The combined multi-vintage development population itself is constructed in
memory and is not separately persisted.

Its constituent vintage-specific `model_input` datasets already provide the
durable source artifacts from which the development population can be
reconstructed.

The exact training and validation populations are persisted because
population assignment is important for reproducible model development.

Both persisted datasets retain:

- `loan_id` for lineage and reconciliation,
- `vintage` for temporal and stability analysis,
- eligible origination-time information,
- and the modelling target.

`loan_id` and `vintage` are retained as metadata and are not baseline
predictors.

Repeated execution with an unchanged development population, split
configuration, and random state reproduces the same population assignment.

Routine reruns of an unchanged configuration do not require creation of a
new split version.

A new development-dataset version should instead be established when the
underlying population definition or splitting methodology changes
materially, including changes to:

- development vintages,
- cohort eligibility,
- target definition,
- validation fraction,
- random state,
- or splitting strategy.

Model experiments will subsequently be versioned separately from routine
reconstruction of an unchanged development split.


## Independent Out-of-Time Validation

A later Freddie Mac origination vintage will be used as a separate
**out-of-time (OOT) validation population** once the baseline modelling
pipeline has been finalized.

This population should represent loans originating after the development
population and should not contribute information to:

- feature engineering decisions,
- imputation parameters,
- encoding parameters,
- model fitting,
- hyperparameter selection,
- or decision-threshold development.

This creates a distinction between:

**Development validation**

and

**Temporal generalization validation**

The development validation set supports model iteration within the current
development population, while the later-vintage out-of-time population
evaluates whether the finalized modelling framework generalizes to a
genuinely later mortgage population.


## Preprocessing and Leakage Control

The development split is completed before any fitted modelling-stage
preprocessing is performed.

Any transformation that learns information from the data must therefore be
fitted using the training population only.

Examples include:

- numerical imputation values,
- categorical imputation rules that depend on observed distributions,
- categorical encodings,
- scaling parameters,
- feature-selection statistics,
- and other fitted transformations.

The fitted transformation is then applied unchanged to the validation and
future out-of-time populations.

The modelling sequence is therefore:

`Model Inputs -> Development Population -> Train / Validation Split`

followed by:

`Training Data -> Fit Preprocessor -> Transform Training Data`

`Validation Data -> Apply Fitted Preprocessor -> Transform Validation Data`

`OOT Data -> Apply Fitted Preprocessor -> Transform OOT Data`

The validation population therefore contributes no information to the
estimation of preprocessing parameters.

Likewise, future out-of-time populations will not influence fitted
preprocessing or model construction.

This ordering prevents validation or future-population information from
influencing model development.


# Missing-Value Treatment

## Missingness Assessment

Sentinel normalization identified missing information among four baseline
features:

| Feature | Missing Loans | Missing Rate |
|---|---:|---:|
| `original_dti` | 3,899 | 8.25% |
| `first_time_homebuyer_flag` | 1 | ~0.00% |
| `original_ltv` | 1 | ~0.00% |
| `original_cltv` | 1 | ~0.00% |

Missingness is therefore primarily concentrated in Original
Debt-to-Income Ratio (DTI).


## Informative DTI Missingness

The relationship between DTI availability and the serious-delinquency
target was evaluated before selecting an imputation strategy.

| DTI Availability | Loans | Events | Event Rate |
|---|---:|---:|---:|
| DTI available | 43,356 | 275 | 0.634% |
| DTI unavailable | 3,899 | 68 | 1.744% |

Loans with unavailable DTI therefore exhibit an observed serious-delinquency
rate approximately **2.75 times** that of loans with available DTI.

This indicates that DTI missingness itself contains potentially useful
predictive information.


## Baseline Missing-Value Strategy

Sentinel normalization and the `original_dti_missing` indicator are already
implemented during data construction.

The development train/validation split is also implemented.

The remaining imputation and categorical-level treatments represent the
next modelling-stage implementation.

The baseline treatment is:

| Feature | Treatment |
|---|---|
| `original_dti` | Training-set median imputation + explicit missingness indicator |
| `original_ltv` | Training-set median imputation |
| `original_cltv` | Training-set median imputation |
| `first_time_homebuyer_flag` | Explicit `Unknown` categorical level |

The DTI missingness indicator distinguishes borrowers with genuinely
observed DTI values from borrowers whose DTI values were unavailable and
subsequently imputed.

No separate missingness indicators are created for Original LTV or Original
CLTV because only one missing observation is present in each field in the
current development population.


## Training-Only Imputation

Numerical imputation values are not calculated using the complete modelling
cohort.

The median values used for DTI, LTV, and CLTV will be estimated from the
**training population only**.

The same fitted values will subsequently be applied unchanged to:

- the validation population,
- and future out-of-time populations.

This ensures that preprocessing does not introduce information leakage from
validation or future observations.


# Current Modelling Pipeline State

The implemented modelling-data preparation flow is:

```text
Per-Vintage Model Inputs
        ↓
Load Configured Development Vintages
        ↓
Validate Schemas and Loan Identifiers
        ↓
Combine Development Population
        ↓
Configurable Stratified Development Split
        ↓
        ├── Training Population
        │       ↓
        │   Persist Train Dataset
        │
        └── Validation Population
                ↓
            Persist Validation Dataset
```

The modelling stage is integrated with the main project pipeline and can be
enabled or skipped through configuration.

The modelling-data preparation components are covered by unit tests for:

- single-vintage loading,
- multi-vintage loading,
- vintage metadata retention,
- empty vintage configuration,
- duplicate loan detection,
- schema consistency,
- development split size,
- target stratification,
- train/validation population separation,
- complete population reconciliation,
- split reproducibility,
- validation-size constraints,
- target integrity,
- modelling-pipeline orchestration,
- persistence routing,
- and configured pipeline skipping.


# Next Modelling Phase

The development-population and validation infrastructure is now complete.

The next phase is **fitted preprocessing**.

The immediate modelling sequence is:

```text
Persisted Training Population
        ↓
Fit Preprocessing
        ↓
Imputation
        ↓
Categorical Encoding
        ↓
Other Fitted Transformations
        ↓
Transform Training Population
        │
        ├───────────────┐
        │               │
        ▼               ▼
Model Development   Apply Same Fitted
                    Preprocessor
                        ↓
                  Validation Population
```

The first implementation priorities are:

1. define numerical, categorical, metadata, and target columns;
2. implement training-only numerical imputation;
3. implement categorical missing-value treatment;
4. define categorical encoding;
5. ensure `loan_id`, `vintage`, and the target are excluded from the
   predictor matrix;
6. fit all learned transformations exclusively on training data;
7. apply the fitted preprocessing pipeline unchanged to validation data;
8. persist the fitted preprocessing artifact for reproducibility;
9. add unit and pipeline-level tests for leakage prevention and transformation
   consistency.

Only after this preprocessing boundary is established will baseline model
estimation begin.