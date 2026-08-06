# Mortgage Credit Risk Modelling Methodology

## Baseline Modelling Objective

The baseline model estimates, using information available at mortgage
origination, whether a loan will experience serious delinquency within
its first 24 months.

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
through special or unexplained termination states such as Zero Balance
Code (ZBC) `15`, ZBC `96`, or missing ZBC are excluded from the baseline
cohort.

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

Sentinel values will therefore be explicitly normalized during feature
engineering rather than treated as genuine numerical observations.

No imputation strategy is defined at the feature-eligibility stage.


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

The next modelling phase is **Feature Engineering**.

This phase will address:

- sentinel-value normalization,
- missing-value treatment,
- categorical representation,
- numerical transformations where justified,
- derived features where economically meaningful,
- feature-pipeline reproducibility,
- train/validation consistency,
- automated data-quality checks.

No transformation should change the economic meaning of an origination
field without an explicit methodological justification.