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

---

## Performance Window

The baseline outcome horizon is 24 months from the beginning of the
loan's performance history.

Alternative horizons such as 12 and 36 months may be evaluated in
future model versions.

---

## Cohort Eligibility

Loans must have sufficiently complete early performance history for the
24-month outcome to be observable.

For the baseline cohort, the first available performance observation
must occur at Loan Age 0 or Loan Age 1.

Loans whose first performance observation occurs at Loan Age 2 or later
are excluded from the baseline modelling cohort because earlier
delinquency events cannot be ruled out.

### Empirical validation

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

---

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

Loans terminating through ZBC `01` are treated as non-events in the
baseline binary model if serious delinquency was not observed before
termination.

Prepayment is recognized as a competing event. A future extension may
model prepayment and serious delinquency using survival or
competing-risks methods.

### Adverse Terminations

All 13 early terminations with ZBC `02`, `03`, or `09` in the exploratory
sample had already experienced serious delinquency.

Therefore, these Zero Balance Codes do not currently add positive cases
beyond the delinquency-based target definition.

ZBC is used as a target-validation and termination-classification field,
rather than being directly incorporated into `ever_90dpd_24m`.

### Special or Unexplained Terminations

If serious delinquency is observed before termination, the loan remains
a positive event regardless of the subsequent termination code.

Otherwise, loans that terminate before completion of the outcome window
through special or unexplained termination states such as ZBC `15`,
ZBC `96`, or missing ZBC are excluded from the baseline cohort.

This avoids assigning a negative outcome when the full performance
window cannot be confidently observed.



### Final Cohort Reconciliation

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
Code (ZBC) 96 and had no previously observed serious-delinquency event.

Because the complete 24-month outcome for these loans cannot be
established, they are treated as censored and excluded rather than
being assigned to the non-event population.

### Final Target Distribution

The final modelling cohort contains:

| Target | Definition | Loans | Share |
|---|---|---:|---:|
| 0 | No observed 90+ DPD within the defined observable outcome | 46,912 | 99.274% |
| 1 | 90+ DPD / REO Acquisition within 24 months | 343 | 0.726% |
| **Total** | | **47,255** | **100.000%** |

The resulting event rate demonstrates substantial class imbalance.
Treatment of this imbalance will be evaluated during model development
rather than assumed during target construction.