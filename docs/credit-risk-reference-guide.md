# Mortgage Credit Risk Reference Guide

This document provides definitions, code mappings, and credit-risk interpretations
for concepts and fields used throughout the mortgage credit-risk modelling project.

---

## 1. Core Credit Risk Terminology

| Acronym | Full Term | Meaning |
|---|---|---|
| PD | Probability of Default | Probability that a borrower experiences a defined default event |
| LGD | Loss Given Default | Proportion of exposure lost if default occurs |
| EAD | Exposure at Default | Exposure outstanding when default occurs |
| UPB | Unpaid Principal Balance | Remaining mortgage principal |
| DPD | Days Past Due | Number of days a scheduled payment is overdue |
| REO | Real Estate Owned | Property acquired by the lender/investor through foreclosure |
| RPL | Reperforming Loan | Previously delinquent loan that has returned to performing status |
| ZBC | Zero Balance Code | Code describing why a mortgage leaves active performance reporting |
| LTV | Loan-to-Value Ratio | Mortgage balance relative to property value |
| CLTV | Combined Loan-to-Value Ratio | Combined property-secured debt relative to property value |
| DTI | Debt-to-Income Ratio | Borrower's monthly debt obligations relative to income |
| MI | Mortgage Insurance | Insurance protecting the lender/investor against certain mortgage credit losses |
| FRM | Fixed-Rate Mortgage | Mortgage whose contractual interest rate remains fixed |
| HARP | Home Affordable Refinance Program | U.S. refinance program created for eligible borrowers who had difficulty refinancing conventionally |
| MSA | Metropolitan Statistical Area | Geographic area representing an urban economic region |

### Expected Loss

A simplified credit-risk relationship is:

**Expected Loss = PD × LGD × EAD**

The current project primarily focuses on modelling a serious-delinquency
outcome and is therefore most closely related to the **Probability of Default
(PD)** component of credit risk.

---

## 2. Delinquency and Loan Termination

### Current Loan Delinquency Status

| Code | Interpretation | Credit Risk Meaning |
|---|---|---|
| 00 | Current / less than 30 Days Past Due (DPD) | Performing |
| 01 | 30–59 DPD | Early delinquency |
| 02 | 60–89 DPD | Elevated delinquency |
| 03 | 90–119 DPD | Serious delinquency |
| 04+ | 120+ DPD | Severe delinquency |
| RA | Real Estate Owned (REO) Acquisition | Severe credit event |
| XX | Not available | Unknown |

For this project, **serious delinquency begins at 90+ Days Past Due (DPD)**,
corresponding to numeric delinquency status `>= 03`.

The project target is therefore based on whether a mortgage reaches serious
delinquency during the defined 24-month performance window.

---

### Zero Balance Code (ZBC)

Zero Balance Code describes why a mortgage terminates or leaves active
performance reporting.

| Code | Meaning | Project Interpretation |
|---|---|---|
| 01 | Prepaid or matured | Non-credit termination |
| 02 | Third-party sale | Adverse credit termination |
| 03 | Short sale / charge-off | Adverse credit termination |
| 09 | Real Estate Owned (REO) disposition | Severe adverse termination |
| 15 | Whole-loan sale | Special termination |
| 16 | Reperforming Loan (RPL) securitization | Special termination / prior distress |
| 96 | Defect-related removal | Special termination |

### Delinquency Status vs. Zero Balance Code

These variables describe different parts of the mortgage lifecycle:

- **Current Loan Delinquency Status** describes the loan's credit condition
  during a reporting month.
- **Zero Balance Code (ZBC)** describes the reason the loan terminates or
  leaves active reporting.

A loan may therefore progress through delinquency states before eventually
receiving a Zero Balance Code.

For target construction, an early voluntary payoff (`ZBC = 01`) can provide
an observable non-event outcome, while certain special early terminations may
need to be treated as censored when the complete performance outcome cannot
be established.

---

## 3. Borrower Credit Quality and Repayment Capacity

### Credit Score

Credit Score summarizes information about a borrower's historical credit
behaviour and creditworthiness.

In general:

**Lower Credit Score → Higher expected credit risk**

The relationship is not hard-coded into the model and must be validated
empirically.

In the current modelling cohort, observed Credit Scores range from
approximately **462 to 832**, with no identified missing-value sentinel.

---

### VantageScore 4

VantageScore 4 is an alternative measure of borrower creditworthiness.

In the current modelling cohort:

- All observations contain `9999`.
- The field therefore contains no usable variation.

Consequently, VantageScore 4 is not useful as a predictor for the current
vintage.

---

### Debt-to-Income Ratio (DTI)

Debt-to-Income Ratio measures borrower debt obligations relative to income.

Conceptually:

**DTI = Monthly Debt Obligations / Monthly Income**

Higher DTI generally represents lower remaining financial capacity after debt
payments.

In the current dataset:

- Valid observed values range from approximately `1` to `50`.
- `999` represents unavailable / non-usable DTI information for modelling.
- Approximately **8.25%** of the final modelling cohort contains `DTI = 999`.

`999` must therefore **not** be interpreted as a genuine 999% DTI.

Missing DTI requires explicit treatment during feature engineering.

---

### Number of Borrowers

The field identifies the number of borrowers associated with the mortgage.

The current modelling population contains:

- 1 borrower
- 2 borrowers

Borrower count may provide information about repayment capacity and household
structure, although its relationship with credit risk must be established
empirically.

---

### First-Time Homebuyer Flag

Indicates whether the borrower is classified as a first-time homebuyer.

| Code | Meaning |
|---|---|
| Y | First-time homebuyer |
| N | Not a first-time homebuyer |
| 9 | Unknown / unavailable |

The current modelling cohort contains only one observation with code `9`.

---

## 4. Collateral and Leverage

### Loan-to-Value Ratio (LTV)

Loan-to-Value Ratio measures the mortgage amount relative to the value of the
underlying property.

**LTV = Mortgage Amount / Property Value**

Example:

Property Value = $500,000  
Mortgage = $400,000

**LTV = 80%**

Higher LTV generally means the borrower has less equity in the property and
the lender/investor has a smaller collateral cushion.

Values above 100% must **not automatically be treated as data errors**.
Special mortgage structures and program characteristics may result in
reported leverage above 100%.

The special value `999` should not be interpreted as genuine leverage.

---

### Combined Loan-to-Value Ratio (CLTV)

Combined Loan-to-Value Ratio incorporates combined property-secured financing
relative to property value.

Conceptually:

**CLTV = Combined Property-Secured Debt / Property Value**

CLTV can therefore capture leverage not visible from the first mortgage alone.

A domain consistency check on the current modelling cohort found:

- `CLTV < LTV`: **0 loans**
- `CLTV = LTV`: **44,457 loans**
- `CLTV > LTV`: **2,798 loans**

Thus:

**CLTV >= LTV**

holds across the entire current modelling cohort.

This is a useful data-quality relationship to monitor.

---

### Mortgage Insurance (MI)

Mortgage Insurance protects the lender/investor against certain losses if a
borrower defaults.

It should not be confused with insurance protecting the borrower.

Mortgage Insurance is strongly related to leverage.

The current dataset exhibits a clear structural relationship around an
Original LTV of approximately 80%:

- At or below 80% LTV, Mortgage Insurance is uncommon.
- Above 80% LTV, Mortgage Insurance becomes substantially more common.

Consequently, Mortgage Insurance and LTV should not be interpreted as
independent borrower-risk characteristics.

Mortgage Insurance may also affect **Loss Given Default (LGD)** differently
from its relationship with the probability of serious delinquency.

---

### Occupancy Status

| Code | Meaning |
|---|---|
| P | Principal residence |
| I | Investment property |
| S | Second home |

Occupancy can affect borrower incentives and mortgage performance and is
therefore potentially relevant to credit risk.

---

### Property Type

Property Type identifies the type of property securing the mortgage.

Common codes in the current dataset include:

| Code | Property Type |
|---|---|
| SF | Single-Family |
| CO | Condominium |
| PU | Planned Unit Development |
| MH | Manufactured Housing |
| CP | Cooperative |

Different property types may have different collateral, valuation, liquidity,
and resale characteristics.

---

## 5. Loan Structure

### Original Unpaid Principal Balance (UPB)

Original UPB represents the original principal amount of the mortgage.

It represents **absolute loan exposure**, not borrower affordability by itself.

A larger mortgage is therefore not automatically riskier than a smaller
mortgage. UPB should be interpreted alongside borrower income capacity,
Debt-to-Income Ratio (DTI), Loan-to-Value Ratio (LTV), geography, and other
risk characteristics.

---

### Original Interest Rate

The Original Interest Rate is the contractual mortgage interest rate at
origination.

For otherwise equivalent mortgages:

**Higher Interest Rate → Higher Scheduled Payment Burden**

However, interest rates also reflect market conditions, origination vintage,
product characteristics, and potentially borrower pricing.

Observed relationships must therefore not automatically be interpreted as
causal.

---

### Original Loan Term

Original Loan Term represents the contractual term of the mortgage in months.

Common examples:

| Months | Approximate Term |
|---:|---|
| 120 | 10 years |
| 180 | 15 years |
| 240 | 20 years |
| 360 | 30 years |

The current portfolio is primarily composed of 30-year and 15-year mortgages.

---

### Loan Purpose

| Code | Meaning |
|---|---|
| P | Purchase |
| N | No Cash-out Refinance |
| C | Cash-out Refinance |

A purchase mortgage finances property acquisition.

A no-cash-out refinance primarily replaces or restructures existing mortgage
debt without substantial equity extraction.

A cash-out refinance allows the borrower to refinance while extracting
property equity.

These populations may have different credit-risk characteristics.

---

### Amortization Type

`FRM` represents a **Fixed-Rate Mortgage**.

The current modelling cohort contains only:

`FRM`

Therefore the field contains no variation in the current population and
cannot provide discriminatory information to the model.

---

### Prepayment Penalty Flag

Indicates whether the mortgage contains a prepayment penalty.

The current modelling cohort contains only:

`N`

Therefore this field contains no variation and cannot provide discriminatory
information in the current population.

---

## 6. Origination, Programs, and Geography

### Home Affordable Refinance Program (HARP)

HARP stands for **Home Affordable Refinance Program**.

The program was designed to facilitate refinancing for eligible borrowers who
could have difficulty refinancing through conventional channels, including
borrowers with limited property equity.

The `harp_indicator` identifies whether the mortgage is associated with HARP.

| Code | Meaning |
|---|---|
| Y | HARP loan |
| N | Non-HARP loan |

---

### Pre-HARP Loan ID

`pre_harp_loan_id` identifies the mortgage associated with the loan before
HARP refinancing.

It is an **identifier**, not a borrower risk characteristic.

The current data shows a strong structural relationship between HARP status
and availability of the Pre-HARP Loan ID.

The identifier itself should not be used as a model predictor.

---

### Super-Conforming Flag

Identifies mortgages associated with applicable higher conforming loan limits.

| Code | Meaning |
|---|---|
| Y | Super-conforming |
| N | Not super-conforming |

This field may provide information about loan structure and applicable
conforming-limit regimes.

---

### Metropolitan Statistical Area (MSA)

MSA identifies the metropolitan geographic area associated with the property.

In the current modelling cohort:

- approximately 446 distinct values are present;
- approximately 11.5% of observations are missing.

MSA may capture meaningful local housing and economic conditions but is a
high-cardinality geographic variable and therefore requires careful treatment
before use in modelling.

---

### Property State

Property State identifies the U.S. state or territory in which the mortgaged
property is located.

Geographic differences may reflect variation in:

- housing-market conditions,
- economic conditions,
- foreclosure processes,
- borrower populations,
- property-market dynamics.

Geographic variables should be evaluated for stability across time and
vintages rather than assumed to generalize automatically.

---

### Postal Code

The dataset contains a geographically aggregated postal-code field rather than
a full borrower address.

The current modelling cohort contains approximately 866 distinct values.

Because this is substantially more granular than Property State, it may create
high-cardinality and geographic memorization risks if used directly in a
model.

---

### Seller Name

Seller Name identifies the institution associated with selling the mortgage
into Freddie Mac's portfolio.

Seller information may capture differences in borrower populations,
origination practices, and underwriting processes.

However, seller effects may also be institution-specific and unstable over
time. Seller Name therefore requires careful stability analysis before being
used as a model predictor.

---

## 7. Identifiers and Non-Predictive Fields

### Loan ID

`loan_id` uniquely identifies a mortgage.

It is required for:

- joining origination and performance data,
- tracing observations,
- quality control,
- reproducibility.

It must **never be used as a model predictor**.

---

### Property Valuation Method

The current modelling cohort contains only:

`7`

Because the field is constant within the current modelling population, it
contains no discriminatory information.

---

### Interest-Only Indicator

The current modelling cohort contains only:

`N`

Therefore the current population contains no variation in this feature.

---

## 8. Missing Values and Sentinel Codes

A critical distinction in the Freddie Mac data is that:

**`isna() == False` does not necessarily mean information is available.**

Some fields use special values to represent unavailable information.

Known examples relevant to the current project include:

| Variable | Special Value | Project Interpretation |
|---|---:|---|
| `original_dti` | 999 | Missing / unavailable |
| `original_ltv` | 999 | Missing / unavailable |
| `original_cltv` | 999 | Missing / unavailable |
| `vantage_score_4` | 9999 | Unavailable / non-usable in current population |
| `first_time_homebuyer_flag` | 9 | Unknown / unavailable |
| `current_loan_delinquency_status` | XX | Unknown / unavailable |

These values must be handled explicitly during data cleaning and feature
engineering rather than treated as genuine numeric or categorical values.

---

## 9. Important Modelling Distinctions

### Predictive Association Is Not Causation

A feature may predict serious delinquency without causing delinquency.

For example:

Mortgage Insurance may be associated with higher observed delinquency because
Mortgage Insurance is more common among high-LTV mortgages.

Therefore:

**Predictive relationship ≠ causal relationship**

---

### Probability of Default vs. Loss Severity

Features can affect different dimensions of credit risk differently.

For example, Mortgage Insurance may reduce investor losses following default
while not necessarily reducing the probability that the borrower becomes
delinquent.

This distinction corresponds broadly to:

- **PD — Probability of Default**
- **LGD — Loss Given Default**

The current project primarily models a serious-delinquency outcome and should
not interpret all predictors as determinants of loss severity.

---

### Feature Availability Is Not Sufficient for Feature Eligibility

A field being present at origination does not automatically make it suitable
for modelling.

Potential concerns include:

- identifiers,
- zero-variance fields,
- high-cardinality variables,
- institutional dependence,
- geographic memorization,
- temporal instability,
- sparse program indicators,
- redundant information,
- missing/sentinel values,
- information unavailable at the intended prediction point.

These considerations are evaluated before final feature engineering.