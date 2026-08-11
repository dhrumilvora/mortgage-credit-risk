# Mortgage Credit Risk Reference Guide

This guide defines the terms and Freddie Mac fields used by the project. Dataset-specific mappings should be checked against the applicable Freddie Mac data dictionary before applying them to another release or product.

## Core terminology

| Term | Meaning |
|---|---|
| PD | Probability of Default — probability of a defined credit event |
| LGD | Loss Given Default — proportion of exposure lost following default |
| EAD | Exposure at Default — exposure outstanding at default |
| UPB | Unpaid Principal Balance |
| DPD | Days Past Due |
| REO | Real Estate Owned; property acquired through foreclosure |
| ZBC | Zero Balance Code; why a loan leaves active performance reporting |
| LTV / CLTV | First-lien / combined debt relative to property value |
| DTI | Borrower debt obligations relative to income |
| MI | Mortgage insurance protecting lender/investor credit exposure |
| HARP | Home Affordable Refinance Program |

The project models a serious-delinquency outcome and is therefore PD-like. It does not estimate expected loss, commonly summarized as `PD × LGD × EAD`.

## Delinquency and target mapping

| Performance status | Interpretation | Project treatment |
|---|---|---|
| `00` | Current / under 30 DPD | Non-serious |
| `01` | Approximately 30–59 DPD | Non-serious |
| `02` | Approximately 60–89 DPD | Non-serious |
| `03` | Approximately 90–119 DPD | Serious delinquency |
| `04+` | 120+ DPD | Serious delinquency |
| `RA` | REO acquisition | Serious delinquency event |
| `XX` | Unavailable | Not a valid numeric delinquency value |

`ever_90dpd_24m` is one when a loan reaches numeric status `>= 3` or `RA` in its first 24 observable performance months.

## Zero Balance Codes and observability

ZBC describes termination, while delinquency status describes a loan's condition during a reporting month. They are not interchangeable.

| Code | Common interpretation | Project handling when outcome is incomplete and no event was observed |
|---|---|---|
| `01` | Voluntary payoff or maturity | Non-event |
| `02`, `03`, `09` | Credit-related or adverse termination | Exclude / censor unless event was already observed |
| `15`, `16`, `96`, missing, other | Special, transfer, defect, or unexplained termination | Exclude / censor unless event was already observed |

If serious delinquency precedes termination, the loan remains an event regardless of ZBC.

## Origination fields

| Field | Meaning | Modelling note |
|---|---|---|
| `credit_score` | Borrower credit-quality score | Current feature engineering can replace it with score bands |
| `original_dti` | Debt-to-income ratio | `999` is treated as unavailable, not as a genuine ratio |
| `number_of_borrowers` | Number of borrowers | May proxy household repayment structure |
| `first_time_homebuyer_flag` | First-time-homebuyer status | `9` is treated as unavailable |
| `original_ltv` | Original first-lien LTV | `999` is treated as unavailable |
| `original_cltv` | Original combined LTV | `999` is treated as unavailable |
| `mi_percentage` | Mortgage-insurance coverage percentage | Related to leverage; not independent of LTV |
| `original_upb` | Original loan balance | Exposure, not affordability on its own |
| `original_interest_rate` | Contract rate at origination | Also reflects vintage and pricing conditions |
| `original_loan_term` | Contractual term in months | Common values include 180 and 360 |
| `loan_purpose` | Purchase, no-cash-out refinance, or cash-out refinance | May distinguish borrower populations |
| `occupancy_status` | Principal, investment, or second-home occupancy | Potential incentive and performance differences |
| `property_type` | Collateral type | Potential collateral/liquidity differences |
| `property_state` | Property state | Broad geography; assess time stability |
| `channel` | Origination/acquisition channel | May capture process or population effects |
| `super_conforming_flag` | Higher-conforming-limit indicator | Loan-structure information |
| `harp_indicator` | HARP participation | Program information |

### Configured transformed features

The active modelling configuration uses `credit_score_bins`, `original_ltv_bins`, and `original_cltv_bins` as categorical representations of their respective origination fields. It also includes `original_upb_log`, the natural logarithm of positive original UPB values, alongside the raw UPB field. These are model representations rather than new source-data fields.

## Common categorical codes

| Field | Code | Meaning |
|---|---|---|
| Occupancy status | `P`, `I`, `S` | Principal residence, investment property, second home |
| Loan purpose | `P`, `N`, `C` | Purchase, no-cash-out refinance, cash-out refinance |
| Property type | `SF`, `CO`, `PU`, `MH`, `CP` | Single-family, condominium, planned unit development, manufactured housing, cooperative |
| First-time-homebuyer flag | `Y`, `N`, `9` | Yes, no, unavailable |
| HARP / super-conforming flag | `Y`, `N` | Yes, no |

## Sentinel values and missingness

Missingness must be interpreted before modelling. A non-null value can still be a sentinel.

| Field | Sentinel | Implemented project treatment |
|---|---:|---|
| `original_dti` | `999` | Convert to missing; add `original_dti_missing` |
| `original_ltv` | `999` | Convert to missing |
| `original_cltv` | `999` | Convert to missing |
| `first_time_homebuyer_flag` | `9` | Convert to missing |
| `current_loan_delinquency_status` | `XX` | Treat as unavailable for numeric delinquency parsing |
| `vantage_score_4` | `9999` | Domain guidance; not part of baseline predictors |

Numeric missing values are median-imputed after the training/validation split. Categorical missing values are represented as `Unknown` by the fitted model preprocessor.

## Important interpretation rules

- Predictive association does not establish causation. For example, MI can be associated with risk because it is more common on higher-LTV loans.
- A field's availability does not make it an eligible model feature. Consider leakage, cardinality, stability, missingness, and governance.
- Loan IDs and other identifiers support joining and quality control; they must never be model predictors.
- Performance fields must not enter an origination-time predictor set. They are used here only to construct the outcome.
- Geographic, seller, and program effects can drift across vintages and require validation before use.
