# Mortgage Credit Risk Reference Guide

This guide applies to V2. Verify release-specific Freddie Mac mappings against the applicable data dictionary before applying this guidance elsewhere; V3 is maintained separately.

## Core terminology

| Term | Meaning |
|---|---|
| PD | Probability of Default |
| LGD | Loss Given Default |
| EAD | Exposure at Default |
| UPB | Unpaid Principal Balance |
| DPD | Days Past Due |
| REO | Real Estate Owned |
| ZBC | Zero Balance Code |
| LTV / CLTV | First-lien / combined debt relative to property value |
| DTI | Borrower debt obligations relative to income |
| MI | Mortgage insurance |
| HARP | Home Affordable Refinance Program |

`future_90dpd_12m` is a PD-like serious-delinquency outcome. It does not estimate expected loss (`PD × LGD × EAD`).

## Delinquency and target mapping

| Performance status | Interpretation | Project treatment |
|---|---|---|
| `00` | Current / under 30 DPD | Non-serious |
| `01` | About 30–59 DPD | Non-serious |
| `02` | About 60–89 DPD | Non-serious |
| `03` | About 90–119 DPD | Serious delinquency |
| `04+` | 120+ DPD | Serious delinquency |
| `RA` | REO acquisition | Serious-delinquency event |
| `XX` | Unavailable | Not a numeric delinquency value |

At observation age *a*, the target is one when numeric status `>= 3` or `RA` occurs from *a + 1* through *a + 12*. Performance data through *a* may be used only for features.

## Zero Balance Codes and observability

ZBC describes termination, whereas delinquency status describes monthly condition; they are not interchangeable.

| Code | Common interpretation | Treatment without a prior event |
|---|---|---|
| `01` | Voluntary payoff or maturity | Non-event |
| `02`, `03`, `09` | Credit-related/adverse termination | Exclude |
| `15`, `16`, `96`, missing, other | Special, transfer, defect, or unexplained termination | Exclude |

An observation remains an event if serious delinquency precedes termination.

## Active feature groups

| Group | Examples | Constraint |
|---|---|---|
| Origination risk/capacity | `credit_score`, `original_dti`, borrower count | Available at origination |
| Leverage/collateral | LTV, CLTV, MI, property type, occupancy | MI is related to leverage |
| Structure/programme | UPB, rate, term, purpose, channel, HARP | May proxy vintage/pricing |
| Geography | `property_state` | Requires stability/governance review |
| Point-in-time state | current UPB/rate, estimated LTV, age, remaining term | Available at observation age |
| Behavioural history | current/max DPD, delinquency count, recency | No post-observation information |

## Common codes and sentinels

| Field | Values or sentinel | Treatment |
|---|---|---|
| Occupancy | `P`, `I`, `S` | Principal, investment, second home |
| Purpose | `P`, `N`, `C` | Purchase, no-cash-out, cash-out refinance |
| First-time-homebuyer | `Y`, `N`, `9` | `9` becomes missing |
| `original_dti` | `999` | Missing; add `original_dti_missing` |
| `original_ltv` / `original_cltv` | `999` | Missing |
| Delinquency status | `XX` | Unavailable for numeric parsing |

Numeric missing values are median-imputed and categorical values become `Unknown`; fitting occurs on training data only.

## Interpretation rules

- Predictive association is not causation.
- Availability alone does not make a field eligible: assess leakage, cardinality, stability, missingness, and governance.
- Identifiers are never model predictors.
- Geographic, seller, programme, and pricing effects can drift across vintages and require independent validation.
