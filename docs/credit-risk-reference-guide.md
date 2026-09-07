# Mortgage Credit Risk Reference Guide

This guide defines the main credit-risk terminology and Freddie Mac performance concepts used by the project.

> Freddie Mac field meanings and mappings should be checked against the applicable data dictionary for the release being modelled.

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
| LTV | Loan-to-value ratio |
| CLTV | Combined loan-to-value ratio |
| DTI | Debt-to-income ratio |
| MI | Mortgage insurance |
| HARP | Home Affordable Refinance Program |

The active target, `future_90dpd_12m`, is a serious-delinquency/REO outcome used as a PD-like modelling target. It is not an expected-loss calculation.

---

## Delinquency mapping

The project uses the following conceptual mapping for numeric delinquency status:

| Status | Interpretation | Project treatment |
|---|---|---|
| `00` | Current / under 30 DPD | Non-serious |
| `01` | Approximately 30–59 DPD | Non-serious |
| `02` | Approximately 60–89 DPD | Non-serious |
| `03` | Approximately 90–119 DPD | Serious event |
| `04+` | 120+ DPD | Serious event |
| `RA` | REO acquisition | Serious event |
| `XX` | Unavailable/non-numeric | Not parsed as numeric DPD |

For an observation at age `t`, the active target looks forward from:

```text
t + 1 through t + 12
```

Performance through `t` may contribute to predictors but not to the future event window.

---

## Zero Balance Codes

Zero Balance Codes describe loan termination and should not be treated as interchangeable with delinquency status.

The project specifically treats:

```text
ZBC = 01
```

as voluntary payoff/maturity when no prior serious event has occurred.

Other early termination categories are handled according to the configured target-observability rules.

A serious delinquency observed before termination remains an event.

---

## Sentinel values

Freddie Mac data contains special values that require semantic handling.

Examples in the active project include:

| Field | Sentinel | Treatment |
|---|---:|---|
| Original DTI | `999` | Missing |
| Original LTV | `999` | Missing |
| Original CLTV | `999` | Missing |
| First-time-homebuyer | `9` | Missing |
| Delinquency status | `XX` | Unavailable/non-numeric |

The project retains an explicit missingness indicator for original DTI.

Sentinel handling should happen before modelling so that sentinel values are not accidentally interpreted as economically meaningful numeric values.

---

## Point-in-time modelling

The core modelling unit is:

```text
loan_id × observation_age
```

A feature is eligible only if its value would have been available by that observation point.

Examples:

```text
current UPB at t       → valid predictor
current rate at t      → valid predictor
delinquency history ≤t → valid predictor

delinquency at t+3     → target information only
```

This distinction is fundamental to preventing temporal leakage.

---

## Active feature groups

| Group | Examples |
|---|---|
| Origination risk | credit score, DTI, borrower count |
| Leverage/collateral | LTV, CLTV, MI, estimated LTV |
| Loan structure | UPB, interest rate, term, purpose, channel |
| Programme/geography | HARP/programme flags, state |
| Current state | current UPB/rate, loan age, remaining term |
| Behavioural history | current/max DPD, delinquency counts, recency |
| Recent behaviour | 3-/12-month delinquency and assistance windows |
| Trajectory | DPD trend/acceleration/severity change, UPB/rate trajectory |
| Macro | unemployment, mortgage rate, HPI, Federal Funds rate |

---

## Missing-data treatment

The model pipeline uses training-fitted transformations.

Conceptually:

```text
numeric       → training median imputation
categorical   → Unknown + categorical encoding
```

The imputation/encoding state is learned on training data and reused for validation/test/OOT.

This prevents later populations from influencing the transformation state.

---

## Risk interpretation

Several principles apply throughout the project:

- predictive association is not causation;
- an available field is not automatically a safe model predictor;
- leakage, cardinality, missingness and temporal stability must be assessed;
- identifiers are not predictors;
- geographic/programme effects may drift across vintages;
- discrimination and calibration are separate model-quality dimensions.

---

## PD versus loss modelling

A serious-delinquency PD-like target answers a question such as:

```text
How likely is this eligible loan to experience the defined serious
event during the next 12 months?
```

It does not answer:

```text
How much money will be lost?
```

The latter requires additional concepts such as:

```text
PD × LGD × EAD
```

which are outside the scope of the current target.
