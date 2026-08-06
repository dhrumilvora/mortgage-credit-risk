# Mortgage Credit Risk — Project Flow

This document tracks the end-to-end development of the mortgage credit-risk
modelling framework. The flow is updated as each stage of the project is
completed.

```mermaid
flowchart TD

    A["Freddie Mac Single-Family<br/>Loan-Level Dataset"]

    A --> B["Origination Data<br/>1 row per mortgage"]
    A --> C["Monthly Performance Data<br/>1 row per mortgage × month"]

    B --> D["Origination Data<br/>Ingestion & Transformers"]
    C --> E["Performance Data<br/>Ingestion & Transformers"]

    D --> D1["Canonical Origination<br/>Parquet"]
    E --> E1["Canonical Performance<br/>Parquet"]

    D1 --> P["Origination Feature<br/>Eligibility Review"]

    P --> P1["Baseline Features<br/>17 Candidates"]
    P --> P2["Challenger Features<br/>4 Candidates"]
    P --> P3["Exclude Identifiers,<br/>Constant & Unavailable Fields"]

    P1 --> Q["Origination Preprocessing"]
    P2 -.-> Q

    Q --> Q1["Sentinel Normalization<br/>COMPLETE"]
    Q1 --> Q2["Missingness Analysis<br/>COMPLETE"]
    Q2 --> Q3["DTI Missingness Indicator<br/>COMPLETE"]

    E1 --> PE["Performance Feature<br/>Eligibility Review"]
    PE --> PF["Performance Preprocessing<br/>COMPLETE"]

    Q3 --> MASTER["Master Loan-Month Dataset"]
    PF --> MASTER

    MASTER --> F["Performance History Analysis"]
    F --> G["Serious Delinquency Definition<br/>90+ DPD / REO"]
    G --> H["Cohort Eligibility<br/>First Loan Age = 0 or 1"]
    H --> I["24-Month Outcome Observability"]

    I --> J{"Outcome observable?"}
    J -->|"No"| K["Exclude / Censor"]
    J -->|"Yes"| L["Construct Target<br/>ever_90dpd_24m"]

    L --> M["Final Modelling Dataset<br/>47,255 loans × 30 columns"]

    M --> N["343 Events<br/>0.726%"]
    M --> O["46,912 Non-Events<br/>99.274%"]

    M --> PIPE["Package Entry Point<br/>credit_risk.run_pipeline()"]

    PIPE --> R["Development Dataset"]

    R --> S["Stratified Development Split<br/>80% Train / 20% Validation"]

    S --> T["Training Set"]
    S --> U["Validation Set"]

    T --> V["Fit Preprocessing<br/>Training Data Only"]

    V --> W["Imputation"]
    V --> X["Categorical Encoding"]
    V --> Y["Other Fitted Transformations"]

    W --> Z["Transform Training Data"]
    X --> Z
    Y --> Z

    V --> AA["Apply Fitted Preprocessor<br/>to Validation Data"]

    Z --> AB["Model Development<br/>TBD"]
    AA --> AB

    AB --> AC["Model Validation<br/>TBD"]

    AC --> AD["Calibration & Risk Bands<br/>TBD"]

    AD --> AE["Explainability & Stability<br/>TBD"]

    AE --> AF["Later Freddie Mac Vintage"]

    AF --> AG["Independent Out-of-Time<br/>Validation"]
```

## Current Status

| Stage | Status |
|---|---|
| Data ingestion | Complete |
| Canonical transformations | Complete |
| Schema validation | Complete |
| Performance-data investigation | Complete |
| Target definition | Complete |
| Cohort construction | Complete |
| Target validation | Complete |
| Origination feature eligibility | Complete |
| Performance feature eligibility | Complete |
| Sentinel normalization | Complete |
| Missingness analysis | Complete |
| DTI missingness indicator | Complete |
| Origination preprocessing | Complete |
| Performance preprocessing | Complete |
| Master loan-month integration | Complete |
| Final modelling dataset construction | Complete |
| End-to-end `run_pipeline()` integration | **Complete** |
| Development / validation strategy | Complete |
| Development split implementation | **Not started** |
| Fitted preprocessing | Not started |
| Model development | Not started |
| Model validation | Not started |
| Calibration | Not started |
| Explainability and stability | Not started |
| Independent out-of-time validation | Future phase |

## Current Pipeline

The end-to-end data pipeline is exposed through the package-level entry point:

```python
from credit_risk import run_pipeline

modeling_df = run_pipeline(2015)
```

The pipeline currently executes:

```text
Raw Freddie Mac Data
        ↓
Ingestion & Transformers
        ↓
Canonical Origination + Performance Data
        ↓
Origination + Performance Preprocessing
        ↓
Master Loan-Month Dataset
        ↓
24-Month Serious-Delinquency Target
        ↓
Final Loan-Level Modelling Dataset
```

### Validated Pipeline Output

The integrated 2015 pipeline produces:

| Metric | Value |
|---|---:|
| Final modelling rows | 47,255 |
| Unique loans | 47,255 |
| Duplicate loan IDs | 0 |
| Dataset columns | 30 |
| Serious-delinquency events | 343 |
| Non-events | 46,912 |
| Event rate | 0.726% |

The final modelling dataset therefore has exactly one row per eligible loan.

## Current Phase

The complete data-construction pipeline is now operational and validated.

The project is transitioning from **data pipeline development** to
**development-split implementation**.

The current development population will be divided using an approximately
80% / 20% stratified training and validation split.

All transformations that learn parameters from data will subsequently be
fitted on the training population only.

This includes:

- numerical imputation,
- categorical encoding,
- scaling where required,
- feature-selection statistics,
- and other fitted preprocessing transformations.

The validation population will not contribute information to fitted
preprocessing or model estimation.

A later Freddie Mac origination vintage will eventually provide an
independent out-of-time validation population.