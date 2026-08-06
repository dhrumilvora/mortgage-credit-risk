# Mortgage Credit Risk — Project Flow

This document tracks the end-to-end development of the mortgage credit-risk
modelling framework. The flow is updated as each stage of the project is
completed.

```mermaid
flowchart TD

    A["Freddie Mac Single-Family<br/>Loan-Level Dataset"]

    A --> B["Origination Data<br/>1 row per mortgage"]
    A --> C["Monthly Performance Data<br/>1 row per mortgage × month"]

    B --> D["Origination Data<br/>Ingestion & Validation"]
    C --> E["Performance Data<br/>Ingestion & Validation"]

    E --> F["Performance History Analysis"]
    F --> G["Serious Delinquency Definition<br/>90+ DPD / REO"]
    G --> H["Cohort Eligibility<br/>First Loan Age = 0 or 1"]
    H --> I["24-Month Outcome Observability"]

    I --> J{"Outcome observable?"}
    J -->|"No"| K["Exclude / Censor"]
    J -->|"Yes"| L["Construct Target<br/>ever_90dpd_24m"]

    L --> M["Final Modelling Cohort<br/>47,255 loans"]
    M --> N["343 Events<br/>0.726%"]
    M --> O["46,912 Non-Events<br/>99.274%"]

    D --> P["Origination Feature<br/>Eligibility Review"]

    P --> P1["Baseline Features<br/>17 Candidates"]
    P --> P2["Challenger Features<br/>4 Candidates"]
    P --> P3["Exclude Identifiers,<br/>Constant & Unavailable Fields"]

    P1 --> Q["Raw Feature Preparation"]
    P2 -.-> Q

    Q --> Q1["Sentinel Normalization<br/>COMPLETE"]
    Q1 --> Q2["Missingness Analysis<br/>COMPLETE"]
    Q2 --> Q3["DTI Missingness Indicator<br/>COMPLETE"]

    M --> R["Development Dataset"]
    Q3 --> R

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
| Schema validation | Complete |
| Performance-data investigation | Complete |
| Target definition | Complete |
| Cohort construction | Complete |
| Target validation | Complete |
| Origination feature eligibility | Complete |
| Sentinel normalization | Complete |
| Missingness analysis | Complete |
| DTI missingness indicator | Complete |
| Development / validation strategy | **Complete** |
| Development split implementation | **Not started** |
| Fitted preprocessing | Not started |
| Model development | Not started |
| Model validation | Not started |
| Calibration | Not started |
| Explainability and stability | Not started |
| Independent out-of-time validation | Future phase |

## Current Phase

The project is currently transitioning from **raw feature preparation**
to **development-split implementation**.

Feature eligibility and raw missing-value handling rules have been established.

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