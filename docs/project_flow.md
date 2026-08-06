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

    F --> G["Serious Delinquency Definition<br/>90+ Days Past Due / REO"]

    G --> H["Cohort Eligibility<br/>First Loan Age = 0 or 1"]

    H --> I["24-Month Outcome<br/>Observability"]

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

    P1 --> Q["Feature Engineering<br/>IN PROGRESS"]
    P2 -.-> Q

    M --> Q

    Q --> R["Train / Validation / Test<br/>Strategy — TBD"]

    R --> S["Model Development<br/>TBD"]

    S --> T["Model Validation<br/>TBD"]

    T --> U["Calibration & Risk Bands<br/>TBD"]

    U --> V["Explainability & Stability<br/>TBD"]
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
| Origination feature eligibility | **Complete** |
| Feature engineering | **In progress** |
| Train / validation / test design | Not started |
| Model development | Not started |
| Model validation | Not started |
| Calibration | Not started |
| Explainability and stability | Not started |

## Current Phase

The project is currently in the **Feature Engineering** phase.

The eligible information set for the baseline model has been established
through the Origination Feature Eligibility Review.

The baseline feature set contains 17 candidate predictors spanning:

- borrower credit quality and repayment capacity,
- collateral and leverage,
- loan structure,
- refinance-program characteristics,
- and broad property geography.

High-cardinality, sparse, or potentially unstable variables have been
separated into a challenger feature set rather than automatically included
in the baseline model.

Feature engineering will now convert the eligible raw origination fields
into a reproducible modelling dataset while preserving their economic and
credit-risk meaning.