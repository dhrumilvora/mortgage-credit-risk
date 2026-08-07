# 🏠 Mortgage Credit Risk Prediction

An in-progress mortgage credit-risk modelling project using Freddie Mac's Single-Family Loan-Level Dataset. Its implemented data-construction and quality-reporting components prepare an origination-time modelling dataset for **future mortgage delinquency risk**.

The core question behind the project is:

> **Using information known at mortgage origination, how likely is a mortgage to become seriously delinquent within its first 24 months?**

Unlike a typical credit-risk dataset where each borrower appears once with a predefined default label, Freddie Mac provides the **monthly performance history of each mortgage**. This makes it possible to model how credit risk evolves over time.

---

## 🎯 Problem

The implemented target is whether a mortgage reaches **90+ Days Past Due (DPD)** or REO acquisition within its first 24 months of performance history: `ever_90dpd_24m`.

Freddie Mac represents delinquency approximately as the number of monthly payments the borrower is behind.

| Delinquency Status | Approximate Meaning |
| -----------------: | ------------------- |
|                  0 | Current             |
|                  1 | ~30 DPD             |
|                  2 | ~60 DPD             |
|                  3 | ~90 DPD             |
|                 4+ | 120+ DPD            |

The cleaned numeric representation used throughout the project is `delq_numeric`.

```text id="rks3mw"
delq_numeric >= 3 → Serious Delinquency
```

The model therefore estimates a **forward-looking probability of serious delinquency**, rather than simply classifying loans based on their eventual outcome.

---

## 🗃️ Data

The project uses Freddie Mac's Single-Family Loan-Level Dataset.

The data contains two main components.

### Origination Data

Information describing the mortgage when it was originated:

* Credit score
* Debt-to-Income ratio (DTI)
* Loan-to-Value ratio (LTV)
* Original loan balance
* Interest rate
* Loan term
* Loan purpose
* Occupancy status
* Property and geographic characteristics

### Monthly Performance Data

Each mortgage is subsequently tracked over time:

* Current loan balance
* Loan age
* Current interest rate
* Remaining maturity
* Delinquency status
* Modification status
* Loan termination / zero-balance information

This allows the project to reconstruct the **monthly history of each mortgage**.

---

## 🧠 Modelling Approach

The raw monthly performance data is transformed into a **point-in-time modelling dataset**.

For every observation date:

```text id="hn7g34"
                   Observation Date
                          │
        Past              │             Future
◄─────────────────────────┼──────────────────────────►
                          │
              Information known today
                          │
                          └──── Next 12 months ────►
                                  90+ DPD?
```

The current dataset uses origination-time information only. Monthly performance data is used to construct and validate the target, not as a model feature source.

This is particularly important because using information from later months would introduce **look-ahead bias / target leakage**.

---

## 🔧 Feature Engineering

Features combine static mortgage characteristics with information derived from the loan's historical performance.

### Borrower Risk

* Credit score
* DTI
* Original LTV / CLTV
* Occupancy status

### Mortgage Characteristics

* Original balance
* Current unpaid principal balance
* Interest rate
* Loan age
* Remaining maturity
* Loan purpose

### Historical Behaviour (planned)

The monthly performance history allows behavioural features such as:

* Previous delinquency
* Maximum historical delinquency
* Number of previous delinquency episodes
* Months since previous delinquency
* Recent delinquency behaviour
* Historical cure behaviour

These point-in-time features are planned for a future loan-month modelling version. They are not part of the current origination-time model input.

---

## 🤖 Models (planned)

The intended modelling approach is a **champion–challenger setup**. Model training, validation, calibration, and out-of-time evaluation are not implemented yet.

### Logistic Regression

Logistic Regression provides an interpretable credit-risk baseline and allows the relationship between borrower characteristics and delinquency risk to be examined directly.

### Gradient Boosting

A gradient-boosted decision-tree model is used as a nonlinear challenger capable of capturing interactions and nonlinear relationships between risk factors.

The comparison is not based purely on predictive accuracy.

Models are evaluated across:

* Discrimination
* Calibration
* Stability
* Interpretability

---

## 📊 Model Evaluation (planned)

Serious mortgage delinquency is an imbalanced outcome, making standard accuracy a poor measure of model quality.

Evaluation therefore focuses on:

* ROC-AUC
* Precision-Recall AUC
* KS statistic
* Precision
* Recall
* Lift
* Risk capture
* Probability calibration

A practical question is also considered:

> **If only the highest-risk 5–10% of mortgages could be reviewed, what percentage of future serious delinquencies would the model identify?**

---

## ⏳ Out-of-Time Validation (planned)

Mortgage credit risk changes with economic conditions.

Rather than randomly mixing observations across years, models are evaluated using **time-based validation**.

```text id="a8t1kn"
TIME ─────────────────────────────────────────────►

TRAIN                 VALIDATION              TEST

████████████████      ██████████              █████████
Past                                            Future
```

The final test period therefore represents a genuinely unseen future period.

Performance can also be examined across:

* Origination vintages
* Credit-score bands
* LTV bands
* Loan age
* Calendar periods

This helps determine whether model performance remains stable across different parts of the mortgage portfolio.

---

## 🏦 Credit Risk Interpretation

The project predicts **serious delinquency**, which is not necessarily the same as realized financial loss.

A borrower reaching 90+ DPD may subsequently:

```text id="0r73hj"
90+ DPD
   │
   ├── Cure
   ├── Modify
   ├── Remain Delinquent
   └── Progress toward Default / Foreclosure
```

In traditional credit-risk terminology, expected loss is often decomposed into:

```text id="0fqhkn"
Probability of Default (PD)
          ×
Loss Given Default (LGD)
          ×
Exposure at Default (EAD)
          │
          ▼
     Expected Loss
```

The model developed here focuses on the **probability of credit deterioration** and does not treat 90+ DPD as equivalent to realized credit loss.

---

## ⚠️ Key Modelling Challenges

Mortgage data introduces several complications beyond standard binary classification.

**Prepayment** — Borrowers may refinance or repay their mortgage before the end of the prediction window.

**Cures** — Delinquency is not always permanent. Borrowers can transition back to current status.

**Economic cycles** — Relationships between borrower characteristics and delinquency can change across interest-rate and economic environments.

**Repeated observations** — The same mortgage is observed repeatedly over its lifetime.

**Population drift** — Mortgage vintages originated under different underwriting and economic conditions may behave differently.

**Data leakage** — Monthly performance information occurring after the observation date must never enter model features.

Handling these correctly is a central part of the project rather than simply preprocessing before model training.
