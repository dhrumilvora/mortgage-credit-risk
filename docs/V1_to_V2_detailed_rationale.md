# V1 → V2: Detailed Model Evolution Rationale

> **Current implementation note (August 2026):** This document records the historical diagnosis that motivated the move away from the V1 origination-only model. V2 is implemented as a point-in-time behavioural model, not yet as the discrete-time hazard/survival model proposed below. The V2 configuration uses observation ages 6 and 12, strictly lagged performance features, and a 12-month forward serious-delinquency target. V3 is being updated separately. See [the V2 methodology](modelling-methodology.md) for the implemented design.

## 1. Executive Summary

V1 was designed as a **static origination-time mortgage credit-risk model**.

Its question was:

> **Given the information available when a mortgage is originated, how likely is the loan to experience serious delinquency within the following 24 months?**

V1 successfully demonstrated that origination information contains meaningful predictive signal. The model achieved strong validation discrimination and produced useful risk ordering.

However, when evaluated out-of-time on later vintages, the model's performance deteriorated materially. We investigated whether this could be solved through calibration, drift handling, or temporal retraining.

Those experiments showed that:

1. V1 has meaningful risk-ranking power.
2. V1's probability estimates become unstable out-of-time.
3. The feature population changes across vintages.
4. More importantly, the relationship between features and delinquency changes across vintages.
5. Static calibration does not solve the OOT problem.
6. Retraining with a more recent vintage improves the overall probability level but does not recover the lost discrimination or calibration slope.

Therefore, the issue appears deeper than model aging or probability calibration.

We are moving to V2 to investigate a **time-dependent credit-risk formulation**, beginning with a **discrete-time hazard model** and subsequently comparing it with a survival-model approach.

V1 remains the baseline and will continue to be supported.

---

# 2. What V1 Was Trying to Predict

V1 treats each mortgage primarily as an origination-time observation.

The modelling structure is:

```text
Origination data
      ↓
Origination-time features
      ↓
24-month serious-delinquency target
      ↓
Loan-level model
      ↓
Probability of serious delinquency
```

The target is:

```text
ever_90dpd_24m
```

The model therefore asks whether serious delinquency occurs at any point within the defined 24-month horizon.

A key design principle was that the final model features come from the **origination-time feature population**.

Performance history is used to construct the target, but future monthly performance information is not carried into the final V1 feature population.

This gives V1 a clean interpretation:

> **What can we predict about future credit risk using information available at loan origination?**

---

# 3. V1 Was Not a Failed Model

This is important.

The conclusion from the V1 investigation is **not**:

> "The model doesn't work."

V1 demonstrated substantial predictive signal.

The validation results included approximately:

```text
ROC-AUC       0.824
PR-AUC        0.068
KS            0.523
Brier score   0.0155
Log loss      0.0797
```

The ROC-AUC of approximately 0.824 indicates that the model was able to rank higher-risk loans above lower-risk loans reasonably well during validation.

The risk-decile analysis supported the same conclusion.

For example:

```text
Validation

Bottom risk decile
Actual event rate ≈ 0.026%

Top risk decile
Actual event rate ≈ 6.26%
```

Therefore, the origination features contain meaningful information about future delinquency.

---

# 4. The First Problem: Probability Calibration

Although V1 ranked risk reasonably well, its predicted probabilities were not perfectly aligned with the observed event rate.

For the validation population:

```text
Actual event rate       ≈ 1.33%
Average predicted PD    ≈ 4.74%
```

However, the calibration slope was approximately:

```text
Calibration slope ≈ 1.08
```

with the ideal value being:

```text
Calibration slope = 1
Calibration intercept = 0
```

This initially suggested that the model might have a probability-scale problem.

That led naturally to calibration analysis.

---

# 5. Calibration Experiments

We tested model-agnostic calibration approaches, including:

- Sigmoid / Platt-style calibration
- Isotonic calibration

The hypothesis was:

```text
Raw model probability
        ↓
Calibration mapping
        ↓
Better probability estimate
```

The calibration mapping was learned from the historical development population and then evaluated out-of-time.

The results did not support this as a solution.

For the later OOT population, the raw model was already underpredicting the observed event rate, while calibration learned from the earlier period pushed the predictions even further away from the later observed risk.

Approximate 2019 results were:

```text
Actual event rate       ≈ 4.65%
Raw model average PD    ≈ 3.49%
Sigmoid average PD      ≈ 0.97%
Isotonic average PD     ≈ 0.98%
```

The OOT calibration slope also remained substantially different from the ideal value.

### What this told us

Calibration can adjust the mapping:

```text
predicted probability → calibrated probability
```

but it cannot fundamentally solve a situation where the underlying relationship itself changes across time.

The calibration mapping learned in one environment was simply not transferable to the later environment.

Therefore:

> **The problem was not merely a static probability-calibration problem.**

---

# 6. The OOT Problem Became Clearer

The original V1 OOT results showed substantial deterioration.

Approximately:

```text
                    Validation       2019 OOT
------------------------------------------------
ROC-AUC               0.824            0.696
KS                    0.523            0.292
Brier score           0.0155           0.0443
Log loss              0.0797           0.1816
```

The model still retained useful ranking information, but its performance was materially worse in the later vintage.

At the same time, the overall event environment had changed significantly.

```text
Validation event rate ≈ 1.33%

2019 OOT event rate   ≈ 4.65%
```

The later population therefore represented a materially different risk environment.

This led to the next question:

> **What changed between the development vintages and the later OOT vintages?**

---

# 7. Temporal Drift Analysis

We investigated two different types of temporal change.

## 7.1 Population / feature-distribution drift

The first question was:

> Did the distribution of the model inputs change?

In other words, did:

```text
P(X)
```

change over time?

There was evidence that it did.

The strongest example was:

```text
original_interest_rate
PSI ≈ 0.394
```

This indicates substantial distributional change.

Other features showed smaller changes.

However, population drift alone did not explain the complete OOT deterioration.

---

# 8. Relationship Drift Was More Important

The second question was:

> **Did the relationship between a feature and delinquency change over time?**

This is a different problem.

Instead of looking only at:

```text
P(X)
```

we examined changes in:

```text
P(Y | X)
```

For several features, the observed event rate for comparable feature ranges changed materially between vintages.

Examples included:

```text
original_dti
    max event-rate change ≈ 6.91pp

mi_percentage
    max event-rate change ≈ 5.65pp

original_upb
    max event-rate change ≈ 5.25pp

original_loan_term
    max event-rate change ≈ 4.15pp

original_interest_rate
    max event-rate change ≈ 4.10pp
```

This was particularly informative for DTI and UPB.

Some of these features had relatively modest distribution drift while their relationship with delinquency changed substantially.

Conceptually:

```text
Feature distribution
        ↓
relatively stable

BUT

Feature → delinquency relationship
        ↓
changed substantially
```

Therefore, the problem was not simply that the model was seeing different feature values.

The same feature values could correspond to different levels of future risk in different vintages.

This is a much deeper form of temporal instability.

---

# 9. Why Relationship Drift Matters

Suppose the model learned something like:

```text
DTI = 40%
        ↓
Historical delinquency risk = X
```

If later vintages show:

```text
DTI = 40%
        ↓
Later-vintage delinquency risk = significantly different
```

then the model's historical relationship is no longer fully representative.

The model can still correctly learn:

```text
Higher DTI → generally higher risk
```

but the absolute risk associated with a given DTI can change.

That distinction became central to the V1 diagnosis.

---

# 10. We Then Tested Temporal Retraining

The next hypothesis was straightforward:

> Maybe V1 simply became stale because it was trained on older vintages.

We therefore tested:

```text
Train:
2015
2016
2017
2018

Test:
2019 OOT
```

The existing V1 model had effectively been trained using the earlier development period.

The temporal experiment allowed the model to learn from 2018 before being evaluated on 2019.

The purpose was **not** to tune hyperparameters.

The purpose was to isolate one question:

> **Does adding a recent vintage solve the OOT deterioration?**

---

# 11. Temporal Retraining Did Help One Part of the Problem

The temporal model's average predicted probability on 2019 moved much closer to the actual event rate.

Approximately:

```text
2019 actual event rate       ≈ 4.65%
Temporal model average PD    ≈ 5.08%
```

This was useful evidence.

It showed that adding recent information helped the model understand the **higher baseline risk environment**.

In other words:

```text
Old training environment
        ↓
risk level underestimated

Add 2018
        ↓
model adapts to newer risk level
```

So temporal retraining was not completely useless.

It improved the probability-level adaptation.

---

# 12. But Temporal Retraining Did Not Solve V1

The problem was that the core OOT deterioration remained.

The temporal model produced approximately:

```text
ROC-AUC          ≈ 0.696
Calibration slope ≈ 0.599
Calibration intercept ≈ -1.13
```

The calibration slope remained far from the ideal value of 1.

The discrimination remained materially below the validation level.

Therefore:

```text
Temporal retraining
        ↓
Improved average PD
        ↓
BUT
        ↓
Did not recover discrimination
        ↓
Did not recover calibration slope
```

This was a critical result.

It meant that the problem was not simply:

> "The model needs more recent training data."

Recent data helped with the overall risk level, but it did not restore the underlying predictive relationship.

---

# 13. The Calibration Bins Confirmed the Problem

The temporal retrained model's 2019 calibration bins showed increasing overprediction at higher risk levels.

Approximately:

| Average predicted PD | Actual event rate |
|---:|---:|
| 0.64% | 1.15% |
| 1.45% | 2.56% |
| 3.23% | 4.41% |
| 7.13% | 6.48% |
| 13.82% | 9.42% |
| 27.29% | 14.04% |
| 55.30% | 33.33% |

The important pattern is:

```text
Lower predicted risk
        ↓
Reasonably close / manageable differences

Higher predicted risk
        ↓
Increasing overprediction
```

This is consistent with the calibration slope around 0.60.

The model was effectively **too extreme in its probability estimates** in the later environment.

---

# 14. Yet V1 Still Had Useful Ranking Power

This is why we are not throwing V1 away.

The 2019 risk deciles still showed meaningful ordering.

Approximately:

```text
Bottom decile
Actual event rate ≈ 0.8%

Top decile
Actual event rate ≈ 11.8%
```

The top 10% of the population captured approximately:

```text
25.4% of all observed events
```

That is substantially better than random selection.

Therefore V1 still answers a useful question:

> **Which loans are relatively more risky?**

What it struggles with is:

> **What is the stable absolute probability of serious delinquency for this loan in a future environment?**

This distinction is extremely important.

---

# 15. Final Diagnosis of V1

After all of the experiments, we can describe the V1 limitation as follows.

### V1 has predictive signal

The model can rank loans.

### V1 has temporal instability

Its performance deteriorates materially OOT.

### The population changes

There is evidence of feature-distribution drift.

### The feature-to-target relationships also change

This is more important.

The same feature values do not necessarily imply the same future delinquency risk across vintages.

### Static calibration cannot solve the issue

A calibration mapping learned historically does not remain valid later.

### Recent-vintage retraining only partially solves it

It improves the overall risk level but does not restore discrimination or calibration slope.

Therefore:

> **The fundamental limitation is the static origination-only formulation itself, rather than simply the choice of calibration method, threshold, or training vintage.**

---

# 16. Why We Are Moving to V2

V1 treats the loan as a snapshot:

```text
Origination
     ↓
Static feature vector
     ↓
24-month binary outcome
```

But a mortgage is inherently longitudinal.

The loan evolves:

```text
Origination
    ↓
Month 1
    ↓
Month 2
    ↓
Month 3
    ↓
...
    ↓
Month 24+
```

Its risk can change throughout this trajectory.

The borrower's current payment behaviour, delinquency state, loan age, and other evolving characteristics can provide information that simply does not exist at origination.

V1 intentionally ignores this information because it is an origination-time model.

That design decision is exactly what we now want to challenge in V2.

---

# 17. The Conceptual Shift to V2

V1 asks:

```text
Given X at origination,
what is P(event within 24 months)?
```

V2 asks something closer to:

```text
Given the loan's state at month t,
what is P(event at month t | survived to month t)?
```

This is the **hazard** concept.

The modelling grain becomes:

```text
Loan × Month
```

rather than:

```text
Loan
```

Conceptually:

```text
Loan A

Month 1 → no event → risk
Month 2 → no event → risk
Month 3 → no event → risk
Month 4 → no event → risk
...
Month 18 → event
```

Instead of assigning one static 24-month probability to the loan, V2 can model how risk evolves through time.

---

# 18. Why Start With a Discrete-Time Hazard Model

The Freddie Mac performance data is naturally observed at monthly intervals.

That makes discrete-time hazard modelling a natural first V2 formulation.

The model can estimate:

\[
h_{i,t}
=
P(T_i=t \mid T_i \geq t, X_{i,t})
\]

where:

- `i` = loan
- `t` = month
- `T` = event time
- `X` = current loan state/features
- `h` = monthly hazard

This gives us a monthly probability of event conditional on the loan having survived up to that point.

From those monthly hazards, we can derive cumulative event probabilities over time.

---

# 19. Why This Could Address the V1 Problem

V1 effectively assumes:

```text
Risk = f(origination state)
```

V2 can model:

```text
Risk_t =
    f(
        origination state,
        loan age,
        current loan state,
        historical performance,
        time-varying features
    )
```

This gives the model the ability to adapt to the evolving condition of the loan.

That is particularly relevant given the V1 finding that:

> The relationship between loan characteristics and future delinquency changes over time.

V2 does not magically eliminate macroeconomic or structural drift, but it gives the model a much richer representation of the loan's current state and risk trajectory.

---

# 20. V2 Is a Change in Problem Formulation, Not Just Algorithm

This is perhaps the most important conceptual distinction.

We are **not** doing:

```text
V1:
XGBoost

V2:
Bigger XGBoost
```

We are doing:

```text
V1:
Static loan-level prediction
        ↓
24-month binary outcome


V2:
Longitudinal loan-month prediction
        ↓
Time-dependent event probability
```

The algorithm is secondary.

The important change is the **statistical formulation of the problem**.

---

# 21. V1 and V2 Will Coexist

V2 should not replace V1 immediately.

The architecture should support both:

```text
                     Shared data
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
             V1                    V2
       Origination model     Time-dependent model
              │                     │
              ▼                     ▼
         Loan-level             Loan-month
              │                     │
              └──────────┬──────────┘
                         ▼
                  Shared evaluation
```

This gives us a clean benchmark.

V1 answers:

> **What can we predict from origination information alone?**

V2 answers:

> **Can we improve stability and usefulness by modelling the evolution of loan risk through time?**

---

# 22. What V2 Is Not Supposed to Do Yet

We should avoid immediately making V2 overly complicated.

The first V2 experiment should be controlled.

The initial objective is:

1. Build the correct loan-month modelling grain.
2. Define event and censoring logic correctly.
3. Build a discrete-time hazard baseline.
4. Use appropriate time-based validation.
5. Evaluate OOT performance.
6. Compare the result against V1.
7. Only then add additional complexity.

We should not immediately introduce every possible behavioural feature, interaction, algorithm, or survival technique.

The first question is simply:

> **Does explicitly modelling time improve the stability of mortgage credit-risk prediction?**

---

# 23. What Success Looks Like

V2 does not necessarily need to beat V1 on every metric.

We particularly want to see whether it improves:

- OOT discrimination;
- probability calibration;
- stability across loan age;
- temporal robustness;
- cumulative event-risk estimation;
- risk ranking across later vintages.

For V1, we focused heavily on:

```text
ROC-AUC
PR-AUC
Brier
Log loss
Calibration slope
Calibration intercept
Risk deciles
```

V2 should retain appropriate comparable metrics while adding time-dependent evaluation.

---

# 24. Why We Are Parking V1 Now

At this point, continuing to tune V1 would mostly be trying increasingly complicated solutions around a static formulation whose limitation we have already identified.

The investigation has covered:

```text
V1 baseline
    ↓
LR vs XGBoost
    ↓
Calibration
    ↓
Temporal drift
    ↓
Temporal retraining
```

The evidence consistently points toward the same conclusion:

> **V1 has useful signal, but its static origination-only representation does not remain sufficiently stable across time.**

Therefore, V1 is being **parked as a baseline**, not discarded.

Threshold optimization can still be revisited as a business decision-analysis exercise, but it is not necessary to continue the core modelling investigation.

---

# 25. Final V1 → V2 Reasoning

The entire decision can be summarized as:

```text
V1 has useful predictive signal
        ↓
Strong validation ranking
        ↓
OOT performance deteriorates
        ↓
Calibration deteriorates
        ↓
Static calibration does not fix OOT
        ↓
Feature distributions change over time
        ↓
Feature → target relationships also change
        ↓
Temporal retraining improves baseline risk
but does not restore discrimination/calibration
        ↓
Therefore the problem is deeper than
simple model aging or calibration
        ↓
Investigate a time-dependent formulation
        ↓
V2: Discrete-Time Hazard
        ↓
Compare against Survival Model
```

---

# 26. Final Conclusion

**V1 is a useful baseline, not a failure.**

It demonstrates that origination-time information contains meaningful predictive signal and can rank loans by relative risk.

However, the V1 model's absolute risk estimates and discrimination are not stable enough across later vintages. The investigations showed that this is driven not only by changes in the feature distribution, but also by changes in the relationship between loan characteristics and subsequent delinquency.

Calibration cannot fully solve that problem, and retraining with a newer vintage only partially improves it.

Therefore, the next logical step is to change the modelling formulation.

**V2 will investigate time-dependent credit-risk modelling, beginning with a discrete-time hazard model and subsequently comparing it with a survival-model approach.**

The goal is not to assume that V2 will automatically be better.

The goal is to test whether explicitly modelling:

```text
loan state
    +
time
    +
performance trajectory
```

provides a more stable representation of mortgage credit risk than the static origination-only formulation used by V1.

**V1 remains fully supported as the benchmark against which V2 will be evaluated.**


---

# 10. The Deeper Structural Limitation of V1

The temporal-drift findings point to a broader modelling limitation that is important to distinguish from ordinary model performance issues.

At origination, the model only observes a relatively narrow set of borrower and loan characteristics:

```text
credit score
DTI
LTV / CLTV
loan amount
interest rate
property / state
loan purpose
mortgage structure
etc.
```

These variables contain real predictive signal. However, an important selection effect occurs **before the V1 model ever sees the loan**.

The mortgage has already passed the lender's underwriting and origination process.

Conceptually:

```text
Broad applicant population
        ↓
Underwriting / eligibility / credit policy
        ↓
Sanctioned mortgage population
        ↓
V1 model observes the loan
```

Therefore, V1 is not trying to distinguish an unrestricted population containing extremely risky and extremely safe applicants.

It is trying to distinguish future defaulters from non-defaulters **within a population that has already been screened for acceptable credit risk**.

This can compress the range of observable financial characteristics at origination.

The consequence is important:

> **Origination variables describe baseline financial risk, but they do not fully describe how the loan will behave after origination.**

---

# 11. Why Similar Origination Profiles Can Still Produce Different Outcomes

Consider two mortgages with broadly similar origination characteristics:

```text
Loan A
credit score ≈ 710
DTI ≈ 32%
LTV ≈ 80%

Loan B
credit score ≈ 715
DTI ≈ 30%
LTV ≈ 78%
```

At origination, both loans can appear relatively similar from the information available to V1.

However, their future trajectories can diverge:

```text
                Origination
                     │
           ┌─────────┴─────────┐
           │                   │
        Loan A             Loan B
        looks fine         looks fine
           │                   │
           └─────────┬─────────┘
                     ↓
                 Month 12
                     │
           ┌─────────┴─────────┐
           │                   │
       payment stress      stable payments
       recent delinquency  no delinquency
       financial pressure  stable position
           │                   │
           ↓                   ↓
        higher risk         lower risk
```

The information that separates these two outcomes may emerge **after origination**.

Examples of potentially informative post-origination states include:

```text
current delinquency status
recent delinquency behaviour
number / severity of prior delinquencies
months since delinquency
payment behaviour over time
loan age
balance trajectory
other performance-derived indicators
```

These are fundamentally different from static origination characteristics.

V1 deliberately cannot use this information because its prediction point is origination.

---

# 12. Static Baseline Risk vs Dynamic Evolving Risk

This leads to a useful conceptual distinction.

## V1 estimates baseline risk

V1 is effectively trying to learn:

```text
Risk_0 = f(origination characteristics)
```

where `Risk_0` is the risk inferred from the state of the mortgage at origination.

This is valuable, but limited.

## V2 can investigate evolving risk

V2 can investigate:

```text
Risk_t =
    f(
        origination characteristics,
        loan age,
        observed performance through t,
        evolving loan state
    )
```

The distinction is therefore:

```text
V1
────
"What did the loan look like when it was originated?"

V2
────
"What does the loan look like now, given what has happened since origination?"
```

That is the fundamental conceptual reason for moving toward a time-dependent formulation.

---

# 13. Why This Explains the V1 Ceiling

The V1 experiments can now be interpreted in light of this structural limitation.

### Strong validation performance

Origination characteristics contain meaningful baseline signal.

### OOT deterioration

The mapping from origination characteristics to future delinquency is not sufficiently stationary across later vintages.

### Calibration instability

The probability scale changes as the underlying environment and feature-to-risk relationships change.

### Feature relationship drift

Even comparable DTI, UPB, MI percentage, loan-term, and interest-rate groups can have materially different delinquency rates across vintages.

### Temporal retraining only partially helps

Recent data can help the model recognize a newer baseline risk environment, but it cannot recover information about post-origination behaviour that was never present in the feature set.

This is why V1 eventually reaches a practical ceiling.

The ceiling is not simply a matter of finding the perfect algorithm.

It is partly an **information-set limitation**:

> **A static snapshot at origination cannot directly observe the future behavioural trajectory that causes otherwise similar sanctioned loans to diverge.**

---

# 14. Why V2 Is a Natural Next Step

V2 therefore should not be viewed as:

```text
"V1 failed, so let's train another model."
```

The correct interpretation is:

```text
V1
↓
Establishes baseline origination risk
↓
Reveals temporal instability
↓
Shows that static origination information is insufficient
       for fully stable future-risk estimation
↓
V2
↓
Adds the dimension that V1 deliberately excluded:
time + observed loan behaviour
```

This gives V2 a clear modelling hypothesis:

> **Post-origination performance information contains incremental information about future delinquency that cannot be recovered from origination characteristics alone.**

V2 can then test this hypothesis directly.

---

# 15. Important Qualification

We should not overstate the argument.

It would be incorrect to say:

> "All sanctioned mortgages look the same."

They do not.

There is meaningful variation in:

- credit score;
- DTI;
- LTV / CLTV;
- loan amount;
- interest rate;
- geography;
- purpose;
- property characteristics;
- mortgage structure.

The V1 validation results demonstrate that this variation contains real predictive information.

The stronger and more defensible statement is:

> **Underwriting and origination selection constrain the observable risk range, while the remaining variation in future delinquency can increasingly depend on post-origination events and behavioural trajectories that are not observable at origination.**

That is the specific limitation V2 is designed to investigate.
