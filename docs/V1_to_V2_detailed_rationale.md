# V1 → V2: Detailed Model Evolution Rationale

> **Status:** Historical rationale for the transition from an origination-only model to the implemented behavioural point-in-time framework. The current V2 implementation is not a full discrete-time hazard/survival model; it is a horizon-specific point-in-time behavioural model. V3 remains a separate research track.

## 1. Executive summary

V1 was designed as a static origination-time mortgage credit-risk model.

Its conceptual question was:

```text
Given information available at origination,
how likely is the loan to experience serious delinquency
within the defined future horizon?
```

V1 demonstrated meaningful predictive signal and useful risk ordering.

The main concern was temporal robustness. Later populations exhibited changes in both feature distributions and feature-to-outcome relationships. Calibration and more recent-vintage retraining could partially adjust probability levels but did not fully restore the original discrimination/calibration behaviour.

This motivated the transition to V2.

---

## 2. What changed conceptually

### V1

```text
Origination
    ↓
Static feature vector
    ↓
Future delinquency outcome
    ↓
Loan-level model
```

### V2

```text
Loan history
    ↓
Observation at age t
    ↓
Current + historical behavioural features
    ↓
Forward 12-month outcome
    ↓
Point-in-time model
```

The key change is not simply a different algorithm. It is the **information set** used to represent the loan.

---

## 3. Why origination-only modelling is limited

A mortgage is longitudinal.

Its risk-relevant state changes through time:

```text
origination
    ↓
payment behaviour
    ↓
delinquency history
    ↓
balance trajectory
    ↓
rate/affordability changes
    ↓
assistance/modification
    ↓
future event
```

An origination-only model cannot observe these later states.

V2 therefore introduces repeated point-in-time observations.

---

## 4. The V2 formulation

The active V2 observation grain is:

```text
loan_id × observation_age
```

with observation ages:

```text
2, 4, 6, 8, 10, 12
```

At each observation point, predictors use information available through that month.

The target looks forward 12 months:

```text
t + 1 ... t + 12
```

This gives the model access to behavioural information while maintaining a strict temporal boundary.

---

## 5. Why behavioural information matters

Origination variables describe the starting state of the mortgage.

Behavioural variables describe what has actually happened since origination.

Examples include:

- current DPD;
- maximum DPD to date;
- delinquency frequency;
- delinquency recency;
- recent 30/60 DPD counts;
- delinquency streaks;
- DPD trend and acceleration;
- modification/payment-deferral history;
- borrower assistance;
- UPB trajectory;
- rate trajectory.

These variables can distinguish loans that looked similar at origination but have subsequently followed very different paths.

---

## 6. Leakage boundary

The V2 design is explicitly point-in-time:

```text
Information through t
        ↓
    predictors

Information after t
        ↓
      target
```

The forward target is based on serious delinquency or REO during the next 12 months.

Voluntary payoff/maturity without a prior event is handled as a non-event; other incomplete early exits are handled according to the target-observability rules.

---

## 7. From V1 instability to V2

The V1 investigation suggested several distinct phenomena can occur across time:

### Population drift

The distribution of model inputs can change:

```text
P(X)_development
        ≠
P(X)_later
```

### Relationship drift

More importantly, the feature-to-outcome relationship can also change:

```text
P(Y | X)_development
        ≠
P(Y | X)_later
```

A model trained on historical origination relationships can therefore become less representative of a later risk environment.

---

## 8. Why calibration alone is insufficient

Calibration changes the mapping between model scores and probabilities.

Conceptually:

```text
raw score
   ↓
calibration mapping
   ↓
probability
```

This can correct a probability-scale problem.

It cannot, by itself, reconstruct behavioural information that was never present in the original feature vector.

If the conditional relationship itself changes, a historical calibration mapping may not remain valid.

This was an important motivation for expanding the information set rather than relying only on post-hoc calibration.

---

## 9. Why recent-vintage retraining is not the whole solution

Adding newer observations can help a model adapt to the overall level of risk.

However, retraining does not guarantee that a static origination feature set will capture evolving borrower behaviour.

The V2 approach therefore changes the representation itself:

```text
static origination state
        ↓
current + historical behavioural state
```

This is a structural change rather than merely a refresh of training data.

---

## 10. Why V2 uses a 12-month horizon

The active target is:

```text
future_90dpd_12m
```

A 12-month horizon provides a consistent risk window across observation ages and allows the project to compare risk at multiple points in a loan's life.

The horizon is configured rather than hard-coded so that future experiments can change it without rewriting the modelling architecture.

---

## 11. Why a GAM

The V2 model needs to capture nonlinear relationships while remaining interpretable.

A GAM provides:

```text
linear effects
+
selected smooth nonlinear effects
+
controlled interactions
```

This is useful for mortgage risk because relationships such as:

```text
credit score → risk
LTV → risk
DTI → risk
interest rate → risk
delinquency trajectory → risk
```

need not be strictly linear.

The active GAM therefore uses cubic spline effects for selected variables and linear effects by default.

---

## 12. Why controlled interactions

An additive GAM assumes:

```text
effect(x,z) = f(x) + g(z)
```

But mortgage risk can plausibly contain dependencies such as:

```text
credit quality × collateral leverage
behaviour × exposure
affordability × rate pressure
```

V2 therefore allows explicitly configured interactions.

The interaction design is intentionally constrained to avoid turning the GAM into an uncontrolled high-dimensional interaction model.

---

## 13. Current V2 configuration

The active modelling configuration is:

```text
Engine:
    PySpark

Observation ages:
    2, 4, 6, 8, 10, 12

Prediction horizon:
    12 months

Training:
    2015–2020

Test:
    2021

OOT:
    2022

Algorithm:
    GAM

Spline:
    degree 3
    6 quantile-based knots

Interactions:
    enabled

Macro features:
    enabled
```

---

## 14. Evaluation philosophy

V2 should not be judged by a single metric.

The evaluation framework considers:

```text
Discrimination
    ROC-AUC
    PR-AUC
    KS

Probability quality
    Brier
    log loss
    calibration

Risk ranking
    deciles
    capture
    lift

Temporal robustness
    test/OOT stability
```

The primary question is whether behavioural information produces a model that is both useful for ranking and more robust across time.

---

## 15. V2 is not yet a survival model

The original conceptual direction involved discrete-time hazard/survival modelling.

That remains a valid research direction.

However, the implemented V2 model should be described accurately as:

> **a point-in-time behavioural mortgage PD-like model with a 12-month forward outcome.**

It should not be described as a fully implemented survival or competing-risks model unless that architecture is actually introduced.

---

## 16. Relationship to V3

V3 is maintained separately.

Potential V3 research areas include:

- temporal calibration;
- macroeconomic conditioning;
- feature correctness/auditing;
- redundancy and ablation analysis;
- dynamic trajectory modelling;
- alternative temporal model formulations.

The current V2 documentation should describe the implemented pipeline rather than future V3 proposals.
