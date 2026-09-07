# Trajectory-Conditioned Dynamic Mortgage PD — Research / Experimental Model Specification

> **Status: Experimental / future research.** This document is not part of the active production-style V2 modelling path.

## Core idea

At observation month `t`, use only the loan's history through `t`, identify historical loan-observations with similar behavioural trajectories, inspect what happened to those loans during `t+1` through `t+12`, and estimate current 12-month PD from their subsequent outcomes.

The research question is:

> Can historical mortgage loans with similar observed behavioural trajectories have materially similar conditional future default distributions beyond current-state covariates?

---

## Production-compatible boundary

The experiment must preserve the existing point-in-time information boundary:

```text
history through t → trajectory representation
history after t  → outcome only
```

No future observation may enter trajectory construction or neighbour retrieval.

The existing target/eligibility framework should be reused rather than creating a second target definition.

---

## Stage 1 — Interpretable similarity prototype

1. Build a compact behavioural trajectory representation.
2. Normalise using training data only.
3. Define a weighted distance.
4. Retrieve historical neighbours.
5. Observe their subsequent 12-month outcomes.
6. Estimate a distance-weighted PD.

Conceptually:

```text
P_hat_i = Σ_j w_ij Y_j / Σ_j w_ij
```

Candidate distance components include:

- DPD trajectory;
- UPB trajectory;
- LTV / estimated LTV;
- rate;
- delinquency frequency/severity;
- recency/recovery;
- modification/payment-deferral history;
- assistance/disaster history;
- event-history mismatch.

---

## Stage 2 — Existing trajectory features

The first prototype should test whether the current engineered trajectory features already provide most of the available trajectory signal.

Candidate representation:

- `current_delinquency_streak`
- `max_delinquency_streak_12m`
- `months_since_last_30dpd`
- `months_since_last_60dpd`
- `dpd_trend_6m`
- `dpd_acceleration_6m`
- `delinquency_intensity_change_6m`
- `dpd_severity_change_6m`
- `delinquency_episode_count`
- `relapse_after_current`

The objective is to establish incremental value before introducing more complex sequence methods.

---

## Stage 3 — Hybrid model

Generate neighbour-derived features such as:

- weighted neighbour default rate;
- unweighted neighbour default rate;
- serious-event share;
- distance to nearest defaulting trajectory;
- distance to nearest non-defaulting trajectory;
- mean/median neighbour distance;
- outcome rates for multiple values of `k`.

Then compare:

```text
existing state + trajectory features
                vs
existing state + trajectory features + similarity features
```

Candidate downstream models include the current XGBoost/GAM benchmark families.

---

## Stage 4 — Learned trajectory embedding

Only if the interpretable prototype demonstrates incremental signal should the project consider learned sequence embeddings.

A candidate representation is:

```text
z(i,t) = f(X(i,1:t))
```

where `f` could be implemented with:

- LSTM;
- GRU;
- temporal CNN;
- Transformer.

The embedding would then be used for trajectory retrieval.

This should not be the first implementation.

---

## Behavioural archetypes

Potential qualitative patterns include:

```text
stable → stable → payoff

isolated 30 DPD → recovery

repeated 30 DPD → increasing severity

30 → 60 → 90

delinquency → modification → recovery

delinquency → recovery → relapse

gradual deterioration
```

These are hypotheses for analysis, not predefined production classes.

---

## Scale considerations

The mortgage panel can contain tens of millions of observations.

An all-pairs trajectory comparison is therefore infeasible.

A practical sequence is:

```text
prototype on sample
      ↓
compact trajectory representation
      ↓
candidate reduction/indexing
      ↓
approximate/distributed retrieval
      ↓
exact distance on candidates
```

The ANN/indexing technology should be selected only after the representation has demonstrated useful signal.

---

## Validation

Keep a frozen benchmark.

Compare:

```text
A. Current-state benchmark

B. Benchmark + existing trajectory features

C. Pure trajectory-similarity PD

D. GAM/XGB + similarity-derived features
```

Evaluate:

- ROC-AUC;
- PR-AUC;
- KS;
- log loss;
- Brier;
- calibration/ECE;
- observed-to-expected ratios;
- top-5/10/20% capture and lift;
- temporal stability;
- segment stability.

Do not optimise solely for AUC.

---

## Guardrails

The experiment must enforce:

- no future observations in scoring trajectories;
- no future information in neighbour representations;
- temporal train/test/OOT separation;
- training-only normalisation;
- proper probability calibration;
- awareness that repeated observations from the same loan are correlated;
- an untouched benchmark for comparison.

---

## Recommended progression

```text
1. Similarity prototype
        ↓
2. Test whether similar trajectories
   have similar future outcomes
        ↓
3. Compare with current-state /
   engineered-trajectory baselines
        ↓
4. Add similarity features to GAM/XGB
        ↓
5. If incremental signal exists,
   learn trajectory embeddings
        ↓
6. Only then consider
   LSTM / GRU / Transformer
```

---

## Success criteria

Strong evidence would include:

- OOT PR-AUC improvement;
- improved or competitive top-k capture/lift;
- improved calibration;
- temporal stability;
- economically interpretable historical analogues;
- neighbour explanations that make sense to a credit-risk reviewer.

A null result is also useful if the existing trajectory features already capture most of the available information.

---

## One-sentence definition

> **Can we build a dynamic mortgage PD model that evaluates a loan not only by what it looks like today, but by which historical loan trajectories it most closely resembles—and what happened to those loans afterward?**

---

## Relationship to the active V2 pipeline

This research track should reuse:

- the monthly loan panel;
- the existing target;
- the observation-age framework;
- the current point-in-time feature philosophy;
- the engineered trajectory features.

It is a research extension, not a replacement for the active GAM benchmark.

The first practical question remains:

> Does historical trajectory similarity add incremental information beyond current state plus engineered trajectory features?

Scale and deep sequence modelling should be addressed only after that question is answered.
