# Trajectory-Conditioned Dynamic Mortgage PD — Research / Experimental Model Specification

This document captures the complete context of the proposed trajectory-similarity mortgage PD experiment.

## Core idea
At observation month t, use only the loan's history through t, find historical loan-observations with similar behavioral trajectories, inspect what happened to those loans during t+1 through t+12, and estimate current 12-month PD from their subsequent outcomes.

**Research question:** Can historical mortgage loans with similar observed behavioral trajectories have materially similar conditional future default distributions beyond current-state covariates?

## Production setup
- History available through t only.
- Target: serious event in t+1..t+12.
- Reuse existing target/eligibility rules.
- Initial observation ages: [3,6,9,12].
- Never use future information in trajectory representation or neighbor retrieval.

## Trajectory candidates
Monthly DPD, UPB, LTV/estimated LTV, rate, modification/payment-deferral history, assistance/disaster history, loan age/maturity, plus current state.

## Stage 1: interpretable similarity prototype
1. Build compact trajectory representation.
2. Normalize using training data only.
3. Define weighted distance.
4. Retrieve k historical neighbors.
5. Observe their next-12-month outcomes.
6. Estimate distance-weighted PD.

Conceptually:
P_hat_i = sum_j w_ij Y_j / sum_j w_ij.

Candidate distance components: DPD, UPB, LTV, rate, delinquency frequency/severity, recency/recovery, event-history mismatch.

## Stage 2: existing trajectory features
Use:
- current_delinquency_streak
- max_delinquency_streak_12m
- months_since_last_30dpd
- months_since_last_60dpd
- dpd_trend_6m
- dpd_acceleration_6m
- delinquency_intensity_change_6m
- dpd_severity_change_6m
- delinquency_episode_count
- relapse_after_current

First test whether similarity using this representation adds value.

## Stage 3: hybrid model
Generate neighbor-derived features:
- weighted/unweighted neighbor default rate
- serious-event share among neighbors
- distance to nearest defaulting/non-defaulting trajectory
- mean/median neighbor distance
- outcome rates for multiple k

Feed these to XGB/GAM and test incremental value beyond current state + trajectory features.

## Stage 4: learned trajectory embedding
If signal exists, learn z(i,t)=f(X(i,1:t)) using LSTM/GRU/temporal CNN/Transformer, then retrieve similar trajectories in embedding space. Do not jump here initially.

## Behavioral archetypes to investigate
Stable→stable→payoff; isolated 30→recovery; repeated 30→increasing severity; 30→60→90; delinquency→modification→recovery; relapse; gradual deterioration.

These are hypotheses, not predefined classes.

## Scale
Tens of millions of rows make all-pairs comparison infeasible. Prototype on a sample, compact representations, index/reduce candidates, then use approximate/distributed retrieval and exact distance only for candidates. Choose ANN technology after validating the representation.

## Validation
Frozen XGB remains control. Compare:
A. frozen/current-state XGB
B. XGB + existing trajectory features
C. pure trajectory-similarity PD
D. XGB/GAM + similarity features

Metrics: ROC-AUC, PR-AUC, log loss, Brier, ECE/MCE, O/E, KS, top-5/10/20% capture/lift, temporal and segment stability.

Do not optimize for AUC alone.

## Guardrails
No future observations in scoring trajectories; no future information in neighbor representations; temporal splits; proper calibration; repeated observations treated as correlated; frozen benchmark untouched.

## Literature
Key related streams:
- Peng & Lessmann (2026): Freddie Mac dynamic/longitudinal survival and behavioral trajectory modeling.
- Medina-Olivares et al. (2023): joint longitudinal + discrete survival models in credit scoring.
- Landmarking + ML survival: dynamic prediction using history available at each prediction point.
- Le, Ku & Jun (2021): sequence-based clustering for long-term credit risk.
- Dynamic-DeepHit and sequence-survival literature.
- Case-based/memory-based credit risk.

## Recommended progression
1. Similarity prototype.
2. Test whether similar trajectories have meaningful future outcome similarity.
3. Compare against current-state and trajectory-feature baselines.
4. Add similarity-derived features to XGB/GAM.
5. If incremental signal exists, learn trajectory embeddings.
6. Only then consider LSTM/GRU/Transformer.

## Success
Strong evidence: OOT PR-AUC/top-k improvement, competitive/improved calibration, temporal stability, economically interpretable neighbors, explainable historical analogues.

A null result is valuable if existing trajectory features already capture most of the signal.

## One-sentence definition
**Can we build a dynamic mortgage PD model that evaluates a loan not only by what it looks like today, but by which historical loan trajectories it most closely resembles—and what happened to those loans afterward?**

## Future implementation context
This is a separate experimental research track, not a replacement for frozen XGB. The first implementation is an interpretable similarity prototype. Temporal leakage prevention is non-negotiable. Reuse the current monthly panel, target, observation-age framework and trajectory features. The first practical question is whether historical trajectory similarity adds incremental information beyond current state + engineered trajectory features. Solve scale only after proving signal. The end goal is production-compatible monthly scoring.
