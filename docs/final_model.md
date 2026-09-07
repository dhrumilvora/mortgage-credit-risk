# Final Model Specification

## 1. Model Status

This document specifies the final fitted model artefact produced by the mortgage credit-risk modelling framework.

| Specification       | Final model                      |
| ------------------- | -------------------------------- |
| Model version       | `gam_macro_3_horizon`            |
| Algorithm           | Generalized Additive Model (GAM) |
| Training vintages   | 2015–2020                        |
| Test vintage        | 2021                             |
| Out-of-time vintage | 2022                             |
| Observation ages    | 3, 6, 9, 12 months               |
| Target              | `future_90dpd`                   |
| Prediction horizon  | 3 months                         |
| Calibration         | Beta calibration                 |
| Decision threshold  | 0.10                             |

The wider modelling framework supports multiple interchangeable model architectures:

* Logistic Regression
* GAM
* XGBoost
* LightGBM

The model documented here is the **final fitted GAM configuration**.

---

## 2. Modelling Objective

The model estimates the probability that an eligible mortgage observation experiences a serious delinquency event during the **three months following the observation point**.

The modelling grain is:

```text
loan_id × observation_age
```

with the final observation ages:

```text
3, 6, 9, 12 months
```

Each observation represents a mortgage's risk state at a defined point in its lifecycle.

The model uses information available at that point to estimate near-term serious-delinquency risk.

---

## 3. Target Definition

The final target is:

```text
future_90dpd
```

The target represents whether a qualifying 90+ DPD event occurs within the **three-month prediction horizon** following the observation point.

Conceptually:

```text
Observation at t
        │
        ▼
Future 3-month performance
        │
        ├── 90+ DPD event
        │
        └── No 90+ DPD event
```

The target is therefore explicitly forward-looking, while all predictors are constructed using information available at or before the observation point.

The information boundary is:

```text
                         OBSERVATION POINT
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Information ≤ t                  Information > t
                │                               │
                ▼                               ▼
             FEATURES                         TARGET
                                                │
                                                ▼
                                         Next 3 months
```

---

## 4. Observation Framework

The final modelling population is constructed at:

```text
loan_id × observation_age
```

with:

```text
observation_age ∈ {3, 6, 9, 12}
```

A mortgage can therefore contribute multiple point-in-time observations:

```text
Loan A × age 3
Loan A × age 6
Loan A × age 9
Loan A × age 12
```

These observations represent different behavioural and financial states of the same mortgage.

The observation-age framework allows the model to assess risk at multiple stages of the mortgage lifecycle.

---

## 5. Prediction Horizon

The final model is specifically a **3-month horizon model**.

For an observation at time `t`:

```text
t
│
├── Month +1
├── Month +2
└── Month +3
       │
       ▼
future_90dpd
```

This distinction is important.

The project framework may support other prediction horizons, but the final model represented by:

```text
gam_macro_3_horizon
```

is the **3-month serious-delinquency model**.

The performance results documented below therefore correspond to the 3-month target definition and should not be interpreted as performance for a 12-month prediction horizon.

---

## 6. Leakage Control

The feature framework is designed around a strict point-in-time information boundary.

Predictors may use information available at or before the observation point, including:

* origination characteristics;
* current loan state;
* historical performance;
* recent behavioural history;
* behavioural trajectory;
* macroeconomic information available for the relevant period.

Predictors must not use performance information occurring after the observation point.

Conceptually:

```text
                         TIME
──────────────────────────────────────────────────────►

 Historical information
          │
          │
          ▼
──────────●───────────────────────────────────────────
       observation t
          │
          └──────────────► future 3-month performance
                                      │
                                      ▼
                               future_90dpd
```

The same information boundary applies to every supported model architecture.

Training-only transformations must likewise be fitted using the training population and reused for validation, test and OOT populations.

---

## 7. Temporal Design

The final model uses chronological vintage separation:

```text
2015–2020
    │
    └── Training population
            │
            └── configured 20% yearly/chronological validation split

2021
    │
    └── Test population

2022
    │
    └── Out-of-time evaluation
```

The OOT population is reserved for assessing temporal robustness.

It is not used to fit:

* preprocessing parameters;
* model coefficients;
* spline knots;
* GAM interactions;
* calibration parameters.

The temporal design provides a test of whether relationships learned from earlier mortgage vintages remain useful on a later population.

---

## 8. Feature Framework

The final model combines structural, behavioural, trajectory and macroeconomic information.

### 8.1 Origination Features

The feature framework includes structural characteristics available at origination, including:

* number of borrowers;
* mortgage insurance percentage;
* original UPB;
* credit score;
* original DTI;
* original LTV;
* original CLTV;
* loan purpose;
* occupancy;
* property type;
* origination channel;
* property state.

These features provide the baseline structural risk profile of each mortgage.

---

### 8.2 Current Loan-State Features

Current-state features describe the mortgage at the observation point.

Examples include:

* current actual UPB;
* current interest rate;
* estimated LTV;
* calculated loan age;
* remaining months to legal maturity;
* non-interest-bearing UPB percentage;
* interest-bearing UPB percentage;
* current delinquency state.

These features capture the mortgage's contemporaneous financial and performance state.

---

### 8.3 Lifetime Behavioural Features

Lifetime behavioural features summarise performance history through the observation point.

Examples include:

* maximum DPD to date;
* delinquency months to date;
* months since last delinquency;
* ever modified;
* ever payment deferred;
* ever borrower assistance;
* ever disaster delinquency;
* delinquency episode count;
* relapse after current delinquency.

These features capture persistent borrower and servicing behaviour.

---

### 8.4 Recent Behavioural Features

Recent behavioural features capture performance over shorter historical windows.

The final feature contract includes 12-month and 3-month measures covering:

* 30 DPD counts;
* 60 DPD counts;
* maximum DPD;
* modification counts;
* payment-deferral counts;
* borrower-assistance counts;
* disaster-delinquency counts;
* rate-step counts.

Recent behaviour is particularly relevant for a short-horizon prediction problem because recent deterioration can provide a strong signal of near-term risk.

---

### 8.5 Behavioural Trajectory Features

The model explicitly captures changes in borrower performance over time.

Important trajectory features include:

```text
current_delinquency_streak
max_delinquency_streak_12m
months_since_last_30dpd
months_since_last_60dpd
dpd_trend_6m
dpd_acceleration_6m
delinquency_intensity_change_6m
dpd_severity_change_6m
```

These features allow the model to distinguish between different behavioural trajectories:

```text
Stable
   │
   ├── Improving
   │
   ├── Deteriorating
   │
   └── Persistently delinquent
```

This is a central component of the behavioural modelling approach.

---

### 8.6 Exposure and Balance Dynamics

The final numerical feature contract also includes:

* UPB percentage change from origination;
* rate change from origination;
* current actual UPB;
* estimated LTV;
* remaining term;
* current interest rate.

These features provide information on changing borrower exposure and financial pressure.

---

### 8.7 Macroeconomic Features

The final model incorporates macroeconomic information:

* unemployment rate;
* 30-year mortgage rate;
* purchase-only house price index;
* Federal Funds rate.

These variables provide contextual information about the economic and financing environment surrounding each observation.

---

## 9. Categorical Features

The final categorical feature contract includes:

```text
property_type
occupancy_status
loan_purpose
channel
super_conforming_flag
harp_indicator
property_state
modification_flag
interest_rate_step_indicator
payment_deferral_flag
delinquency_due_to_disaster
borrower_assistance_plan
```

Categorical variables are transformed into model-compatible representations through the fitted preprocessing pipeline.

---

## 10. Final Feature Count

The final fitted model was trained using:

```text
66 model features
```

The feature set combines:

```text
Origination
+
Current state
+
Lifetime behaviour
+
Recent behaviour
+
Behavioural trajectory
+
Exposure dynamics
+
Macroeconomics
+
Categorical attributes
```

No separately configured engineered-feature list was required in the final training configuration.

---

## 11. Preprocessing

The final model uses a persisted preprocessing pipeline.

The conceptual processing sequence is:

```text
Point-in-time model population
            ↓
Training-fitted preprocessing
            ↓
Numerical missing-value treatment
            ↓
Categorical indexing / encoding
            ↓
GAM-specific representation
```

The preprocessing state is fitted on the training population.

The same fitted transformation is subsequently applied to:

* validation;
* test;
* OOT;
* scoring populations.

The transformation is therefore not independently refitted on later populations.

---

## 12. Model Architecture

The broader project uses a configuration-driven model-selection layer.

Supported architectures are:

```text
Logistic Regression
GAM
XGBoost
LightGBM
```

Conceptually:

```text
                    MODEL SELECTOR
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 Logistic Regression    GAM             XGBoost
                         │
                         │
                         ▼
                      LightGBM
```

The final persisted model uses:

```text
GAM
```

The upstream point-in-time feature population and temporal evaluation framework remain common across model choices.

---

## 13. Final GAM Architecture

The final fitted model is a Generalized Additive Model implemented through a transformed feature representation followed by logistic regression.

The execution path is:

```text
Point-in-time features
        ↓
Training-fitted preprocessing
        ↓
GAM preparation
        ↓
Linear effects
+
Spline effects
+
Categorical effects
+
Configured interactions
        ↓
VectorAssembler
        ↓
Logistic Regression
        ↓
Raw 3-month serious-delinquency probability
```

Conceptually:

```text
logit(P(future_90dpd))
    =
    intercept
    +
    Σ f_j(X_j)
    +
    Σ h_k(X_a, X_b)
```

where:

* `f_j(.)` represents a main effect;
* `h_k(.,.)` represents a configured interaction.

The additive structure provides nonlinear modelling while retaining an interpretable functional form.

---

## 14. Spline Specification

The active GAM configuration uses:

| Parameter                        |    Value |
| -------------------------------- | -------: |
| Spline degree                    |        3 |
| Number of knots                  |        6 |
| Knot strategy                    | Quantile |
| Default numerical representation |   Linear |

The configured spline variables include:

```text
credit_score
original_dti
original_ltv
original_cltv
estimated_ltv
current_interest_rate
current_actual_upb
remaining_months_to_legal_maturity
dpd_trend_6m
dpd_acceleration_6m
dpd_severity_change_6m
current_delinquency_streak
max_delinquency_streak_12m
months_since_last_30dpd
months_since_last_60dpd
delinquency_intensity_change_6m
unemployment_rate
mortgage_rate_30y
hpi_purchase_only
fed_funds_rate
```

Numerical features not selected for spline transformation use the configured linear representation.

Spline support is learned from the training population.

---

## 15. GAM Interaction Specification

Explicit interactions are enabled.

The final configuration contains 20 configured interaction pairs.

The interactions focus on economically meaningful relationships involving:

### Credit quality and collateral

```text
credit_score × estimated_ltv
credit_score × modification_flag
estimated_ltv × occupancy_status
```

### Delinquency trajectory

```text
dpd_trend_6m × property_type
dpd_trend_6m × dpd_acceleration_6m
dpd_trend_6m × estimated_ltv
dpd_acceleration_6m × estimated_ltv
dpd_trend_6m × credit_score
dpd_acceleration_6m × credit_score
dpd_severity_change_6m × credit_score
delinquency_intensity_change_6m × credit_score
dpd_severity_change_6m × estimated_ltv
delinquency_intensity_change_6m × estimated_ltv
current_delinquency_streak × estimated_ltv
```

### Affordability and collateral

```text
original_dti × credit_score
original_dti × estimated_ltv
original_dti × original_ltv
```

### Interest-rate pressure

```text
current_interest_rate × original_dti
rate_change_from_origination × original_dti
rate_change_from_origination × estimated_ltv
```

The interaction framework is deliberately controlled rather than exhaustively generated.

Detailed interaction methodology is documented separately in:

```text
docs/gam_interactions.md
```

---

## 16. Logistic Regression Backend

The GAM uses logistic regression as its final coefficient-estimation stage.

The final configuration is:

| Parameter             | Value |
| --------------------- | ----: |
| Regularisation        |   1.0 |
| Elastic-net parameter |   0.0 |
| Maximum iterations    | 1,000 |
| Fit intercept         |   Yes |
| Standardisation       |   Yes |

This corresponds to an L2-regularised logistic regression backend without elastic-net mixing.

---

## 17. Training Population

The final fitted model contains:

```text
Training rows:       43,114,736
Training features:   66
Validation rows:     15,827,668
```

The training event rate is:

```text
0.2618%
```

This represents a highly imbalanced serious-delinquency prediction problem.

Consequently, accuracy alone is not an informative measure of model quality.

The evaluation therefore focuses on:

* ROC-AUC;
* PR-AUC;
* KS;
* calibration;
* risk concentration;
* lift;
* threshold performance;
* OOT stability.

---

## 18. Validation Performance

The final model achieved:

| Metric      | Validation |
| ----------- | ---------: |
| ROC-AUC     |     0.8751 |
| PR-AUC      |     0.2199 |
| KS          |     0.6197 |
| Log loss    |     0.0251 |
| Brier score |    0.00464 |
| Precision   |     0.5104 |
| Recall      |     0.1933 |
| F1          |     0.2804 |

The validation event rate was approximately:

```text
0.516%
```

The model demonstrates strong discrimination despite the highly imbalanced target.

The highest-risk population contains a disproportionately large share of observed events.

---

## 19. Out-of-Time Performance

The 2022 population provides the primary OOT evaluation.

| Metric      |     OOT |
| ----------- | ------: |
| ROC-AUC     |  0.8446 |
| PR-AUC      |  0.4012 |
| KS          |  0.6485 |
| Log loss    |  0.0544 |
| Brier score | 0.00998 |

The OOT event rate is approximately:

```text
0.907%
```

The model retains strong discrimination on the later OOT population.

The change from validation to OOT is particularly important because the observed event environment is materially different.

---

## 20. OOT Risk Concentration

The model demonstrates strong concentration of observed events within the highest predicted-risk observations.

### Top 5%

The highest-risk 5% captures approximately:

```text
69.2% of observed events
```

with:

```text
13.83× lift
```

### Top 10%

The highest-risk 10% captures approximately:

```text
72.4% of observed events
```

with:

```text
7.24× lift
```

### Top 20%

The highest-risk 20% captures approximately:

```text
76.8% of observed events
```

with:

```text
3.84× lift
```

The key OOT ranking result is therefore:

```text
10% of population
        ↓
72.4% of events
        ↓
7.24× lift
```

This indicates strong risk concentration and demonstrates that the model is particularly effective as a ranking framework.

---

## 21. Calibration

The final scoring framework includes a Beta calibration layer.

The fitted calibration parameters are:

```text
a =  0.6845275107
b = -0.7356649810
c = -1.3849348277
```

### Raw OOT predictions

Before calibration:

| Metric      | Raw OOT |
| ----------- | ------: |
| Log loss    | 0.05441 |
| Brier score | 0.00998 |
| ECE         | 0.02529 |
| MCE         | 0.36435 |

Mean predicted probability:

```text
3.436%
```

Observed event rate:

```text
0.907%
```

### Calibrated OOT predictions

After Beta calibration:

| Metric      | Calibrated OOT |
| ----------- | -------------: |
| Log loss    |        0.04079 |
| Brier score |        0.00697 |
| ECE         |        0.01281 |
| MCE         |        0.06596 |

Mean calibrated predicted probability:

```text
2.188%
```

The calibration layer materially improves the probability-quality metrics.

However, calibration remains imperfect, particularly at the upper end of the risk distribution.

The calibration layer should therefore be treated as a component requiring ongoing monitoring rather than as a permanent correction.

---

## 22. Decision Threshold

The final evaluation applies:

```text
threshold = 0.10
```

The threshold is selected through the configured validation threshold-search framework.

On validation, the 0.10 threshold produces approximately:

| Metric             | Validation |
| ------------------ | ---------: |
| Precision          |     43.74% |
| Recall             |     32.11% |
| F1                 |     0.3704 |
| Specificity        |     99.79% |
| Flagged population |     0.379% |

The threshold is a downstream decision layer and should be distinguished from the underlying probability model.

Different operational objectives may justify different thresholds.

---

## 23. OOT Threshold Performance

Applying the same 0.10 threshold to calibrated OOT predictions gives:

```text
True positives:    9,004
False positives:  25,484
False negatives:   5,192
True negatives: 1,525,663
```

with:

```text
Precision = 26.11%
Recall    = 63.43%
F1        = 36.99%
```

The difference between validation and OOT threshold performance reflects the materially different event environment in 2022.

This demonstrates why threshold performance should be assessed on temporally separated populations rather than inferred solely from validation performance.

---

## 24. Model Interpretation

The GAM architecture provides nonlinear modelling while retaining an additive and relatively interpretable structure.

The final model combines:

```text
Linear effects
+
Nonlinear spline effects
+
Categorical effects
+
Selected interactions
```

The model can therefore represent nonlinear relationships in variables such as:

* credit score;
* DTI;
* LTV / CLTV;
* estimated LTV;
* interest rates;
* delinquency trajectory;
* macroeconomic conditions.

The configured interactions additionally allow risk relationships to vary according to other structural or behavioural characteristics.

---

## 25. Model Strengths

The final model has several important strengths.

### Point-in-time modelling

The modelling framework explicitly separates information available at the observation point from future target information.

### Short-horizon prediction

The final model is designed specifically for **3-month serious-delinquency prediction**, making recent behavioural deterioration particularly relevant.

### Behavioural depth

The feature framework incorporates lifetime, recent and trajectory-based performance information.

### Nonlinear effects

GAM splines allow important nonlinear risk relationships to be represented without requiring a fully unconstrained tree ensemble.

### Controlled interactions

The interaction layer captures selected economically meaningful relationships rather than indiscriminately generating all pairwise interactions.

### Macroeconomic context

Macroeconomic variables provide additional information about the environment affecting mortgage risk.

### Temporal validation

A dedicated 2022 OOT population provides an assessment of performance on a later vintage.

### Strong risk concentration

The model concentrates a large proportion of observed events within the highest predicted-risk groups.

### Calibration

Beta calibration materially improves the quality of the raw predicted probabilities.

---

## 26. Model Limitations

The final model should not be interpreted as a universally calibrated or permanently stable credit-risk model.

Important limitations include:

* OOT discrimination differs from validation discrimination;
* the OOT event rate is materially different from validation;
* probability calibration remains imperfect after calibration;
* threshold performance changes between populations;
* behavioural relationships may drift over time;
* macroeconomic relationships may change across regimes;
* the Freddie Mac population represents a specific mortgage population and should not automatically be assumed representative of every lending portfolio.

The model therefore requires continued temporal, discrimination and calibration monitoring if used operationally.

---

## 27. Reproducibility

The final model artefact preserves the fitted model and preprocessing state required for reproducible scoring.

The persisted package contains artefacts corresponding to:

```text
model
preprocessor
training configuration
training metadata
validation evaluation
test evaluation
OOT evaluation
calibration outputs
```

The intended scoring architecture is:

```text
Training-time transformation
        ↓
Persisted transformation state
        ↓
Same transformation at scoring time
        ↓
Model prediction
        ↓
Calibration
        ↓
Final 3-month risk probability
```

Preprocessing should not be refitted independently on future scoring populations.

Model configuration, feature definitions, target construction and temporal population definitions should remain version-controlled.

---

## 28. Final Model Summary

The final fitted model is a **behavioural, point-in-time mortgage serious-delinquency model predicting `future_90dpd` over a 3-month horizon**.

The complete modelling flow is:

```text
Canonical mortgage data
        ↓
Loan-month master
        ↓
Behavioural risk sets
        ↓
Observation ages:
3 / 6 / 9 / 12 months
        ↓
Point-in-time feature construction
        ↓
future_90dpd target
        ↓
Chronological populations
        ↓
Training-only preprocessing
        ↓
GAM representation
        ├── linear effects
        ├── spline effects
        ├── categorical effects
        └── configured interactions
        ↓
Logistic Regression
        ↓
Raw 3-month probability
        ↓
Beta calibration
        ↓
Final risk probability
        ↓
Risk ranking / threshold analysis
```

The headline OOT results are:

```text
ROC-AUC         = 0.845
PR-AUC          = 0.401
KS              = 0.648
Top 5% capture  = 69.2%
Top 5% lift     = 13.83×
Top 10% capture = 72.4%
Top 10% lift    = 7.24×
```

The most important model characteristic is its ability to **identify near-term serious-delinquency risk and strongly concentrate future events within the highest-risk population**.

The final model should therefore be viewed as a **3-month behavioural risk-ranking and probability-estimation framework with explicit point-in-time controls, nonlinear effects, controlled interactions, temporal OOT validation and probability calibration**.

---

## 29. Relationship to the Broader Modelling Framework

The final GAM is one configuration of the broader mortgage credit-risk framework.

The framework supports four interchangeable model architectures:

```text
                    Point-in-time framework
                            │
                            ▼
                     Model population
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
            LR             GAM            XGB
                                           │
                                           ▼
                                         LGBM
```

The model architecture can therefore be changed without redesigning the underlying:

* observation framework;
* target construction;
* point-in-time information boundary;
* behavioural feature framework;
* temporal population design;
* evaluation framework.

The **GAM described in this document is the final fitted model**, while Logistic Regression, XGBoost and LightGBM remain configurable alternative architectures for comparative modelling and experimentation.
