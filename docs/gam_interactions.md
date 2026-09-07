# GAM V2 — Controlled Interaction Architecture

## 1. Purpose

This document describes the controlled interaction framework used by the mortgage credit-risk GAM.

A Generalized Additive Model represents the log-odds approximately as:

```text
logit(PD) = β0 + Σ f_j(X_j)
```

The project extends the additive structure with a small, explicitly configured set of interaction effects:

```text
logit(PD)
  = β0
  + Σ f_j(X_j)
  + Σ h_m(X_a, X_b)
```

The goal is not unrestricted nonlinear modelling. The goal is to capture defensible dependencies while retaining transparency, reproducibility and governance.

---

## 2. GAM V1 → V2

### Additive benchmark

```text
Features
   ↓
linear numerical effects
spline numerical effects
categorical effects
   ↓
VectorAssembler
   ↓
Logistic Regression
```

### Interaction-enabled GAM

```text
Features
   ↓
GAM preparation
   ↓
Main effects
   ├── linear numerical
   ├── spline numerical
   └── categorical
            +
   Configured interactions
   ├── numeric × numeric
   └── numeric × categorical
            ↓
     VectorAssembler
            ↓
    Logistic Regression
```

With interactions disabled, the architecture reduces to the additive benchmark.

---

## 3. Interaction philosophy

### Controlled rather than exhaustive

Interactions are supplied explicitly through configuration.

The implementation does not automatically generate every pairwise interaction.

### Supported interaction types

| Interaction | Supported | Representation |
|---|---:|---|
| Numeric × Numeric | Yes | Tensor-product spline |
| Numeric × Categorical | Yes | Varying-effect representation |
| Categorical × Numeric | Yes | Same representation |
| Categorical × Categorical | No | Rejected |

Categorical variables, including binary indicators, use the project's existing categorical treatment.

---

## 4. Spline representation

The active GAM configuration uses:

```yaml
degree: 3
num_knots: 6
```

The intended design is:

```text
numerical feature
      ↓
spline eligible?
   ┌──┴──┐
  yes    no
   ↓      ↓
quantile  linear
knots
```

A spline-eligible feature does not necessarily become a spline. If there are insufficient valid/unique quantiles, the feature falls back to its linear representation.

The fitted representation is the source of truth during scoring.

---

## 5. Numeric × Numeric interactions

For numeric features `x` and `z`, the interaction is conceptually:

```text
h(x,z) = Σ_i Σ_j β_ij B_i(x) C_j(z)
```

where `B` and `C` are the fitted basis functions for the two variables.

This permits the effect of one variable to depend on the level of the other.

For example:

```text
credit_score × estimated_ltv
```

can represent different LTV-risk relationships across credit-score levels.

With six knots and degree three, the implementation's intended spline basis dimension is:

```text
6 + 3 - 1 = 8
```

before forming the tensor product.

An interaction between two 8-dimensional bases therefore has up to:

```text
8 × 8 = 64
```

basis terms before any implementation-specific reduction or fallback.

---

## 6. Interaction centering

Tensor-product interactions can overlap with the space represented by the main effects.

The design therefore uses training-derived centering for the spline basis before constructing the interaction.

Conceptually:

```text
B_i^centered(x)
    = B_i(x) - E_train[B_i(x)]
```

and similarly for the second feature.

The tensor product is then constructed from the centered representations.

The centering state is learned during fitting and persisted with the model.

---

## 7. Numeric × Categorical interactions

A numeric × categorical interaction allows the numeric relationship to vary by category.

For:

```text
estimated_ltv × occupancy_status
```

the conceptual form is:

```text
η = β0
  + f(x)
  + Σ_k I(C=k) g_k(x)
```

where:

- `f(x)` is the main numeric effect;
- `g_k(x)` is the category-specific deviation;
- the reference category provides the baseline relationship.

This is preferable to independently fitting an unrelated curve for every category because the interaction represents deviations from the shared main relationship.

---

## 8. Reuse of fitted main-effect representation

A key implementation principle is that an interaction should use the **actual fitted representation** of its numeric feature.

Therefore:

```text
main effect fitted as spline
        ↓
interaction uses spline basis

main effect falls back to linear
        ↓
interaction uses linear representation
```

This avoids fitting a different transformation for the same feature inside an interaction.

It also ensures that training-time decisions are reproduced at scoring time.

---

## 9. Training-only learned state

The following are training-derived state:

- spline knots
- spline support/bounds
- spline-basis centering quantities
- categorical encoding state
- imputation values

They must be learned on the training population only.

Validation, test and OOT populations are transformed using the fitted training state.

---

## 10. Configuration

Interactions are specified as feature pairs.

Conceptually:

```yaml
interactions:
  enabled: true
  pairs:
    - [credit_score, estimated_ltv]
    - [credit_score, modification_flag]
    - [estimated_ltv, occupancy_status]
```

The actual active pair list is maintained in `config/parameters/base.yml`.

There is no separate configuration mechanism for binary categorical variables.

The resolver determines the interaction type from the configured feature universe:

```text
numeric + numeric
    → numeric_numeric

numeric + categorical
    → numeric_categorical

categorical + categorical
    → reject
```

---

## 11. Validation and error handling

The interaction configuration should reject:

### Unknown features

Every pair member must exist in the configured numerical or categorical feature universe.

### Duplicate pairs

Duplicate specifications should not create duplicate model terms.

### Categorical × categorical

This is intentionally unsupported to avoid uncontrolled dummy interaction expansion.

### Invalid spline state

Spline construction must respect the existing validation requirements for degree, knot count, finite support and valid ordered knots.

---

## 12. Spark pipeline persistence

The interaction stage belongs inside the persisted Spark modelling pipeline.

Conceptually:

```text
GAMPreparationTransformer
        ↓
GAMSplineEstimator
        ↓
GAMSplineModel
        ↓
Interaction construction
        ↓
VectorAssembler
        ↓
Logistic Regression
```

The fitted artefact should therefore preserve the transformation state required to reproduce the same feature representation during scoring.

This is particularly important for a distributed PySpark workflow: interactions should not depend on an external Pandas-only transformation that is unavailable when the saved Spark model is loaded.

---

## 13. Complexity control

Interactions increase model dimensionality quickly.

For that reason:

- only economically/model-risk motivated pairs should be configured;
- categorical × categorical interactions remain excluded;
- spline fallback prevents fragile features from breaking fitting;
- interaction count should remain small enough for stable fitting and review;
- OOT performance must be assessed rather than selecting interactions solely on in-sample improvement.

The objective is a **controlled GAM**, not an unrestricted interaction model.

---

## 14. Evaluation principle

An interaction is valuable only if it adds stable information beyond the corresponding additive main effects.

Evaluation should therefore compare:

```text
GAM additive benchmark
        vs
GAM + controlled interactions
```

using:

- ROC-AUC
- PR-AUC
- KS
- Brier score
- log loss
- calibration
- risk-decile capture/lift
- temporal/OOT stability

A small improvement in discrimination that materially damages calibration or temporal stability should not automatically be treated as a successful model enhancement.
