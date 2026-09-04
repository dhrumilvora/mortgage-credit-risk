# GAM V2 — Controlled Interaction Architecture

## 1. Purpose

This document describes the **GAM V2 interaction framework** for the mortgage credit-risk modelling pipeline.

GAM V1 is a strictly additive Generalized Additive Model:

```text
logit(PD) = β₀ + Σ fⱼ(Xⱼ)
```

V2 extends that architecture by allowing a **small, explicitly configured set of interactions** while retaining the core advantages of a GAM:

- controlled model complexity
- transparent feature treatment
- explicit interaction specification
- train-only learned transformation state
- reproducible scoring
- persistence inside a single Spark `PipelineModel`
- straightforward governance and model review

The objective is **not** to turn the GAM into an unrestricted nonlinear model. Interactions are deliberately constrained and must be justified as model components.

---

## 2. V1 → V2 Evolution

### GAM V1

```text
Raw features
     │
     ▼
GAM preparation
     │
     ├── Linear numerical effects
     ├── Spline numerical effects
     └── Categorical effects
             │
             ▼
      VectorAssembler
             │
             ▼
   Logistic Regression
```

Mathematically:

```text
η = β₀ + Σ fⱼ(Xⱼ) + Σ βₖZₖ
```

where `fⱼ(.)` represents a spline-based main effect and `Zₖ` represents a linear/categorical model component.

### GAM V2

```text
Raw features
     │
     ▼
GAMPreparationTransformer
     │
     ▼
GAMSplineEstimator
     │
     ▼
GAMSplineModel
     │
     ├─────────────── Main effects
     │
     └─────────────── Interaction effects
                         │
              ┌──────────┴──────────┐
              │                     │
        Numeric × Numeric    Numeric × Categorical
              │                     │
       Tensor-product        Varying-effect
           spline                spline
              │                     │
              └──────────┬──────────┘
                         ▼
                  VectorAssembler
                         │
                         ▼
                Logistic Regression
```

The resulting linear predictor is:

```text
η = β₀
  + Σ fⱼ(Xⱼ)
  + Σ hₘ(Xₐ, X_b)
```

where `hₘ(.)` is a deliberately configured interaction effect.

---

## 3. Interaction Philosophy

The interaction framework follows five principles.

### 3.1 Controlled rather than exhaustive

Interactions are supplied explicitly in configuration.

The model does **not** automatically generate all pairwise interactions.

This keeps the model reviewable and limits unnecessary complexity.

### 3.2 At least one side must be numeric

Supported interaction types are:

| Interaction | Supported | Representation |
|---|---:|---|
| Numeric × Numeric | Yes | Tensor-product spline |
| Numeric × Categorical | Yes | Varying-effect spline |
| Categorical × Numeric | Yes | Same as Numeric × Categorical |
| Categorical × Categorical | No | Explicitly rejected |

Binary variables are **not a separate model type**. They are already represented in the categorical feature set and therefore use the same Numeric × Categorical machinery.

### 3.3 Reuse the fitted main-effect spline basis

Interactions do not independently refit spline knots.

For a numeric feature participating in an interaction, the interaction reuses the spline basis learned for that feature's main effect.

This provides:

- consistent treatment of the same variable
- no duplicate knot-learning logic
- no train/score mismatch
- simpler persistence
- easier interpretation

### 3.4 All learned state is train-only

Anything learned from the data must be fitted using the training partition only.

This includes:

- spline knots
- spline support bounds
- spline-basis centering quantities
- categorical encoding state

Validation and OOT data only receive transformations learned from training data.

### 3.5 V1 must remain reproducible

With:

```yaml
interactions:
  enabled: false
```

the interaction stage is absent and the pipeline follows the V1 architecture.

This allows V1 to remain the frozen additive benchmark/control while V2 is evaluated as a separate candidate.

---

# 4. Configuration

Interactions are configured under the GAM configuration.

Example:

```yaml
interactions:
  enabled: true
  pairs:
    - [credit_score, estimated_ltv]
    - [credit_score, modification_flag]
    - [estimated_ltv, occupancy_status]
```

The configuration intentionally contains only **feature pairs**.

There is no separate configuration for binary variables.

The implementation determines the interaction type from the existing feature groups:

```text
if feature ∈ numerical_features:
    type = numeric
else:
    type = categorical
```

The resolver then applies:

```text
numeric × numeric
        → numeric_numeric

numeric × categorical
        → numeric_categorical

categorical × categorical
        → reject
```

---

# 5. Interaction Type 1 — Numeric × Numeric

## 5.1 Motivation

A standard additive GAM assumes:

```text
effect = f(x) + g(z)
```

This means the effect of `x` is the same regardless of the value of `z`.

A numeric × numeric interaction allows:

```text
effect = f(x) + g(z) + h(x,z)
```

The additional term captures situations where the effect of one risk variable changes depending on the level of another.

For example:

```text
credit_score × estimated_ltv
```

can capture the possibility that the effect of a given LTV is different for lower-credit-score borrowers than for higher-credit-score borrowers.

---

## 5.2 Tensor-product spline

Suppose:

```text
B₁(x), B₂(x), ..., Bₚ(x)
```

are the spline basis functions for `x`, and:

```text
C₁(z), C₂(z), ..., C_q(z)
```

are the spline basis functions for `z`.

The interaction basis consists of:

```text
Bᵢ(x) × Cⱼ(z)
```

for every pair `(i,j)`.

Illustration:

```text
                 z spline basis
              C1   C2   C3 ... Cq
             ┌────────────────────
B1(x)        │ ×    ×    ×  ... ×
B2(x)        │ ×    ×    ×  ... ×
B3(x)        │ ×    ×    ×  ... ×
...          │
Bp(x)        │ ×    ×    ×  ... ×
```

The resulting terms represent a flexible two-dimensional response surface.

---

## 5.3 Current spline configuration

With:

```yaml
degree: 3
num_knots: 6
```

the implementation produces:

```text
number of spline basis functions
= num_knots + degree - 1
= 6 + 3 - 1
= 8
```

Therefore one Numeric × Numeric interaction produces:

```text
8 × 8 = 64
```

interaction basis terms.

For:

```text
credit_score × estimated_ltv
```

the interaction therefore has 64 tensor-product basis terms.

---

# 6. Interaction Centering

A full tensor-product basis can overlap conceptually with the main-effect space.

Without constraints, it becomes harder to say:

> "This part of the fitted effect belongs to the interaction"

versus:

> "This part belongs to the main effects."

That is undesirable for an interpretable risk model.

The V2 design therefore centers the spline basis before constructing the Numeric × Numeric tensor product.

For a basis function `Bᵢ(x)`:

```text
Bᵢᶜ(x) = Bᵢ(x) - E_train[Bᵢ(x)]
```

The interaction then uses:

```text
Bᵢᶜ(x) × Cⱼᶜ(z)
```

where the centering quantities are learned from training data and persisted with the fitted pipeline.

Conceptually:

```text
Raw spline basis
       │
       ▼
Training-data centering
       │
       ▼
Centered spline basis
       │
       ├──────────────┐
       │              │
       ▼              ▼
     x basis        z basis
       │              │
       └──────┬───────┘
              ▼
       Tensor product
              │
              ▼
       Interaction surface
```

This keeps the interaction representation more cleanly separated from the main-effect representation.

---

# 7. Interaction Type 2 — Numeric × Categorical

## 7.1 Motivation

A Numeric × Categorical interaction allows the effect of a numeric variable to vary by category.

For example:

```text
estimated_ltv × occupancy_status
```

asks whether the relationship between LTV and risk differs across occupancy categories.

Instead of fitting one universal function:

```text
f(estimated_ltv)
```

the model can represent:

```text
f(estimated_ltv)
+
category-specific deviation
```

---

## 7.2 Varying-effect formulation

For a categorical variable `C` with a reference category:

```text
η = β₀
  + f(x)
  + Σₖ I(C = k) gₖ(x)
```

where:

- `f(x)` is the main effect
- `gₖ(x)` is the deviation curve for category `k`
- the omitted/reference category has no additional deviation term

This is preferable to creating a completely independent spline curve for every category because the main effect remains explicit and the interaction represents the **difference from the reference relationship**.

---

## 7.3 Illustration

```text
                 Numeric variable x
                        │
                 ┌──────┴──────┐
                 │ spline basis │
                 └──────┬──────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Category A    Category B    Category C
       reference       deviation      deviation
          │             │             │
          │          × spline       × spline
          │             │             │
          └─────────────┴─────────────┘
                        │
                        ▼
                Interaction vector
```

The categorical variable uses the existing Spark `StringIndexer` + `OneHotEncoder` treatment.

There is therefore no separate binary implementation.

A binary categorical variable is simply the special case where the encoded categorical dimension corresponds to two levels.

---

# 8. Interaction Pipeline Architecture

The interaction implementation is deliberately placed **inside** the persisted Spark pipeline.

```text
                 TRAINING DATA
                      │
                      ▼
          GAMPreparationTransformer
                      │
          ┌───────────┴───────────┐
          │                       │
       Imputation          Categorical encoding
          │                       │
          └───────────┬───────────┘
                      ▼
             GAMSplineEstimator
                      │
                      ▼
                GAMSplineModel
                      │
          ┌───────────┴───────────┐
          │                       │
     Main effects           Interaction stage
          │                       │
          │              ┌────────┴─────────┐
          │              │                  │
          │           Num × Num          Num × Cat
          │              │                  │
          │         tensor surface     varying effect
          │              │                  │
          └──────────────┴──────────────────┘
                         │
                         ▼
                  VectorAssembler
                         │
                         ▼
                Logistic Regression
```

The fitted artifact therefore contains:

```text
preprocessing
+ categorical encoding
+ spline knots
+ spline support
+ interaction state
+ feature assembly
+ logistic coefficients
```

inside the Spark `PipelineModel`.

---

# 9. Feature Representation

The model deliberately keeps interaction outputs as **vector features** rather than creating an uncontrolled number of permanent top-level DataFrame columns.

Conceptually:

```text
Main effects
    │
    ├── linear numerical features
    ├── spline basis vectors
    └── categorical vectors

Interactions
    │
    ├── credit_score × estimated_ltv
    │       └── 64-dimensional vector
    │
    ├── credit_score × modification_flag
    │       └── varying-effect vector
    │
    └── estimated_ltv × occupancy_status
            └── varying-effect vector

                    ↓
              VectorAssembler
                    ↓
               LR features
```

This keeps the feature assembly manageable as the interaction configuration grows.

---

# 10. Validation and Error Handling

The interaction configuration is validated before model fitting.

### Duplicate pairs

Duplicate interaction specifications should be rejected rather than silently creating duplicate model terms.

### Unknown features

Every interaction feature must belong to the configured numerical or categorical feature universe.

### Categorical × categorical

This is explicitly unsupported:

```text
occupancy_status × property_type
```

results in an error rather than silently generating a high-cardinality dummy interaction.

### Numeric features

Numeric features used in interactions must have an appropriate spline representation when the interaction engine reuses the fitted spline basis.

This ensures the interaction and its corresponding main effect use consistent transformations.

### Invalid spline configuration

Existing spline validation continues to apply, including:

- valid degree
- valid number of knots
- finite bounds
- strictly increasing internal knot values
- sufficient feature variation

---

# 11. Persistence and Scoring

The interaction model is designed to behave like the existing GAM pipeline:

```text
Training
────────
fit preprocessing
      ↓
fit spline knots
      ↓
fit interaction state
      ↓
fit logistic regression
      ↓
persist PipelineModel


Scoring
───────
load PipelineModel
      ↓
apply stored preprocessing
      ↓
apply stored spline knots
      ↓
apply stored interaction state
      ↓
apply stored coefficients
      ↓
PD
```

No scoring-time fitting occurs.

This is particularly important for the monthly mortgage risk-feed use case because the same frozen model must be applied consistently to future observation months.

---

# 12. V1 Compatibility

Interaction support is optional.

With:

```yaml
interactions:
  enabled: false
```

the pipeline remains:

```text
GAMPreparationTransformer
        ↓
GAMSplineEstimator
        ↓
GAMSplineModel
        ↓
VectorAssembler
        ↓
LogisticRegression
```

No interaction terms are introduced.

This provides a clean control:

```text
             ┌───────────────┐
             │ Same data     │
             │ Same split    │
             │ Same target   │
             │ Same features │
             └───────┬───────┘
                     │
             ┌───────┴────────┐
             ▼                ▼
          GAM V1            GAM V2
        additive        controlled interactions
             │                │
             └───────┬────────┘
                     ▼
              Compare OOT
```

---

# 13. Current Candidate Interactions

The initial V2 configuration contains:

```yaml
interactions:
  enabled: true
  pairs:
    - [credit_score, estimated_ltv]
    - [credit_score, modification_flag]
    - [estimated_ltv, occupancy_status]
```

These represent three distinct modelling hypotheses:

### Credit score × estimated LTV

Whether the risk relationship associated with leverage changes materially across borrower credit quality.

```text
credit quality
      ×
leverage
```

### Credit score × modification flag

Whether the incremental risk signal associated with credit quality differs for modified versus non-modified loans.

```text
credit quality
      ×
loan modification state
```

### Estimated LTV × occupancy status

Whether leverage has different risk implications depending on occupancy.

```text
leverage
      ×
occupancy
```

These are hypotheses to be **tested**, not assumed to be beneficial merely because they are included.

---

# 14. Model Evaluation Framework

V2 should not be accepted solely because its training or validation AUC improves.

The comparison against the frozen GAM V1 benchmark should consider:

### Discrimination

- ROC-AUC
- PR-AUC
- KS
- top-5% capture
- top-10% capture
- top-20% capture

### Probability quality

- log loss
- Brier score
- calibration curve
- ECE
- MCE
- O/E

### Generalization

```text
Train → Validation → OOT
```

The OOT result receives particular weight because the objective is future monthly risk scoring.

### Stability

Assess whether the interaction model produces:

- unstable coefficients
- extreme predictions
- excessive sensitivity
- degraded performance in OOT periods
- materially worse calibration

### Interpretability

Each retained interaction should have a defensible business/risk interpretation.

---

# 15. Interaction Acceptance Principle

An interaction should earn its place in the model.

A useful decision framework is:

```text
Does the interaction improve
OOT risk discrimination?
          │
          ├── No ──► Reject
          │
          ▼
Does it improve or preserve
probability quality?
          │
          ├── No ──► Strong evidence required
          │
          ▼
Is the effect stable OOT?
          │
          ├── No ──► Reject
          │
          ▼
Can the interaction be
explained and governed?
          │
          ├── No ──► Reject
          │
          ▼
       RETAIN
```

The goal is therefore **not maximum flexibility**.

The goal is:

> **the smallest interpretable interaction set that materially improves future risk prediction without sacrificing probability quality or stability.**

---

# 16. Testing Requirements

Before running the full mortgage dataset, the interaction implementation should be tested on a small deterministic fixture.

Minimum tests:

### Configuration

- interactions disabled
- valid Numeric × Numeric
- valid Numeric × Categorical
- categorical × categorical rejected
- unknown feature rejected
- duplicate pair rejected

### Numeric × Numeric

- correct interaction dimension
- correct basis-product construction
- centering is based on training data
- scoring reuses stored centering values
- no new knots are fitted

### Numeric × Categorical

- correct encoded dimension
- reference category treatment
- varying-effect terms generated correctly
- unseen categories follow existing categorical handling

### Pipeline

- interactions appear in the assembled feature vector
- disabled interactions reproduce V1 feature construction
- model saves successfully
- model loads successfully
- predictions before/after save-load are identical within numerical tolerance

### Regression protection

```text
V1 disabled
     ↓
same transformation
     ↓
same feature vector
     ↓
same coefficients
     ↓
same predictions
```

This test is particularly important because GAM V1 is the frozen benchmark.

---

# 17. Governance Considerations

The interaction configuration should be treated as part of the model specification.

For every production candidate, record:

```text
Model version
    │
    ├── feature configuration
    ├── spline configuration
    ├── interaction configuration
    ├── training data period
    ├── validation period
    ├── OOT period
    ├── fitted transformation state
    └── fitted coefficients
```

An interaction should not be added merely through an ad-hoc notebook modification.

The configuration, code, tests, model artifact, and evaluation results should move together through version control.

---

# 18. Summary

GAM V2 extends the frozen additive GAM V1 benchmark with a deliberately constrained interaction framework.

```text
                 GAM V2
                    │
        ┌───────────┴───────────┐
        │                       │
    Main effects            Interactions
        │                       │
   ┌────┴────┐            ┌─────┴─────┐
   │         │            │           │
 linear   spline       Num × Num   Num × Cat
                          │           │
                       tensor      varying
                       product      effect
```

The key architectural decision is that interactions remain **first-class, persisted, configurable model components**, rather than being an uncontrolled feature-engineering layer.

The framework intentionally supports:

```text
Numeric × Numeric       ✓
Numeric × Categorical   ✓
Categorical × Numeric   ✓
Categorical × Categorical ✗
Binary as separate type ✗
```

The V2 model should ultimately be judged against the frozen V1 GAM using **OOT discrimination, calibration, stability, and interpretability**, rather than in-sample fit alone.
