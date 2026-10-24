# Question N — model development document

Copy to `q<N>_methodology.md` for each problem. Owned by modeling: it states the
methods and why they were chosen. Implementation is coding's; rationality
verification is writing's.

Global assumptions (data scope, units, environment) live in `assumptions.md`.
Method-local assumptions — functional-form, estimator, or scope choices tied to
one method — are stated below at the method that uses each, worded explicitly as
assumptions so `sensitivity_analysis.md` and `strengths_and_weaknesses.md` can
cite them. Result interpretation goes in `result_interpretation.md`, not here.
Code-related requirements (output files, figures) belong in the coding work
order, not here.

## 1. Data and notation
_(inputs used, source file, units; symbols beyond the `assumptions.md` table)_

## 2. Data preprocessing
_(missing values, invalid readings, outlier policy — each with its rationale,
stated as an assumption where it is one)_

## 3. Q<N>.1 — <sub-part name>
_(the model/estimator, its equations, and why this method over alternatives)_

## 4. Q<N>.2 — <sub-part name>
_(one section per sub-part)_

## 5. Validation
_(how the method is checked: held-out error, recovery check, physical sanity
bounds. Distinct from sensitivity analysis, which perturbs fixed assumptions.)_
