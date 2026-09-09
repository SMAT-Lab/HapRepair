# Independent Repair Adjudication Guide

Evaluate each candidate against the defective input, target rule, finding location,
and human reference. Do not consult the other annotator or any generation trace.

Allowed labels:

- `Correct`: The candidate resolves the target defect, preserves intended behavior
  and interfaces, is valid ArkTS/configuration for the shown case, and is semantically
  equivalent to the reference even when the implementation differs.
- `Suspicious`: The candidate is plausible, but the shown evidence is insufficient
  to establish semantic equivalence or contains a material unresolved ambiguity.
- `Incorrect`: The candidate leaves the target unresolved, is invalid, removes
  required behavior, changes an interface without justification, introduces an
  evident regression, or repairs a different problem.

Use `Suspicious` only for genuine evidentiary uncertainty. Only `Correct` counts as
strictly correct. `Suspicious` and `Incorrect` both count as not correct.

Each author completes only their own CSV. A third author adjudicates every label
disagreement after both author files are frozen.
