# EXP-RQ1-PRECISION Annotation Guide

Judge whether the HomeCheck finding is a true instance of the named rule in the
frozen source snapshot. Inspect the complete source file and the bundled rule
evidence. Do not consult the other author or any repair outcome.

- `Correct`: the reported location and surrounding repository evidence satisfy
  the rule's defect condition.
- `Suspicious`: the available static evidence is insufficient to decide, or the
  result depends on unavailable generated/runtime behavior.
- `Incorrect`: the report does not satisfy the rule, targets the wrong construct,
  or is contradicted by the shown source evidence.

Only `Correct` counts as a strict true positive. `Suspicious` and `Incorrect`
both count as not correct in strict precision. A rationale is mandatory for
`Suspicious` and `Incorrect`. Each author completes only their own label file.
The third author receives only disagreements after both author files are frozen.
