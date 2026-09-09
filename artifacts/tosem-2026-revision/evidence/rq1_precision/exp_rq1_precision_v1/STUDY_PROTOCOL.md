# EXP-RQ1-PRECISION Study Protocol

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-07
- Verification Status: FROZEN
- Version Label: exp_rq1_precision_v1

## Study Overview

- Research question: What is the strict precision of HomeCheck-defined
  performance and security findings in the frozen 35-project population?
- Design: stratified independent expert annotation of static-analysis findings.
- Unit: one frozen HomeCheck finding and its complete source file.
- Labels: Correct, Suspicious, Incorrect; only Correct is a strict true positive.

## Sampling

- Performance: 170 of 9,401 findings, stratified by all 38 observed rules.
- Security: 61 of 483 findings, with 30 no-commented-code, 30 no-cycle, and the
  sole no-unsafe-hash finding; project stratification is retained within rule.
- Selection: frozen SHA-256 ordering without replacement.

## Annotation

- Two authors label every case independently.
- Neither author can access the other author's labels during annotation.
- A third author receives and resolves disagreements only after both files are
  complete and frozen.
- Suspicious and Incorrect require a written rationale.

## Analysis

- Use design-weighted stratified precision with finite-population correction.
- Report per-rule, category, overall, and category-macro estimates, 95% intervals,
  raw agreement, Cohen's kappa when estimable, and adjudication counts.
- Scope is precision in the frozen population, not natural-project recall or
  confirmed vulnerability prevalence.

## Ethics And Data Handling

- The package contains public source artifacts and author-generated labels, not
  personal participant data.
- The authors must confirm local institutional requirements before involving
  annotators outside the author team.
- Annotation files remain local and access-controlled by role.
