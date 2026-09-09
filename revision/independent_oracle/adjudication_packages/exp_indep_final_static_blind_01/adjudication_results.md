# Final-v14 Independent Adjudication Results

## Validated result

- Model: `gpt-5.6-luna`
- Overall cases: 63
- Human-adjudication candidates: 60
- Automatic generation failures: 3
- Overall strict correctness: 60/63 (95.2%)
- Raw inter-rater agreement: 60/60 (100.0%)
- Cohen's kappa: not estimable (zero marginal variation)
- Third-author adjudications required: 0

Only `Correct` counts as correct. `Suspicious`, `Incorrect`, and the three
automatic `GenerationFailure` outcomes count as not correct. The automatic
failures are excluded from agreement and kappa because they were not shown to
the authors.

| Frozen category | Correct | Total | Strict correctness |
|---|---:|---:|---:|
| performance | 40 | 42 | 95.2% |
| arkts_eslint | 1 | 1 | 100.0% |
| security | 19 | 20 | 95.0% |

## Claim boundary

This result evaluates the controlled final-v14 Skill-based repair component
with `gpt-5.6-luna`. It does not establish whole-repository correctness,
cross-model robustness, or a causal benefit from consulting the bundled static
references.
