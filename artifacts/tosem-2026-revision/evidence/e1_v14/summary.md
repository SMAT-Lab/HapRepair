# E1 v14 Reconciled Summary

The reconciled dataset contains 34 valid manifests from the original v14 scheduler
and the independently audited `flutter_embedding` retry. The original invalid
`flutter_embedding` condition remains preserved and excluded.

## Main descriptive result

- Projects: 35
- Alerts: 9884 initial, 39 final,
  9847 eliminated, 37 remaining,
  2 introduced, and 9845 net reduced
- Gross detected-alert elimination: 99.6257%
- Net detected-alert reduction: 99.6054%
- Zero-final-alert projects: 31
- Validation scans: 83
- Total trace tokens: 946129959
- Public-rate model-cost estimate: $34.81 ($0.99/project and
  $0.0035/eliminated alert), using the standard short-context
  `gpt-5.6-luna` prices accessed on August 11, 2026
- Provider invoice: unavailable

## Sensitivity result

Excluding development-exposed `wifi_testapp`: 34 projects,
8810 initial alerts, 39 final
alerts, and 99.5573% net detected-alert reduction.

## Static-guide access boundary

Exact guide filenames appeared in completed command traces for
98 of
241 initial project-rule requirements
(40.6639%). Only
6 projects had complete
trace-supported initial-rule guide access. This result must be reported as observed;
opening semantic specifications or reading the generated plan is not counted as
opening a bundled static guide.

All alert results are HomeCheck-defined detected-alert outcomes. They do not by
themselves establish semantic correctness or confirmed vulnerability repair.
