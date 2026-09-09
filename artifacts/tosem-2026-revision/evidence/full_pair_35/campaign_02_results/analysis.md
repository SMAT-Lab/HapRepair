# Clean 35-Project Paired Comparison

Package: `AN-FULL-PAIR-35-CLEAN-R2-20260811`. 70 project-repetition pairs.

All 35 projects and both repetitions are retained. A failed condition is counted as no deployable repair and receives unchanged-source ITT metrics. Positive paired-benefit values favor HapRepair.

| Outcome | HapRepair | Baseline | Paired benefit | Project-clustered 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mean final alerts | 1.657 | 21.043 | 19.386 fewer | [4.828, 37.329] |
| Acceptable candidate rate | 100.000% | 82.857% | 17.143% | [5.714%, 30.000%] |
| Mean introduced-alert benefit | 0.143 introduced | 0.171 introduced | 0.029 | [-0.171, 0.286] |

## Repetitions

| Repetition | Skill acceptable | Baseline acceptable | Mean final-alert benefit |
| --- | ---: | ---: | ---: |
| 1 | 100.000% | 80.000% | 20.800 |
| 2 | 100.000% | 85.714% | 17.971 |

## Burden

| Frozen burden stratum | Pairs | Mean final-alert benefit | Mean acceptable-candidate benefit |
| --- | ---: | ---: | ---: |
| low | 24 | 13.583 | 20.833% |
| middle | 22 | 48.773 | 31.818% |
| high | 24 | -1.750 | 0.000% |

The clustered linear model uses `log1p(initial alerts)` and all 70 project-repetition observations: slope=-0.672, cluster-robust 95% CI [-6.673, 5.329], p=0.8262.

## Robustness And Failures

Leave-one-project-out mean final-alert benefits range from 13.676 to 20.544; direction reversals: 0. The completed-pair-only descriptive sensitivity retains 58 pairs.

There are 12 failed conditions: structural_or_api_guard_exhaustion=12. No failure is classified as a provider API or infrastructure failure.

Raw per-project rows, failure details, bootstrap settings, regression outputs, and source hashes are stored beside this report. Token, invocation, and wall-time data are intentionally not reported as effectiveness evidence.
