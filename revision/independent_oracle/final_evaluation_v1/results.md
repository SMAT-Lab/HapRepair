# Final-v14 Independent Semantic Correctness

The canonical E3 result is **63/63 Correct (100%)**: 42/42 performance, 1/1 ArkTS-ESLint, and 20/20 security.

Both authors independently labeled all 63 cases `Correct`. Raw agreement is 63/63, with no disagreements and no third-author adjudications. Cohen's kappa is not estimable because both author marginals contain only one category.

The result combines the 60 candidates in `exp_indep_final_static_blind_01` with the three independently labeled replacements in `exp_indep_3_v14_retry_sensitivity_blind_01`. The latter replace harness-level lexical false rejections: the original outputs were rejected when a substring filter matched benign local search or environment-inspection commands, although no HomeCheck, CodeLinter, or `repair_session` invocation occurred. This is not an API-retry narrative.

Earlier split summaries remain unchanged in their original artifacts for auditability, but are superseded for canonical reporting by decision `E3-13`.

This result supports controlled repair-component semantic correctness only. It does not establish whole-repository correctness or a causal effect of static-reference access.
