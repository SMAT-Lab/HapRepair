# EXP-INDEP-63 Controlled Benchmark

This directory prepares the controlled independent-oracle experiment requested
in the TOSEM reviews. The retrieval source remains the audited 383-pair,
63-rule view in `revision/knowledge_base/rule_complete_383.jsonl`. Test cases
are constructed separately and are never inserted into the retrieval database.

Run the eligibility audit from the HapRepair repository root:

```bash
python revision/independent_oracle/audit_split_eligibility.py
```

## Selected protocol

The benchmark contains one new defective/reference-repair case for each of the
63 rules. Both versions are assembled as standalone CodeLinter projects:

- the defective project must report the case's target rule;
- the repaired project must not report the target rule;
- the reference repair remains hidden during GPT-5.1 patch generation;
- the 383-pair database is read-only and supplies Top-1 same-rule RAG;
- benchmark-to-database exact and near-duplicate checks are frozen before use.

The manifest keeps the knowledge-base rule ID in `rule`. When the installed
CodeLinter reports a renamed rule ID, `codelinter_rule` records that explicit
alias without changing the retrieval key. CodeLinter 6.0.240 reports
`@performance/init-list-component` for the knowledge-base rule
`@hw-ets-eslint/init-list-component`. The Linux performanceAgent does not
report its documented reuseId examples, so that rule is validated with the
equivalent HomeCheck checker
`@performance/suggest-reuseid-for-if-else-reusable-component-check`.

## CodeLinter overlay

The frozen validator requires the compatibility overlay recorded in
`codelinter_overlay.json` and verifies every integrated file by SHA-256 before
running. The overlay corresponds to the changes in the sibling HomeCheck source
tree at commit `461ad0a2f3a71a22ceeb18f7cc4a9fa67c986c59`:

- normalize the SDK's `../js/api` signatures before comparison;
- accept the current `ArkInvokeStmt` representation of WaterFlow `onAppear`;
- enable the equivalent HomeCheck reuseId checker in the integrated rule set.

Every run records the base CodeLinter version, overlay ID, manifest hash, and
the verified hashes of all integrated files. Results must therefore be named
with the full `tool_identity`, not described as stock CodeLinter 6.0.240.

`cases_manifest.json` is the 63-rule authoring queue. `benchmark/` contains the
defective and repaired projects. Run a development validation while authoring:

```bash
python revision/independent_oracle/validate_benchmark.py \
  --allow-incomplete --run-id <pilot-id>
```

Remove `--allow-incomplete` for the frozen 63-case validation run.

## Near duplicates

`split_eligibility.json` records why the earlier plan to hold out knowledge-base
rows was abandoned: 12 rules are singletons. It is retained as decision
provenance and is not the selected experimental split.

## Completed adjudication

The frozen `gpt-5.6-luna` run has completed two-author blind annotation. Validate
the raw annotations and regenerate the result artifacts with:

```bash
python revision/independent_oracle/summarize_adjudication.py
```

The command verifies annotation completeness, allowed labels, blind IDs, case
hashes, and required third-author decisions without changing the raw CSV files.
It writes machine-readable results and manuscript-ready text under
`adjudication_packages/exp_indep_63_blind_01/`. The frozen benchmark taxonomy is
42 performance rules, 20 security rules, and one ArkTS-ESLint rule.
