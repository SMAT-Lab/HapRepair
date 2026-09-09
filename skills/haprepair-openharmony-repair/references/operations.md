# HapRepair v8 Session Operations

## Boundary

Use `scripts/repair_session.py` for project repair. It composes workspace snapshots,
complete per-location evidence, structural/public-API/semantic guards, configured
build/tests, HomeCheck scan accounting, candidate retention, and best-valid
selection. Use `scripts/homecheck.py` directly only for `verify`, `guide`, `spec`, or
isolated scanner diagnostics.

The evaluator owns the controller lifecycle in an evaluator-controlled agent turn.
Such a turn receives frozen findings, the round plan, and feedback from the
evaluator. The agent may edit project source/configuration files and write the
completion report, but must not invoke `init-session`, `scan-initial`, `begin-round`,
`record-completion`, `preflight-round`, `validate-round`, or `finalize`; it must not
run HomeCheck scans or create a second plan. The runner sets `HAPREPAIR_AGENT_MODE=1`
and the bundled scripts reject those operations. Standalone operators may use the
full lifecycle below.

Source preservation is a hard invariant: retain every existing struct, class,
component, function, method, `build()` method, exported symbol, and public API.
Never replace a component by deleting its original declaration, and never remove
code merely to silence a finding. Structural preflight reports the exact removed
declarations; restore them before any other repair while preserving useful edits.

The pinned paper-facing runtime is CodeLinter `6.0.240` with HomeCheck source commit
`461ad0a2f3a71a22ceeb18f7cc4a9fa67c986c59`. The default build environment injects
Hvigor `ignoreWarning`; warning-class ArkTS diagnostics such as `any` do not stop an
otherwise valid build. Module resolution, compilation, packaging, timeout, and
process failures remain fatal.

## Planning

`scan-initial` and `begin-round` produce a complete plan. Each rule record contains:

- Every affected file and frozen `(line, column)` location.
- `semantic_spec`, including the mandatory core and family specification paths.
- An optional `guide` for the frozen 383-example corpus.
- Same-file, five-line candidate `interaction_clusters` for related rule families.

The clusters are inspection hints, not proof that two diagnostics share a cause.
Resolve the actual owning entity from repository context.

## Completion report

Record every frozen location with this schema:

```json
{
  "selected_rules": ["@performance/foreach-index-check"],
  "consulted_specs": {
    "@performance/foreach-index-check": [
      "/absolute/path/to/core-invariants.md",
      "/absolute/path/to/collections-reuse.md"
    ]
  },
  "consulted_guides": {},
  "entity_repairs": [
    {
      "rule": "@performance/foreach-index-check",
      "relative_path": "entry/src/main/ets/pages/Index.ets",
      "locations": [{"line": 42, "column": 9}],
      "entities": ["ForEach over this.items in ItemList.build"],
      "invariants": ["callback parameter meaning", "key stability and uniqueness"],
      "evidence": ["item.id is assigned once and all producers enforce uniqueness"],
      "transformation": "Kept the render index and changed the key generator to item.id",
      "status": "repaired"
    }
  ],
  "unresolved_external": []
}
```

`record-completion` requires every planned rule, exact normative spec paths, every
rule/file group, and every frozen location. Entity, invariant, evidence, and
transformation fields must be non-empty. `blocked`, deferred work, partial file
coverage, and `unresolved_external` do not pass the coverage gate.

## Preflight and validation

`preflight-round` requires a source edit and runs:

1. Source deletion, large-deletion, and removed-declaration guard.
2. Package-exported public-API comparison with the initial project.
3. Semantic regression guard. It rejects newly introduced `indexOf` calls, mechanical
   `@Reusable` without lifecycle/reuse/identity evidence, and state-decorator count
   changes without ownership/mutation/UI-read evidence.
4. Namespace-export guard. It rejects removal or redirection of any baseline
   `export * as Namespace from '...'` declaration; replacing a namespace barrel with
   named re-exports or an object wrapper is not API-preserving.

`validate-round` then runs configured build and tests. Gate failures retain the active
round for in-place repair and consume no HomeCheck scan. A successful gate sequence
consumes one validation scan. Residual and introduced findings become the next round's
complete input.

A same-file eliminated-to-introduced transition between related rule families is an
unresolved sibling exchange. The candidate remains available for continued repair but
cannot become best-valid until a later round removes the exchange.

Best-valid minimizes `(final target alerts, introduced target alerts, changed source
files, round)` among gate-passing candidates without unresolved sibling exchanges.
No project-level rollback occurs during repair. `finalize` restores best-valid when
needed, then performs one evaluator-only scan.

## Metrics

Alert identity is a multiset over normalized `(relative_path, rule, message)` tuples:

- `Eliminated = A0 - Af`
- `Remaining = A0 intersect Af`
- `Introduced = Af - A0`
- `NetReduction = |A0| - |Af|`

Every scan preserves raw/normalized findings, stdout, stderr, command, elapsed time,
and hashes. Alert elimination is analyzer feedback, not a semantic oracle.
