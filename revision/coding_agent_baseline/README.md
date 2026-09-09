# EXP-AGENT-10

This package prepares the reviewer-requested comparison between HapRepair and a
strong contemporary coding agent on ten OpenHarmony projects. The full frozen
contract is in `protocol.json`; canonical paper-facing metrics are defined in
`json/metric_contract.json`.

The formal ten-project comparison uses `gpt-5.6-luna` for both conditions. The
validated runner pilot below used `gpt-5.6-sol`; it is retained only as
auxiliary evidence and is not pooled with the formal results.

The old `revision/code/run_haprepair_codex_agent.py` is not used. That script
gives Codex access to HapRepair's MCP server and retrieval examples, which would
violate the no-knowledge-base comparator condition.

Preparation order:

```bash
python3 prepare_sources.py
python3 scan_projects.py --run-id candidate_scan_01
python3 select_projects.py --scan-run scan_runs/candidate_scan_01/scan_manifest.json
```

Run a bounded dry-run before spending model tokens:

```bash
python3 run_agent_baseline.py \
  --run-id pilot_ohos_cordova_dry_01 \
  --project ohos_cordova \
  --dry-run
```

Run one coding-agent project condition:

```bash
python3 run_agent_baseline.py \
  --run-id exp_agent_10_coding_agent_01 \
  --project ohos_cordova
```

`validation_gate.py` owns the scan budget. It reserves an attempt before each
post-edit scan, so a failed scan still consumes one of the five allowed
attempts. Initial and evaluator-only final scans do not consume that budget.

`preflight_builds.py` froze initial-version build availability using the exact
project-declared SDKs. API 22 was insufficient; APIs 8, 12, 18, 20, 23, and
26.0.0 were installed without modifying project build profiles. The common
`build_gate.py` is enabled only for `TextComponentTest`, `asn1_ber`, and
`ohos_cordova`. It runs after every repair round in both conditions and rolls a
round back before HomeCheck validation if the build fails. Evaluator builds use
Hvigor's compiler-supported `ignoreWarning` mode, so ArkTS checker diagnostics
such as `arkts-no-any-unknown` do not stop code generation. Module resolution,
packaging, timeout, and other process failures remain fatal. No selected
project has a passing host-side test task, so the semantic claim remains
statically validated only.

The active formal run ID is `exp_agent_10_luna_formal_06`; its complete task
order is recorded in `protocol.json` before execution. Runs `formal_01` through
`formal_05` are retained as harness-diagnostic records and are not pooled with
the formal results. In particular, `formal_05` incorrectly treated ArkTS
checker diagnostics as fatal build failures.

The agent runs in the pinned `hybrid-gym-codex:0.146.0` container. Only the
isolated project copy and a run-local Codex home are writable; HomeCheck rule
documentation and implementations are mounted read-only. The HapRepair
repository, retrieval pairs, held-out references, and prior repair outputs are
not mounted into the container. The authentication file is mounted read-only
and is not copied into run artifacts.

Validated pilot:

- Run: `pilot_ohos_cordova_agent_03`
- Pinned input: `ohos_cordova` at commit
  `339433ac3c459cd4c3ca532ff8dc38e294e14249`
- Initial/final alerts: 26/0
- Eliminated/introduced/net reduction: 26/0/26
- Agent-visible validation scans: 2
- Final evaluator scan: 0 alerts
- Build/tests: not run; the isolated project had no usable `hvigorw`, `hvigor`,
  or `ohpm`, so this pilot is statically validated only
- Restricted-data access: none; the prohibited data was not mounted

The pilot validates the runner and metric path only. It is not a paper-facing
ten-project result, and HomeCheck alert elimination is not semantic correctness.

Matching HapRepair pilot:

- Run: `pilot_ohos_cordova_haprepair_02`
- Frozen local RAG: 383 pairs, 63 rules, same-rule Top-1, Stella encoder
  revision `7817065102fd9e1b031fe874e910c01f40b2f001`
- Initial/final alerts: 26/12
- Eliminated/remaining/introduced/net reduction: 15/11/1/14
- Agent-visible validation scans: 5
- Tokens/wall time: 914,169 / 1,909.5 seconds
- Build/tests: not run; statically validated only

The validated paired pilot summary is outside the Git worktree at:

```text
/home/zhihao/hdd/haprepair/baseline_data/exp_agent_10/paired_pilots/
  pilot_ohos_cordova_pair_01/summary.json
```

All paired comparability checks pass: project commit/tree, initial findings,
model ID, provider endpoint, localization, and scan budget. In this one pilot,
the coding agent eliminated 26/26 alerts using 3,809,014 tokens and 822.8
seconds; HapRepair eliminated 15/26 using 914,169 tokens and 1,909.5 seconds.
This is auxiliary runner evidence, not a general method comparison.

`prepare_sources.py` shallow-clones each distinct upstream repository with a
sparse checkout and writes the exact source commits to `source_manifest.json`.
Large recovered source trees live outside the Git worktree under
`../baseline_data/exp_agent_10/` by default.

No repair result is valid until source recovery, the initial HomeCheck scan,
seeded project selection, the runner pilot, and metric-contract verification
all pass.
