# EXP-LOC-18 Three-Model Run

Status: all 54 formal LLM conditions and the frozen HomeCheck condition are
completed and audited.

The formal conditions are `gpt-5.6-luna`, `gpt-5.6-sol`, and
`deepseek-v4-flash`. Each condition receives the same full source file and the
same eight rule IDs, without repository or tool access. The paper-facing endpoint
is line-free detection over 30 unique `(file, rule)` positives. HomeCheck also
detects all 38 manually constructed instances. Precision and F1 are conditional
on the positive-enriched 18-file benchmark.

Thinking is disabled for all three conditions with `reasoning_effort=none`.
The interrupted `exp_loc_18_three_model_smoke_02` directory is a non-formal
interface diagnostic and is excluded from every metric.

The formal run contains 54 distinct, parseable responses with exact
requested/reported model identity, no API failures, and no reasoning content or
reasoning tokens. The reproducible audit is in `formal_run_audit.json`.

The joint result is HomeCheck `30/30`, DeepSeek `10/30`, Luna `15/30`, and
Sol `15/30` by file-rule recall. On the six single-defect files, the corresponding
counts are `6/6`, `1/6`, `3/6`, and `5/6`. See `combined_summary.json` and
`homecheck_run_audit.json`.

The v1 non-use decision is preserved as superseded audit history. The v2 audit
records the recovered exact upstream tree and the author's pre-prediction GT
freeze confirmation.
