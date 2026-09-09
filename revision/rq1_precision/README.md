# EXP-RQ1-PRECISION

This directory freezes and validates the revised HomeCheck precision sample.

The design uses 38 performance-rule strata and three security rules. Performance
receives 170 samples with a minimum of three per non-census rule. Security uses
fixed rule targets of 30/30/1 and project strata with a minimum of two per
non-census project stratum. Remaining slots use deterministic Hamilton allocation;
within-stratum selection uses a frozen SHA-256 rank.

Commands:

```bash
python3 freeze_sample.py
python3 validate_package.py
python3 -m unittest -v test_rq1_precision.py
python3 summarize_annotations.py
```

The summarizer must not run successfully until both authors have independently
completed all 231 labels and the third author has resolved every disagreement.

## Source-range provenance gate

The frozen v1
findings contain only the line and column emitted by CodeLinter 6.0.240; every
`end_line` falls back to `line`. The frontend's former blue delimiter-balanced
context was therefore a navigation heuristic, not a HomeCheck/ArkAnalyzer defect
range, and has been removed. The yellow reported-line highlight remains.

The updated standalone HomeCheck model now carries explicit native
`rangeStartLine`/`rangeStartColumn` and `endLine`/`endColumn` fields plus range
provenance. Representative rules select AST, IR statement, field declaration,
class declaration, or complete lexical comment-block ranges. CodeLinter 6.0.240,
its frozen compatibility overlay, and every experimental artifact remain unchanged.
`build_range_sidecar.py` strictly aligns updated HomeCheck output to the frozen
finding identity and rejects missing, legacy, or ambiguous matches. This sidecar is
annotation-display metadata only; it never modifies case JSON or labels.

Generate a sidecar after scanning the relevant frozen projects with updated
HomeCheck:

```bash
python3 build_range_sidecar.py \
  --homecheck-commit 696a5079... \
  --report PROJECT=/path/to/issuesReport.json \
  --output /path/to/annotation_range_sidecar.json
```

Do not edit the frozen v1 package or existing annotation CSVs in place. Archive the
scan inputs, exact-match coverage, unmatched reasons, and sidecar hash before using
the blue range display.

The current archived run is `range_scan_runs/native_range_exact_v10`. It scanned the
15 frozen projects needed by the enabled native-range rules. Forty-four of 231 sampled
findings aligned by the exact six-field frozen identity: 11 multiline IR statements,
three multiline AST nodes, and 30 complete parser-confirmed comment blocks. All 30
frozen `@security/no-commented-code` findings now have native ranges. The remaining
187 findings have no matched native range and remain reported-line only;
no fallback or location approximation is applied. In particular, RQ1-005 retains
the frozen CodeLinter location at line 87 while its HomeCheck display range covers
the complete comment block from `40:1` through `105:68`. RQ1-008 retains its parent
WaterFlow report location at `130:7`, while the native range covers the actual
`LazyForEach` statement at `131:9` through `139:35`. RQ1-009 retains its frozen
location at `295:5`, while its complete comment block spans `253:5` through
`311:28`. The sidecar SHA-256 is
`1795b16c33a15d630d564651d27cfbcb4bbe8f99da578a9fdabf80501a6cd6c4`.

## Local annotation interface

The default server exposes one shared frontend with role buttons for both authors
and the adjudicator:

```bash
python3 annotation_app/server.py --port 8771 \
  --range-sidecar range_scan_runs/native_range_exact_v10/annotation_range_sidecar.json
```

Each role writes a separate CSV. Authors must use only their assigned role during
independent annotation. For operationally separate servers, pass `--role author_1`,
`--role author_2`, or `--role adjudicator` and use separate ports.

The adjudicator remains locked until both author files contain all 231 labels and
then receives only disagreements. Once both authors are complete, both author files
are frozen. Annotation writes are atomic. Run the interface tests with:

```bash
python3 -m unittest -v test_build_range_sidecar.py annotation_app/test_server.py
```
