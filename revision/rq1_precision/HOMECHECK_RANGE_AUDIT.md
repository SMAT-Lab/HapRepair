# HomeCheck source-range audit

Date: 2026-08-07

Status: annotation-only transport implemented and frozen-source coverage archived

## Scope

- Paper-pinned HomeCheck source: `461ad0a2f3a71a22ceeb18f7cc4a9fa67c986c59`
- Latest audited upstream HomeCheck source: `696a5079`
- CodeLinter: `6.0.240`
- Frozen precision package: `exp_rq1_precision_v1`

The latest HomeCheck source was fetched and fast-forwarded before this audit. Local
rule compatibility fixes were preserved separately and reapplied to the new matcher
implementations. This source update does not change the paper-pinned scanner or any
frozen experiment artifact.

## Findings

1. The 231 frozen findings contain no native multiline range. Every record has
   `end_line == line` because the CodeLinter JSON report omits `endLine` and
   `endColumn`, and the population normalizer falls back to the start coordinates.
2. HomeCheck `Defects` exposes `reportLine` and `reportColumn`. Its constructor
   receives an `endColumn`, but stores that value only inside the opaque `fixKey`;
   there is no `endLine` field. This is unchanged between the pinned and latest
   audited HomeCheck revisions.
3. ArkAnalyzer does retain four-coordinate `FullPosition` values for AST nodes and
   statement operands. However, `Stmt.getOriginPositionInfo()` returns only a start
   `LineColPosition`. `Stmt.getOriginalText()` is mapped from the associated source
   node and can support an exact node end calculation, but only when that whole node
   is the intended diagnostic entity.
4. Several HomeCheck rules already call `getLastCol()` on a `FullPosition` and pair
   it with `getFirstLine()` when constructing `Defects`. If the position spans lines,
   the resulting single-line tuple is incomplete or internally misleading.
5. CodeLinter's JSON formatter maps each defect only to `line`, `column`, `severity`,
   `message`, and `rule`. It remains deliberately unchanged because CodeLinter
   6.0.240 is still the detector for all later experiments and paper statistics.

The removed blue frontend context used delimiter balancing and chained-call syntax.
It was not analyzer output and must not be used as annotation evidence.

## Required design

Add explicit public start and end properties to `Defects` without changing existing
constructor behavior, `fixKey`, `mergeKey`, or fix-engine parsing. A final optional
range object is safer than inserting new positional parameters into the existing
constructor. Legacy callers should default the start to the report location and
`endLine` to `reportLine`, while retaining their current third-argument end column.

Range selection must be rule-specific:

- AST rules should report the `FullPosition` of the semantic target node.
- IR rules should prefer the matched operand's `FullPosition`; use the mapped source
  node range only when the entire statement is the diagnostic target.
- View-tree rules should trace the relevant create/attribute statement back to an
  operand or AST range rather than balance source delimiters.
- A frozen report location may identify a parent container while the native semantic
  range identifies a child statement. Exact finding identity, same-file provenance,
  valid ordering, and frozen-source bounds are required; artificial containment of
  the report point within the semantic range is not.
- Field and class rules should distinguish identifier-only findings from findings on
  the whole declaration.
- File/graph findings such as dependency cycles must not be expanded to arbitrary
  multiline constructs merely because surrounding source is available.

Propagate both end coordinates through standalone HomeCheck output only. Align them
to frozen CodeLinter findings in a separate annotation-display sidecar. Do not patch
or version-bump the paper-pinned CodeLinter overlay for this convenience feature.

## Implemented annotation path

- `Defects` now serializes `rangeStartLine`, `rangeStartColumn`, `endLine`,
  `endColumn`, and `rangeSource`; legacy callers preserve the old constructor
  contract, `fixKey`, and `mergeKey` behavior.
- `DefectRangeUtils` converts ArkAnalyzer `FullPosition`, mapped IR statement text,
  field text, and class text into 1-based ranges with an exclusive end column.
- Representative AST, IR, view-tree, field, and class rules now attach native ranges.
- `@security/no-commented-code` enumerates comment delimiters and confirms them with
  TypeScript's parser-aware comment API before grouping physically contiguous line
  comments. This preserves parser context after template substitutions and through
  syntax-recovery regions without treating string, template, or regular-expression
  text as comments. A block is retained when the complete block or at least one of
  its physical lines parses as code, so ArkTS DSL fragments still receive the full
  semantic comment range.
- `@performance/hp-arkui-set-cache-count-for-lazyforeach-grid` keeps the frozen
  parent-container location but derives its range from the same-file `LazyForEach`
  create statement.
- `build_range_sidecar.py` accepts per-project standalone HomeCheck reports and
  matches only the exact frozen six-field identity. It skips legacy ranges and
  rejects duplicate matches.
- The annotation server validates sidecar purpose, detector identity, blind IDs,
  start coordinates, ordering, and provenance. It returns `display_range` separately
  from the immutable `finding` object.
- The frontend uses yellow only for the frozen CodeLinter report line and blue only
  for a validated sidecar range. Without a sidecar it shows no blue range.

This mechanism is not a scanner condition, does not alter alert identity, and is not
used by later CodeLinter experiments or manuscript statistics.

## Current frozen-source result

`range_scan_runs/native_range_exact_v10` ran the range-enabled rules with exact
CodeLinter-compatible IDs on 15 frozen project source trees. The scan manifest
records source commits, tree OIDs, report hashes, HomeCheck commit
`696a50798301cac11f3878febfbf909ee18a70aa`, and source diff SHA-256
`bf5ea20f9c1cbdd3c3ac77896450b905d79872806d4b80abcbe100fd2a81a1d0`.

The strict sidecar accepted 44 of 231 sampled findings: 11 IR-statement ranges,
three AST-node ranges, and 30 parser-confirmed comment-block ranges. All 30 frozen
`@security/no-commented-code` findings matched exactly; the other 187 sample cases
were not assigned a native range. No fallback or location approximation was applied.
RQ1-005 is the regression case for a finding inside a block: CodeLinter
remains at `87:1`, while HomeCheck supplies the complete `comment-block` range
`40:1` to `105:68`. RQ1-008 is the parent/child regression case: CodeLinter remains
at the WaterFlow location `130:7`, while HomeCheck supplies the `LazyForEach`
IR-statement range `131:9` to `139:35`. RQ1-009 exercises parser context after a
template substitution: CodeLinter remains at `295:5`, while HomeCheck supplies the
complete comment block `253:5` to `311:28`. The sidecar SHA-256 is
`1795b16c33a15d630d564651d27cfbcb4bbe8f99da578a9fdabf80501a6cd6c4`.

The archived v7 run is not annotation-eligible: HomeCheck logged an invalid
compatibility rule ID while returning exit code zero. The scan runner now extracts
such log entries into `invalid_rules` and fails the run instead of accepting an
empty report. v10 contains no invalid-rule entries or failed project scans.

## Validation status

Annotation may proceed with this sidecar. Every case retains its exact yellow
CodeLinter location, only 44 strictly matched native ranges receive blue display,
and all other cases safely remain reported-line only. Model, standalone JSON, AST,
IR, view-tree, field, class, comment-block, sidecar, server, and frontend tests pass.
The frozen CodeLinter installation and overlay also pass identity verification.

Broader native-range coverage remains follow-up engineering, not an annotation
blocker. Each additional rule still requires coordinate checks, an exact frozen
identity match, and an updated archived sidecar. File/graph rules must remain
single-location unless the analyzer itself provides a meaningful range.

Existing v1 files and all current author labels must be preserved throughout this
work. They may be audited or re-reviewed after the version decision, but never
silently overwritten or reset.
