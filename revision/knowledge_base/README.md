# HapRepair Knowledge-Base Audit

This directory reconstructs the repair-pair corpus from the XLSX files that
remain in the public HapRepair repository. It does not infer deleted records or
alter the source workbooks.

## Rebuild

From the HapRepair repository root, run:

```bash
python revision/knowledge_base/build_manifest.py
```

The script requires `pandas` and an Excel engine supported by pandas. Output is
deterministic for unchanged source workbooks and script settings.

## Dataset views

- `published_378.jsonl` uses the four workbooks whose 426 raw rows underlie the
  previously reported corpus size: `vul_pairs.xlsx` (269), `output.xlsx` (60),
  `security_pairs.xlsx` (96), and `addition.xlsx` (1). Exact deduplication leaves
  378 pairs covering 58 rule IDs.
- `rule_complete_383.jsonl` additionally uses `missing_pairs.xlsx` (5). Exact
  deduplication leaves 383 pairs covering 63 rule IDs. This is the selected
  paper-facing corpus and the input view for new revision experiments.
- `canonical_pairs.jsonl` is the canonical union and currently matches the
  rule-complete view.

These are separate evidence views. The surviving data does not support the
combined claim "378 unique pairs covering 63 rules." The authors selected the
383-pair/63-rule view; the 378-pair/58-rule view is retained only to document
the provenance of the original 426-row count.

## Deduplication

For exact matching, the script converts CRLF and CR line endings to LF, strips
outer whitespace, and hashes the normalized tuple:

```text
(rule, problem_code, repair_code)
```

Descriptions, explanations, and stored diffs are not part of the identity key.
All original workbook occurrences are retained in `source_occurrences`.

`near_duplicate_candidates.json` is only a review queue. It compares pairs
within the same rule and records candidates whose mean problem-code and
repair-code similarity is at least 0.80. No candidate is removed automatically.

## Outputs

- `source_audit.json`: workbook hashes, row counts, rule counts, and view totals.
- `canonical_pairs.jsonl`: canonical union with provenance and view membership.
- `published_378.jsonl`: exact-deduplicated view of the original 426 rows.
- `rule_complete_383.jsonl`: exact-deduplicated 63-rule view.
- `exact_duplicate_groups.json`: every exact duplicate and all source rows.
- `near_duplicate_candidates.json`: same-rule candidates for manual review.
- `rule_coverage.csv`: per-rule pair counts in both views.

## Category caveat

Categories are derived only from rule-ID namespaces. In the 63-rule view, the
surviving files contain 42 `@performance` rules, 20 `@security` rules, and one
`@hw-ets-eslint` rule. This differs from the manuscript's 44-performance and
19-security taxonomy and must be reconciled by the authors before selecting one
held-out pair per rule.
