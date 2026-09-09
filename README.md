# HapRepair

Research artifacts for **HapRepair: Automated Defect Repair for OpenHarmony Apps**.
HapRepair combines HomeCheck analysis with rule-guided coding-agent repair.

## TOSEM revision artifacts

Start with [the revision artifact guide](artifacts/tosem-2026-revision/README.md).
The release includes the 383-pair repair corpus, the evaluated repair configuration,
experiment runners, detection and human-assessment records, and the final paired
comparison across 35 projects and two repetitions.

```sh
python3 artifacts/tosem-2026-revision/verify_artifact.py
```

This command verifies published file hashes and selected result totals offline;
it does not launch agents or incur API charges.

## Earlier experiments

The original project directories remain available. See
[the original README](artifacts/tosem-2026-revision/README_conference.md) for the
earlier retrieval experiments and setup. Historical version labels, prompts,
protocols, and result files are retained as recorded; use the revision guide to
identify the final reporting set.

HomeCheck: <https://github.com/SMAT-Lab/HomeCheck>.
