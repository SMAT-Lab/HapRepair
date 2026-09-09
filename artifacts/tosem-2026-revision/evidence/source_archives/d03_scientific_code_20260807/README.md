# D-03 Scientific Code Archive

This archive closes the versioning gap for paper-facing E1/E2 analysis code and
RQ1 native-range artifacts without changing either repository's Git index.

## Repository anchors

- HapRepair: `822693528ab33062eaa8e0511083004455a1a079` (`main`)
- HomeCheck: `696a50798301cac11f3878febfbf909ee18a70aa` (`master`)

## Contents

- `haprepair_e1_e2_untracked_sources.tar.gz`: 23 E1/E2 analysis, runner,
  protocol, incident-recording, and test files under
  `revision/hapskill_experiment/`.
- `haprepair_rq1_range_untracked_package.tar.gz`: 136 non-ignored RQ1 range
  source, test, frontend, manifest, report, and sidecar files under
  `revision/rq1_precision/`.
- `homecheck_head_to_worktree.patch`: all 33 tracked HomeCheck changes relative
  to the frozen HomeCheck HEAD, independent of index state.
- `homecheck_staged.patch`: the nine changes staged at archival time.
- `homecheck_unstaged.patch`: the 24 tracked changes not staged at archival
  time.
- `homecheck_untracked_sources.tar.gz`: `DefectRangeUtils.ts` and
  `DefectRange.test.ts`.
- `manifest.json`: hashes, counts, reconstruction contract, and evidence
  boundary.

The complete HomeCheck source state is reconstructed by checking out the
recorded HomeCheck HEAD, applying `homecheck_head_to_worktree.patch`, and then
extracting `homecheck_untracked_sources.tar.gz` at repository root. The staged
and unstaged patches are retained only to preserve the original index split;
they must not be applied in addition to the complete patch.

The archive records exact source state. It does not convert uncommitted work
into an upstream HomeCheck release, and it does not alter the frozen CodeLinter
6.0.240 overlay used by the completed experiments.
