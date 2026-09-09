# TOSEM revision artifact guide

## Final reporting set

| Material | Entry point (relative to repository root) |
|---|---|
| Repair corpus: 383 pairs, 63 rule identities | `revision/knowledge_base/rule_complete_383.jsonl` |
| Evaluated rule-guided configuration | `skills/haprepair-openharmony-repair-v14-dev/SKILL.md` |
| Experiment runners and protocols | `revision/hapskill_experiment/` |
| Project source/version manifests | `revision/coding_agent_baseline/source_manifest_35_gitcode.json` |
| Controlled detection, three LLM conditions | `artifacts/tosem-2026-revision/evidence/e4b_localization/exp_loc_18_three_model_v1/` |
| Real-project alert assessment: 231 alerts | `artifacts/tosem-2026-revision/evidence/rq1_precision/exp_rq1_precision_v1/` |
| 35-project repair study: 9,884 to 39 alerts | `artifacts/tosem-2026-revision/evidence/e1_v14/aggregate.json` |
| Controlled repair assessment: 63 cases | `revision/independent_oracle/final_evaluation_v1/` |
| Matched comparison: 70 pairs, 70 vs. 58 accepted candidates | `artifacts/tosem-2026-revision/evidence/full_pair_35/campaign_02_results/` |
| Matched comparison source run manifests | `artifacts/tosem-2026-revision/run-manifests/rq4/` |
| Frozen HomeCheck source changes | `artifacts/tosem-2026-revision/evidence/source_archives/d03_scientific_code_20260807/` |

## Verification and execution

Run `python3 artifacts/tosem-2026-revision/verify_artifact.py` from the repository
root (Python 3.9+, standard library only). It checks every copied publication file,
the corpus size, the final comparison rows against their archived manifests, and
key aggregate totals. This is an artifact consistency check, not a rerun of model
generation or human annotation.

For generation, consult the runner's `--help`, the frozen JSON protocol, and the
component README. Re-create project checkouts from the source manifests, install
the recorded HomeCheck/CodeLinter and SDK dependencies, and configure your own API
credentials outside Git. Protocols contain original absolute server paths; map
these paths to your workspace before a new run and retain a separate run record.
Do not overwrite the archived protocols or results. Agent runs may incur charges.

The matched comparison uses `protocol-full-pair-skill-v16.json` (with recorded
technical replacements) and the baseline v6 protocol. In these archived runs, the
experiment runner invokes the bundled planning/validation/candidate-selection
machinery; the coding agent edits project files and supplies a completion report.
The frozen prompts restrict direct analyzer/controller invocation by the agent.
The code and protocols preserve this actual execution arrangement.

## Provenance and scope

`publication_manifest.json` maps each copied file to its original server-relative
source, byte count, and SHA-256 digest. The source checkout had additional
uncommitted experiment files; the manifest identifies the exact published bytes,
not just the older checkout commit.

Final comparison rows select the canonical source manifests, including technical
replacements; earlier run names alone do not define the final set. The controlled
63-case assessment preserves its three harness-false-rejection replacements,
the original generation records, both annotation packages, and the merge decision.
Historical `378` corpus views, older protocol names, and preflight statistics remain
raw provenance rather than alternative final results.

Caches, credentials, installed SDKs, agent runtime homes, and full environment
copies are omitted. The publication provides canonical manifests and selected
evidence, not every server workspace or every pilot trace. Source archives include
the frozen HomeCheck patch reconstruction instructions. Original conference-era
files remain in the repository and are documented in `README_conference.md`.

## Historical test conditions

The offline publication verifier is self-contained. Some archived integration
tests additionally depend on the original machine layout: the HomeCheck source
binding check distinguishes `/home/zhihao/hdd/haprepair` from the migrated
`/data/zhihao/haprepair` path. Re-establish an explicit workspace mapping for a
fresh integration run; do not edit the original recorded workspace merely to
make the check pass. The archived oracle test
`test_incomplete_author_labels_are_rejected` copies the annotation package and
assumes its labels are initially blank, whereas the released package contains
completed labels. To exercise that negative test, blank labels in its temporary
fixture, never in the published annotation CSVs. These historical tests are not
claimed to pass unchanged in an arbitrary relocated checkout.
