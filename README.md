# HapRepair

HapRepair provides rule-guided repair for OpenHarmony/ArkTS applications. It combines
HomeCheck findings, rule specifications, repair examples, and a verification-guided
repair loop.

## What is included

All tool resources are under [`skills/haprepair-openharmony-repair/`](skills/haprepair-openharmony-repair/):

- `scripts/homecheck.py`: analyzer configuration, scans, repair plans, and rule lookup.
- `scripts/repair_session.py`: repair sessions, preservation checks, validation, and candidate selection.
- `scripts/*.js`: ArkTS declaration and public-API inspection helpers.
- `references/rule-specs/`: rule specifications used by the tool.
- `references/repair-guides/`: optional rule-indexed repair examples used by the tool.
- `assets/`: analyzer configuration and overlay.
- `SKILL.md`: detailed operating instructions for coding-agent integration.

The current branch distributes the tool, not experiment datasets, annotations,
project snapshots, or run logs. Earlier research snapshots remain in Git history.

## Requirements

- Linux or macOS; Python 3.10+ (standard library).
- Node.js 20+ available as `node` on `PATH`.
- A compatible HomeCheck/DevEco CodeLinter installation. The supplied overlay was
  configured for CodeLinter 6.0.240; verification checks the expected analyzer setup.
- The installation's ArkTS-aware `ohos-typescript` parser.
- The target application's SDK/build tools and a coding agent of your choice.

Set paths for your installation; do not copy another user's machine paths:

```sh
export HAPREPAIR_CODELINTER=/absolute/path/to/codelinter
export HAPREPAIR_TYPESCRIPT=/absolute/path/to/ohos-typescript
python3 skills/haprepair-openharmony-repair/scripts/homecheck.py verify
```

The commands below do not launch a language model. The operator or an integrating
runner invokes the session commands, and the coding agent edits the application
between rounds. Follow [the operating instructions](skills/haprepair-openharmony-repair/SKILL.md)
for the complete repair workflow and agent-mode restrictions.

```sh
python3 skills/haprepair-openharmony-repair/scripts/repair_session.py init-session \
  --workspace /absolute/path/to/app \
  --state-dir /absolute/path/to/repair-session
python3 skills/haprepair-openharmony-repair/scripts/repair_session.py --help
```

Supply project-specific build/test commands through `--build-command` and
`--test-command` when available. Use a dedicated application checkout and a state
directory outside it; preserve your original source before applying repairs.

## Tests

With Node.js and `HAPREPAIR_TYPESCRIPT` configured:

```sh
python3 -m unittest discover -s skills/haprepair-openharmony-repair/scripts -p 'test_*.py'
```

The 30 bundled tests passed in the release check. These tests exercise tool logic;
they do not rerun model experiments or establish application-level correctness.
