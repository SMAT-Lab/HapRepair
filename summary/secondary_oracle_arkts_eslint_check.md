# Secondary oracle check: ArkTS-eslint (lint) findings before/after repair

This is a lightweight, low-cost “second signal” check to mitigate the risk of single-oracle compliance gaming. We re-run CodeLinter with `plugin:@ArkTS-eslint/all` enabled and compare lint findings on (i) the original project snapshots and (ii) the final repaired snapshots (GPT-5.1 Round 5 output).

## Configuration

- Config used: `revision/code-linter-performance-security-arkts-eslint.json5`
- Enabled rule sets:
  - `plugin:@performance/all`
  - `plugin:@security/all`
  - `plugin:@ArkTS-eslint/all`

## Data

- 35 target projects: `revision/target_projects_haprepair.json`
- Baseline logs: `logs/codelinter_openharmony/secondary_oracle_arkts_eslint/baseline/*.log`
- Final logs (GPT-5.1 R5 outputs): `logs/codelinter_openharmony/secondary_oracle_arkts_eslint/after_round5_gpt-5.1/*.log`

## Result (ArkTS-eslint)

- Baseline total `@ArkTS-eslint/*` findings: **0** (35/35 projects)
- Final total `@ArkTS-eslint/*` findings: **0** (35/35 projects)
- Projects with increased lint findings after repair: **0/35**

Notes:
- This check is intended as an *auxiliary* guardrail signal (lint non-regression). It does not replace behavioral testing.
- Performance/security counts are not reported here to avoid confusion with the main frozen-config evaluation logs (tool-version drift can change those totals slightly).

