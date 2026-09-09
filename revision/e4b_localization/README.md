# E4b Controlled Localization

This directory contains the direct-API runner for the frozen 18-file,
38-positive benchmark. Formal model conditions are `gpt-5.6-luna`,
`gpt-5.6-sol`, and `deepseek-v4-flash`.

The runner expects `OPENAI_API_BASE`, `OPENAI_API_KEY`, `DS_API_BASE`, and
`DS_API_KEY` in the environment. It never persists their values. Run from the
workspace root after sourcing `.env`:

```bash
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/run_localization.py --mode validate
PYTHONNOUSERSITE=1 python -m unittest discover -s HapRepair/revision/e4b_localization -p 'test_*.py'
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/run_localization.py --mode smoke --run-id exp_loc_18_three_model_smoke_01 --workers 3
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/run_localization.py --mode formal --run-id exp_loc_18_three_model_v1 --workers 9
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/audit_localization_run.py
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/evaluate_homecheck_condition.py
```

An explicitly authorized retry must use a new run ID and a frozen-file subset,
so the original response remains immutable:

```bash
PYTHONNOUSERSITE=1 python HapRepair/revision/e4b_localization/run_localization.py --mode formal --run-id exp_loc_18_three_model_v1_deepseek_parse_retry_01 --models deepseek-v4-flash --files entry/src/main/ets/MainAbility/pages/List/ListLevel1.ets --max-output-tokens 8192 --json-object-output --bounded-retry-prompt --workers 1
```

Primary paper-facing inference is line-free detection over 30 unique file-rule
positives. HomeCheck also detects all 38 manually constructed instances. Precision
and F1 are conditional on the selected 18 files and eight-rule closed set;
real-project recall remains unknown.
