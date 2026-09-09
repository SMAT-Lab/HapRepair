# EXP-INDEP-63 Generation

This directory freezes and runs the single-generation model stage of the
independent patch-correctness experiment. Generation reads only the defective
benchmark, its frozen CodeLinter findings, and the read-only 383-pair retrieval
corpus. It does not read the benchmark reference-repair project.

Prepare the 63 prompts and deterministic same-rule Top-1 retrieval results:

```bash
python revision/independent_oracle/generation/run_generation.py prepare \
  --run-id exp_indep_63_gpt_5_1_frozen_01 \
  --device cuda:0
```

Run the excluded API smoke test, then the formal sequential run:

```bash
python revision/independent_oracle/generation/run_generation.py smoke \
  --smoke-id exp_indep_63_gpt_5_1_smoke_01 \
  --model gpt-5.1

python revision/independent_oracle/generation/run_generation.py execute \
  --run-id exp_indep_63_gpt_5_1_frozen_01 \
  --model gpt-5.1 \
  --fail-fast
```

Verify the prepared inputs at any point. Add `--require-complete` only after all
63 accepted responses exist:

```bash
python revision/independent_oracle/generation/run_generation.py verify \
  --run-id exp_indep_63_gpt_5_1_frozen_01 \
  --require-complete
```

The original `protocol.json` / GPT-5.1 run is retained as blocked evidence.
After the endpoint returned `model_not_found`, the user explicitly approved the
model-only amendment in `protocol_gpt_5_6_luna.json`. The replacement commands
add:

```bash
--protocol revision/independent_oracle/generation/protocol_gpt_5_6_luna.json
```

and use run ID `exp_indep_63_gpt_5_6_luna_frozen_01` with model
`gpt-5.6-luna`.

The runner loads `OPENAI_API_KEY` and `OPENAI_API_BASE` from the workspace
`.env`. It records only hashes of the endpoint and environment file and uses no
automatic API retries.

## Final v14 static-Skill condition

The Top-1 runner and its completed outputs above are immutable predecessor
evidence. The paper-facing final-v14 condition uses a separate runner and run ID:

```bash
python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  validate-protocol \
  --output revision/independent_oracle/generation/protocol_v14_static_skill_gate.json

python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  prepare --run-id exp_indep_63_v14_static_skill_luna_01

python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  verify --run-id exp_indep_63_v14_static_skill_luna_01

python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  smoke --smoke-id exp_indep_63_v14_static_skill_luna_smoke_01
```

Only after those gates pass may the 63-case run start:

```bash
python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  execute --run-id exp_indep_63_v14_static_skill_luna_01

python revision/independent_oracle/generation/run_generation_v14_static_skill.py \
  verify --run-id exp_indep_63_v14_static_skill_luna_01 --require-complete
```

Each case receives an isolated workspace containing only its defective files and
frozen finding metadata. The v14 Skill is installed in a case-local `CODEX_HOME`;
its semantic specifications and deterministic static guides are readable but are
not injected into the task prompt. The raw 383-pair corpus, human repair, historical
outputs, adjudication package, HomeCheck/CodeLinter toolchain, and post-edit feedback
are not mounted. One Codex agent invocation produces at most one accepted candidate,
and there is no automatic or silent retry.

After the immutable generation and read-only audits, the selected E3-05D policy
admits `V14-016` through transparent capture recovery and retains the three
restricted-command rejections as automatic failures. Build and verify the
separate 60-candidate blind package with:

```bash
python revision/independent_oracle/generation/prepare_adjudication_v14_static_skill.py \
  prepare \
  --run-id exp_indep_63_v14_static_skill_luna_01 \
  --package-id exp_indep_final_static_blind_01

python revision/independent_oracle/generation/prepare_adjudication_v14_static_skill.py \
  verify --package-id exp_indep_final_static_blind_01
```

The authors label only the 60 remapped candidates. After both author CSVs and
any required third-author rows are complete, validate and summarize with:

```bash
python revision/independent_oracle/summarize_adjudication_v14_static_skill.py
```

The final summarizer uses 63 as the correctness denominator and 60 as the
agreement/kappa denominator. It does not reuse the old Top-1-RAG labels or
claims.
