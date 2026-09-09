# E3 Protocol-Rejection Root Cause

Direct cause: The runner's post-hoc lexical filter rejected each otherwise completed candidate because one shell command string contained a restricted marker.

Root cause: A hidden acceptance-policy mismatch: the Skill tells the agent about HAPREPAIR_AGENT_MODE and unavailable HomeCheck/CodeLinter operations, while the runner treats even mentioning those tool names inside benign env/rg inspection commands as a terminal violation.

| Parent case | Trigger type | Matched marker | API failure | Retry |
|---|---|---|---|---|
| V14-001 | local task/source metadata search | homecheck, codelinter | no | accepted |
| V14-029 | agent-mode environment discovery | codelinter | no | accepted |
| V14-030 | agent-mode environment discovery | codelinter | no | accepted |

The commands inspected only local task/source metadata or agent-mode
environment variables. No forbidden analyzer/controller was invoked.

Claim update: The three outcomes are harness-level lexical protocol rejections, not semantic repair failures or API failures. The main 60/63 result remains the frozen intention-to-treat estimate; the retry-completed result is a separate sensitivity.
