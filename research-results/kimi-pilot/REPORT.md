# Kimi Azure pilot

Job `61370723`: **COMPLETED**; 240/240 cases saved.

Amogh unchanged commit `44ec6b20afeb790a8c5c625497c89bf0f163967d`; 20 tasks, seed 0, six conditions.
Azure deployment `Kimi-K3.1-Pro`; smoke response model `Kimi-K3.1`; reasoning high; output cap 65,536.

| Condition | Kimi catches | DeepSeek catches | Terra catches | Kimi false alarms | DeepSeek false alarms | Terra false alarms |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 8/20 | 4/20 | 5/20 | 2/20 | 1/20 | 1/20 |
| C2 | 10/20 | 4/20 | 6/20 | 2/20 | 0/20 | 1/20 |
| C3 | 9/20 | 2/20 | 8/20 | 2/20 | 0/20 | 3/20 |
| C4 | 9/20 | 6/20 | 6/20 | 3/20 | 0/20 | 3/20 |
| C5 | 9/20 | 5/20 | 6/20 | 3/20 | 0/20 | 2/20 |
| X | 9/20 | 5/20 | 5/20 | 3/20 | 0/20 | 1/20 |

Only matching saved cases are compared. Primary scores preserve the upstream harness; missing cases are not counted as completed.
Task validations: 20; exclusions: 0; complete tasks: 20/20.
Generation failures with no test code: 13; flags without explicit failing-test evidence: 0.

Recorded API calls: 523; with reasoning text: 523; output-limit finishes: 0.
Recorded API usage (includes summary calls): 9,596,101 input and 2,150,921 completion tokens, including 1,647,658 reported reasoning tokens.

Raw API requests, SSE response bodies, extracted reasoning/final text and usage are in `api-traces/`. Authentication headers are not recorded.
The external recorder forwards request and response bodies unchanged. The original harness replays final assistant content only; captured reasoning is not injected into subsequent prompts. This preserves the earlier protocol but differs from Kimi's recommendation to replay reasoning in multi-turn conversations.
Baseline caveat: the DeepSeek pilot has one skip-related ambiguous catch and two generations without test code. Repeated condition-level observations are not independent tasks. Reasoning labels do not establish equal inference compute.

**Pilot only. Do not start the full study without explicit approval after review.**

Slurm accounting:
```
JobID|Account|State|ElapsedRaw|AllocNodes|Timelimit|ExitCode
61370723|m4999|COMPLETED|3293|1|02:00:00|0:0
```

Final integrity audit: passed.
