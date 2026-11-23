# DeepSeek comparison on Perlmutter

This experiment targets all 169 selected tasks, two stored agent patches per task,
conditions C1–C5 and X, and replicate label 0: 2,028 planned cases before validation
exclusions. It uses the user's Azure deployment `DeepSeek-V4-Flash`, with thinking
enabled through the verified `reasoning_effort=high` parameter and a 65,536-token
output cap. Azure rejects the native DeepSeek `thinking` field. The deployment is
not assumed to be identical to today's `deepseek-flash` alias on api.deepseek.com.
The API has no documented seed control; replicate 0 is not deterministic sampling.

The task prompts, full character budgets, three-round maximum, and stored patches
match the Terra study. C3 summaries come from the respective tester model. Every
shard retains the full trace pool so X's donor selection is independent of sharding.
The `terra` validation policy matches Amogh's task inclusion rule; stricter hidden-test
diagnostics are retained. The new runner excludes empty/failed grading executions.
The comparison report re-scores both models from saved statuses, requires explicit
agent-test failures plus golden-test passes, and reports identical paired cases as
well as all completed DeepSeek cases. This is a controlled model comparison with
documented harness improvements, not a claim that the two executions are identical.

## Allocation and files

- NERSC user: `ics35`; CPU account: `m4999`; verified 10 allocated CPU node-hours,
  zero previously charged at setup. GPUs and EC2 are not used.
- Root: `/pscratch/sd/i/ics35/ttg-deepseek-20261031`.
- `inputs/`: full tasks and traces, selection CSV, pinned Hugging Face revisions,
  SHA-256 manifest. Historical `pilot_*` filenames contain all 169 tasks.
- `code/`: source snapshot and Terra reference at commit
  `44ec6b20afeb790a8c5c625497c89bf0f163967d`.
- `run/`: manifest, submission ledger, eight shard directories, per-round reply
  checkpoints with request messages, generated tests, summaries, grades, logs,
  execution timing, and accounting. Up to 32 API requests / 16 test containers.
- `requirements.lock.txt`: installed dependencies; `api_probe.json`: verified
  deployment response and quota headers, no credential.
- API key: separate owner-only file in `/pscratch/sd/i/ics35/ttg-private`, never
  included in source or result archives. Home is over quota; do not write results there.

One regular CPU node is capped at 9h30m, with no requeue. The workers stop after
9h10m to leave cleanup time. `submit_deepseek.py` refuses a second submission to
the same run directory; do not remove the ledger to bypass the budget. Any retry
requires checking `sacct`, `squeue`, and `iris user ics35` first and must keep total
charged CPU node-hours below 10. Queue waiting is not allocated runtime. The time
cap limits expenditure; it does not guarantee all 2,028 cases finish.

## Commands (from the uploaded code directory)

```bash
source perlmutter/deepseek.env.example
export HF_HOME=/pscratch/sd/i/ics35/ttg-deepseek-20261031/cache/huggingface
../env/bin/python perlmutter/export_study.py --out ../inputs
../env/bin/python perlmutter/run_deepseek.py prepare --inputs ../inputs --run ../run
../env/bin/python perlmutter/submit_deepseek.py --run ../run --python ../env/bin/python \
  --key /pscratch/sd/i/ics35/ttg-private/deepseek.key --account m4999
```

Do not modify code or model settings after preparing the manifest. Runtime checks
reject changed code. Coverage wheels are downloaded before submission. Container
pulls serialize to avoid podman-hpc state corruption; images are removed after
each task, and no global prune runs during the experiment.

After the job ends, refresh accounting, regenerate the report, and copy `run/`,
`inputs/`, `code/`, the source archive, probe, and dependency lock to the local
`results/deepseek-perlmutter-20261031/` directory. Preserve the raw shard files even
when the merged summary is incomplete. Scratch is temporary storage, not a backup.
Checkpoints avoid repeating persisted API replies, but an interrupted request may
have incurred usage that was not saved. Summaries count C3 usage separately from
test-generation rows. Report completion, exclusions, actual charged node-hours,
and any remaining cases; do not present missing cases as measured successes.

## Verified API references

- [DeepSeek request parameters](https://api-docs.deepseek.com/api/create-chat-completion/)
- [Azure DeepSeek integration](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/tutorials/get-started-deepseek-r1)
- [NERSC resource charging](https://docs.nersc.gov/policies/resource-usage/)

The Azure smoke test, not the native-provider documentation alone, determines the
actual request format for this deployment. Reasoning effort labels are provider
specific: DeepSeek high is not asserted to equal Terra medium in compute or cost.
