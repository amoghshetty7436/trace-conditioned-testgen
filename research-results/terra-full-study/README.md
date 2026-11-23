# Amogh's Terra full-study snapshot

This directory reproduces the Terra study files from commit `44ec6b20afeb790a8c5c625497c89bf0f163967d`. The run contains 1,611 unique seed-0 cases for 135 tasks: 134 tasks have all 12 cases and one has three cases. It is an incomplete full-study snapshot, not a completed 169-task run.

`tester_runs_terra.jsonl` contains generated tests, execution feedback, scores, token counts and other recorded fields. `task_checks_terra.jsonl` contains sandbox validation records; `terra.sbatch` is the original launcher. `manifest.json` records hashes.

The saved records do not contain the tester model's reasoning text or raw API response streams. The original source client only accumulated the final response content. The original coding-agent trajectory inputs for the selected full cohort are in `../raw-traces/study_traces.jsonl.gz`; those are agent work traces, not Terra's own reasoning.
