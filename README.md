# Trace-Conditioned Test Generation

CS 6158 project (Amogh Shetty, Ishaan Shah). Does a coding agent's trace help a separate agent write tests that catch the agent's own bugs?

## Data
We join two public Hugging Face datasets on `instance_id`:
- `nebius/SWE-rebench-openhands-trajectories`: trace, agent patch, `resolved` label (67,074 rows, ~2.1 GB)
- `nebius/SWE-rebench`: issue, golden patch, human tests, Docker image (27,878 rows, ~245 MB)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/join.py --limit 2000   # quick smoke test
python src/join.py                # full join -> data/combined_light.parquet, data/join_report.json
```
`data/*.parquet` is git-ignored (too large for GitHub). `data/join_report.json` holds the sanity counts.

## Step A: pick the tasks
```bash
python src/select_tasks.py        # -> data/selected_traces.csv, selected_traces.parquet, selection_report.json
```
One resolved and one unresolved trace per task, after a leak check that drops any task whose traces name a hidden test or show the golden fix in git history output: every eligible new-set task (Sep 2024 to Apr 2025) plus 75 sampled old-set tasks (before Jun 2024). 20 tasks are marked `pilot`. `selection_report.json` shows how many tasks survive each filter.
