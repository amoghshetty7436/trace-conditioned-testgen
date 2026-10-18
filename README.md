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
python src/join.py                # full join -> data/combined.parquet, data/join_report.json
```
`data/*.parquet` is git-ignored (too large for GitHub). `data/join_report.json` holds the sanity counts.
