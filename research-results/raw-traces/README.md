# Raw traces

`pilot_traces.jsonl.gz` and `study_traces.jsonl.gz` contain the original selected coding-agent trajectories used as input. The pilot and main-study X conditions use different donor pools; do not pool them without accounting for this difference.

The `kimi-pilot-api-*.tar.gz` and `kimi-study-api-*.tar.gz` files contain the raw external-recorder attempt directories: `request.json`, `response.sse`, and any `parsed.json` and `metadata.json` produced for each attempt. The manifest lists per-archive SHA-256 hashes and counts. Attempt directory names correspond to IDs in the extracted `reasoning-traces.jsonl.gz` files in the parent experiment directories. HTTP authentication headers were not recorded.

These files are raw research data; model-generated reasoning is not necessarily a faithful explanation of model behavior. The Kimi retry produced no successful generation records.
