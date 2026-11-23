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
One resolved and one unresolved trace per task, after a leak check that drops any task whose git history shows commits from after the task was created, and any trace that downloads from the upstream repo: every eligible new-set task (Sep 2024 to Apr 2025) plus 75 sampled old-set tasks (before Jun 2024). 20 tasks are marked `pilot`. `selection_report.json` shows how many tasks survive each filter.

## Pilot: do the tasks run?
```bash
python src/pilot.py export     # where selected_traces.parquet lives -> data/pilot_tasks.jsonl
python src/pilot.py validate   # on an x86 Linux machine with Docker -> data/pilot_validation.csv/.json
```
For each pilot task this runs the hidden human tests with no change (FAIL_TO_PASS should fail), with the golden patch (all should pass), and with each chosen agent patch (compared with the dataset's `resolved`). It also counts commits newer than HEAD in the image. Task images are usually 1 to 3 GB compressed (a few are ~30 GB) and are deleted after each task.

The harness works around a few image quirks: some images installed the repo in editable mode from another path (such as `/borgstore`), so that path is linked to `/testbed`; some carry an installed copy of the package that shadows `/testbed`, so the repo goes first on `PYTHONPATH`; stored agent patches can lack trailing context, so patches apply with `--recount`; the agent's copy of `issue.md` is removed before a patch re-creates it; and pytest runs only the test files the FAIL_TO_PASS and PASS_TO_PASS ids name, since test patches can also add data files. If Docker Hub rate-limits pulls (429), add `{"registry-mirrors": ["https://mirror.gcr.io"]}` to `/etc/docker/daemon.json`.


## Sandbox: clean the task images
```bash
python src/sandbox.py prepare   # on an x86 Linux machine with Docker -> data/sandbox_report.csv
```
The task images hold git history from after each task was created, which can include the real fix. For each pilot task this makes a cleaned image (`ttg-clean/<instance_id>`) whose `/testbed` has one commit with exactly the base files, then checks it: one commit and no other `.git` directories; no file anywhere holds a line (30+ characters) that the human fix or the hidden tests add, unless the base repo already has it; and the hidden tests still fail on the base code and pass with the human fix. Each tester run starts a fresh container from the cleaned image with the network off. Cleaned images are deleted after the checks unless `--keep-images`.

## Tester and grader (Steps C and D)
```bash
python src/pilot.py export        # where selected_traces.parquet lives: also writes data/pilot_traces.jsonl.gz
python src/llm.py "say hi"        # check the model endpoint
python src/tester.py run          # -> data/tester_runs.jsonl, tester_results.csv, tester_summary.json
```
For each pilot trace and condition, a fresh container from the cleaned image gets the agent patch. The tester sees the issue, the agent patch and the changed files (C1). C2 adds related test files and the README. C3, C4 and C5 add a summary, a step list or the raw trace (`src/traces.py`), and X adds the raw trace of a different task with a similar length. The tester writes one pytest file and sees its failures and the coverage of the patch's added lines, for up to 3 rounds; it is told never to weaken an assertion to match the patch. The final file is graded on the original code, the agent patch and the human fix: a test that fails on the agent patch but passes on the human fix flags the patch, which is a bug caught when the patch is wrong and a false alarm when it is right.

The model is any OpenAI-compatible endpoint, set with `TESTER_BASE_URL`, `TESTER_API_KEY` and `TESTER_MODEL` (see `src/llm.py`). The default is `gemini-3.5-flash` on the free Gemini API, with the key in `GEMINI_API_KEY`. The free tier caps Gemini 3.5 Flash at 20 requests a day, so the pilot uses the open-weight Gemma 4, which is free there too but limited to 16k input tokens a minute:
```bash
TESTER_MODEL=gemma-4-26b-a4b-it TESTER_TPM=15000 python src/tester.py run --prompt-size small
```
`--prompt-size small` cuts each prompt section to a fixed budget (the trace to 14k characters, about 4k tokens) and in revision rounds resends the issue, patch, changed files (plus related tests for C2 and up), current test file and failures, without the trace; `TESTER_TPM` paces calls to the per-minute limit. To use a local vLLM server instead: `TESTER_BASE_URL=http://localhost:8000/v1 TESTER_MODEL=openai/gpt-oss-20b`.

Pilot results on the free model, by condition: [docs/pilot_flash_lite.md](docs/pilot_flash_lite.md).

## Azure reasoning-model experiment

The client also accepts `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_VERSION`,
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, and
`AZURE_OPENAI_REASONING_EFFORT`. Supply the key through the environment; do not
commit it. Explicit `TESTER_BASE_URL` settings take precedence over Azure settings.
Reasoning requests use `max_completion_tokens` (default 16,384, configurable with
`TESTER_MAX_COMPLETION_TOKENS`) and omit temperature.

To reproduce the 30-task newer sample, including the 10 newer pilot tasks:

```bash
python src/prepare_new.py --out results/azure-gpt56-high-new30 --tasks 30 --seed 0
DOCKER_DEFAULT_PLATFORM=linux/amd64 python src/tester.py run \
  --data results/azure-gpt56-high-new30 --seeds 1 --validate-images \
  --workers 4 --docker-workers 2
python scripts/compare_runs.py --data results/azure-gpt56-high-new30 \
  --out docs/azure_new30.md
```

Set the Azure environment variables before running the tester. The data directory
holds the selected task IDs, dataset revisions, a credential-free experiment
configuration, image validation records, generated tests from every round, and
results. Resume checks reject configuration changes in an existing experiment;
use a new directory when changing models or prompts. Original pilot outputs stay
in `data/`. The comparison report separates all newer-task results from the paired
comparison on overlapping pilot tasks.

`--workers` overlaps independent tester cases while processing one task image at a
time; `--docker-workers` separately caps fresh test containers. Each model reply
and generation round is checkpointed before proceeding, so a retry can resume
without regenerating completed rounds. Context is read once per agent patch,
trace summaries are shared safely, and concurrent rate-limit retries share a
cooldown. A directory lock prevents two runners from writing the same experiment.
Timing records and API token-cache counts are saved for measuring throughput.
After reviewing a source-only change, `--allow-code-change` permits a resume and
archives the previous configuration. It never permits a model, prompt setting,
or dataset change in the same experiment directory.

`--validate-images` checks the cleaned image for leaks and reproduces the hidden
test baseline, human fix, and both agent labels before making generation calls.
If `image_sizes.json` is present in the experiment directory, the runner uses its
compressed image sizes to reserve estimated extraction space and records tasks
blocked by disk capacity in `task_errors.jsonl`. It does not prune unrelated Docker
resources. The 30-task run's status is in [docs/azure_new30.md](docs/azure_new30.md).
