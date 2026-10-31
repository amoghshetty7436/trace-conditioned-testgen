#!/bin/bash
# Full study with Terra (gpt-5.6-terra on Azure OpenAI) as the tester. No GPU: the model runs on Azure,
# and this CPU node only runs the task sandboxes with podman-hpc. Every selected task (data/study_tasks.jsonl,
# from `python src/pilot.py export-study`) x 2 patches x 6 conditions x SEEDS seeds.
#
#   sbatch perlmutter/terra.sbatch               # seed 0: 2,028 runs, about 2 to 3 hours
#   SEEDS=3 sbatch perlmutter/terra.sbatch       # later: adds seeds 1 and 2 (finished runs are skipped)
#
# Needs ~/.ttg_azure.env (TESTER_BASE_URL, TESTER_API_KEY, TESTER_MODEL; see perlmutter/README.md).
# Rerunning after a timeout or crash resumes: finished runs are kept in data/tester_runs_terra.jsonl.
# Overrides: SEEDS=3  FIRST_SEED=1  TAG=terra_s12  SHARDS=16  CONDITIONS=C1,C5  TASKS=a,b  EXTRA=--redo-empty
#SBATCH -A m4999
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J ttg-terra
#SBATCH -o ttg-terra-%j.out
set -uo pipefail
ROOT="${TTG_ROOT:-$SCRATCH/ttg}"
cd "${SLURM_SUBMIT_DIR:-.}"
module load python 2>/dev/null || true

[ -f data/study_tasks.jsonl ] || { echo "data/study_tasks.jsonl is missing: run python src/pilot.py export-study"; exit 1; }
[ -f "$HOME/.ttg_azure.env" ] || { echo "$HOME/.ttg_azure.env is missing; see perlmutter/README.md"; exit 1; }
set -a; source "$HOME/.ttg_azure.env"; set +a
export CONTAINER_CLI=podman-hpc
# Default reasoning effort, as in the pilot; a generous output cap so no answer is cut off (the pilot's
# longest call used about 9k tokens). You pay only for tokens actually generated.
export TESTER_MAX_TOKENS="${MAX_TOKENS:-65536}"
unset TESTER_REASONING_EFFORT
# Docker Hub login made on the login node (podman's default auth file is node-local).
[ -f "$HOME/.config/containers/auth.json" ] && export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"

SEEDS="${SEEDS:-1}"
SHARDS="${SHARDS:-32}"
CONDITIONS="${CONDITIONS:-C1,C2,C3,C4,C5,X}"
TAG="${TAG:-terra}"
FIRST_SEED="${FIRST_SEED:-0}"
LOGS="logs/$TAG-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"
PY="$ROOT/ttg-env/bin/python"
"$PY" src/llm.py "Reply with the word ready." || { echo "Azure smoke test failed; check ~/.ttg_azure.env"; exit 1; }

pids=()
for i in $(seq 1 "$SHARDS"); do
  "$PY" src/tester.py run --set study --check --out-tag "$TAG" --seeds "$SEEDS" --first-seed "$FIRST_SEED" --conditions "$CONDITIONS" \
    ${TASKS:+--tasks "$TASKS"} --shard "$i/$SHARDS" ${EXTRA:-} > "$LOGS/shard-$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a tester process exited with an error; see $LOGS/shard-*.log"; done

"$PY" src/tester.py summarize --out-tag "$TAG"
echo "done: data/tester_runs_$TAG.jsonl, data/tester_summary_$TAG.json, data/task_checks_$TAG.jsonl"
