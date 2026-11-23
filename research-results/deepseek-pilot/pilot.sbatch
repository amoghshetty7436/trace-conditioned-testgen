#!/bin/bash
# Operational launcher only; all experiment code is Amogh's unmodified commit.
#SBATCH -A m4999
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH -J ttg-ds-pilot
#SBATCH -o /pscratch/sd/i/ics35/ttg-deepseek-pilot-amogh-20261031/logs/slurm-%j.out
set -uo pipefail
umask 077
ROOT=/pscratch/sd/i/ics35/ttg-deepseek-pilot-amogh-20261031
PY=/pscratch/sd/i/ics35/ttg-deepseek-20261031/env/bin/python
cd "$ROOT/repo" || exit 1
[ "$(git rev-parse HEAD)" = 44ec6b20afeb790a8c5c625497c89bf0f163967d ] || exit 1
git diff --exit-code HEAD -- src perlmutter || exit 1
export CONTAINER_CLI=podman-hpc PYTHONUNBUFFERED=1
export TESTER_BASE_URL=https://ishaan-oai.services.ai.azure.com/openai/v1
export TESTER_MODEL=DeepSeek-V4-Flash TESTER_REASONING_EFFORT=high TESTER_MAX_TOKENS=65536
export TESTER_API_KEY="$(cat /pscratch/sd/i/ics35/ttg-private/deepseek.key)"
export PIP_CACHE_DIR=/pscratch/sd/i/ics35/ttg-deepseek-20261031/cache/pip
unset TESTER_TPM
[ ! -f "$HOME/.config/containers/auth.json" ] || export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"
pids=()
finish() {
  trap - TERM INT EXIT
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  "$PY" src/tester.py summarize --data "$ROOT/pilot-data" --out-tag deepseek > "$ROOT/logs/summary.log" 2>&1 || true
  sacct -j "$SLURM_JOB_ID" -X -P -o JobID,Account,State,ElapsedRaw,AllocNodes,Timelimit > "$ROOT/accounting.txt" || true
  date -u +%FT%TZ > "$ROOT/job-ended.txt"
}
trap finish TERM INT EXIT
date -u +%FT%TZ > "$ROOT/job-started.txt"
for i in $(seq 1 8); do
  "$PY" src/tester.py run --set pilot --data "$ROOT/pilot-data" --check --out-tag deepseek \
    --seeds 1 --first-seed 0 --conditions C1,C2,C3,C4,C5,X --prompt-size full --shard "$i/8" \
    > "$ROOT/logs/shard-$i.log" 2>&1 &
  pids+=("$!")
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
pids=()
exit "$failed"
