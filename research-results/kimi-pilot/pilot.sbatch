#!/bin/bash
# Launcher and external recorder only. Experiment source stays unchanged.
#SBATCH -A m4999
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH -J ttg-kimi-pilot
set -uo pipefail
umask 077
ROOT=/pscratch/sd/i/ics35/ttg-kimi-pilot-amogh-20261115
PY=/pscratch/sd/i/ics35/ttg-deepseek-20261031/env/bin/python
cd "$ROOT/repo" || exit 1
[ "$(git rev-parse HEAD)" = 44ec6b20afeb790a8c5c625497c89bf0f163967d ] || exit 1
git diff --exit-code HEAD -- src || exit 1
[ -f "$ROOT/pilot_api_smoke.json" ] || exit 1
export CONTAINER_CLI=podman-hpc PYTHONUNBUFFERED=1
export TESTER_MODEL=Kimi-K3.1-Pro TESTER_REASONING_EFFORT=high TESTER_MAX_TOKENS=65536
export PIP_CACHE_DIR=/pscratch/sd/i/ics35/ttg-deepseek-20261031/cache/pip
export TTG_RECORDER_TOKEN="$($PY -c 'import secrets; print(secrets.token_hex(32))')"
export TESTER_API_KEY="$TTG_RECORDER_TOKEN"
unset TESTER_TPM
pids=()
proxy_pid=
finish() {
  result=$?
  trap - TERM INT EXIT
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  [ -z "$proxy_pid" ] || kill -TERM "$proxy_pid" 2>/dev/null || true
  "$PY" src/tester.py summarize --data "$ROOT/pilot-data" --out-tag kimi > "$ROOT/logs/summary.log" 2>&1 || true
  sacct -j "$SLURM_JOB_ID" -X -P -o JobID,Account,State,ElapsedRaw,AllocNodes,Timelimit,ExitCode > "$ROOT/accounting.txt" || true
  git status --short > "$ROOT/git-status-after.txt"
  date -u +%FT%TZ > "$ROOT/job-ended.txt"
  echo "$result" > "$ROOT/launcher-exit.txt"
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u +%FT%TZ > "$ROOT/job-started.txt"
echo "$SLURM_JOB_ID" > "$ROOT/job-id.txt"
hostname > "$ROOT/node.txt"
rm -f "$ROOT/recorder-ready.json"
"$PY" "$ROOT/recording_proxy.py" --key /pscratch/sd/i/ics35/ttg-private/deepseek.key \
  --out "$ROOT/api-traces" --ready "$ROOT/recorder-ready.json" > "$ROOT/logs/recorder.log" 2>&1 &
proxy_pid=$!
for i in $(seq 1 100); do
  [ -f "$ROOT/recorder-ready.json" ] && break
  kill -0 "$proxy_pid" 2>/dev/null || exit 1
  sleep 0.1
done
port=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$ROOT/recorder-ready.json") || exit 1
for i in $(seq 1 8); do
  TESTER_BASE_URL="http://127.0.0.1:$port/shard/$i/openai/v1" \
    "$PY" src/tester.py run --set pilot --data "$ROOT/pilot-data" --check --out-tag kimi \
    --seeds 1 --first-seed 0 --conditions C1,C2,C3,C4,C5,X --prompt-size full --shard "$i/8" \
    > "$ROOT/logs/shard-$i.log" 2>&1 &
  pids+=("$!")
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
pids=()
exit "$failed"
