#!/bin/bash
# Submit through submit_deepseek.py: a single, non-requeued CPU allocation.
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 09:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH -J ttg-deepseek
set -euo pipefail
umask 077
RUN="$1"
PY="$2"
KEY="$3"
ROOT="$PWD"
source perlmutter/deepseek.env.example
export TESTER_API_KEY="$(cat "$KEY")"
export HF_HOME="$(dirname "$RUN")/cache/huggingface"
export PIP_CACHE_DIR="$(dirname "$RUN")/cache/pip"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ ! -f "$HOME/.config/containers/auth.json" ] || export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"
child=''
finish() {
  trap - TERM INT EXIT
  if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi
  "$PY" perlmutter/report_deepseek.py --run "$RUN" --baseline "$ROOT/reference/tester_runs_terra.jsonl" || true
  sacct -j "$SLURM_JOB_ID" -X -P -o JobID,Account,State,ElapsedRaw,AllocNodes,Timelimit > "$RUN/accounting.txt" || true
}
trap finish TERM INT EXIT
# Leave 20 minutes of the allocation for cleanup and writing partial results.
timeout --signal=TERM --kill-after=60s 33000 "$PY" perlmutter/run_deepseek.py launch --run "$RUN" &
child=$!
wait "$child"
child=''
