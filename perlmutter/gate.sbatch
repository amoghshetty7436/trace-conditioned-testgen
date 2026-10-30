#!/bin/bash
# Gate pilot: gpt-oss-120b as the tester, served by vLLM on this node, on 8 pilot tasks x 2 patches x 6 conditions
# (96 runs). Task containers run on the same node with podman-hpc. Expected: about 1 node-hour.
#
#   sbatch perlmutter/gate.sbatch                     # from the repo root, after perlmutter/setup.sh
#   sbatch -C "gpu&hbm80g" perlmutter/gate.sbatch     # same, on an 80 GB A100 node if those are free sooner
#
# Rerunning after a timeout or crash resumes: finished runs are kept in data/tester_runs_gpt_oss_120b.jsonl.
# Overrides: TASKS=a,b,c  CONDITIONS=C1,C5  SHARDS=4  EFFORT=medium  (e.g. `TASKS=... sbatch ...`)
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -t 01:30:00
#SBATCH -J ttg-gate
#SBATCH -o ttg-gate-%j.out
set -uo pipefail
ROOT="${TTG_ROOT:-$SCRATCH/ttg}"
cd "${SLURM_SUBMIT_DIR:-.}"
module load python 2>/dev/null || true

# The 8 tasks: 5 where the Azure model caught a bug that flash-lite missed, 2 with a real bug in the
# "correct" patch (add-trailing-comma-20 also has one in its wrong patch), and 2 easy catches that
# check the setup works (dishka-299, alpenhorn-226).
TASKS="${TASKS:-asottile__add-trailing-comma-20,probabl-ai__skore-914,streamlink__streamlink-2765,pypa__setuptools_scm-803,lzakharov__csv2md-22,scrapy__parsel-308,reagento__dishka-299,radiocosmology__alpenhorn-226}"
CONDITIONS="${CONDITIONS:-C1,C2,C3,C4,C5,X}"
SHARDS="${SHARDS:-8}"
TAG="gpt_oss_120b"
LOGS="logs/$TAG-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"

# 1. Serve the model on all 4 GPUs.
export HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1
"$ROOT/vllm-env/bin/vllm" serve "$ROOT/models/gpt-oss-120b" --served-model-name openai/gpt-oss-120b \
  --tensor-parallel-size 4 --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --host 127.0.0.1 --port 8000 > "$LOGS/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
echo "waiting for vLLM (log: $LOGS/vllm.log)"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see $LOGS/vllm.log"; tail -n 40 "$LOGS/vllm.log"; exit 1; }
  sleep 15
done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM not up after 30 min"; exit 1; }

# 2. The harness talks to it like any OpenAI-compatible API.
export TESTER_BASE_URL=http://127.0.0.1:8000/v1 TESTER_API_KEY=none TESTER_MODEL=openai/gpt-oss-120b
export TESTER_REASONING_EFFORT="${EFFORT:-high}" TESTER_MAX_TOKENS=16384 CONTAINER_CLI=podman-hpc
PY="$ROOT/ttg-env/bin/python"
"$PY" src/llm.py "Reply with the word ready." || { echo "model smoke test failed"; exit 1; }

# 3. One tester process per task; each pulls and cleans its own image, then runs its 12 runs.
pids=()
for i in $(seq 1 "$SHARDS"); do
  "$PY" src/tester.py run --out-tag "$TAG" --tasks "$TASKS" --conditions "$CONDITIONS" --shard "$i/$SHARDS" \
    > "$LOGS/shard-$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a tester process exited with an error; see $LOGS/shard-*.log"; done

"$PY" src/tester.py summarize --out-tag "$TAG"
echo "done: data/tester_runs_$TAG.jsonl, data/tester_summary_$TAG.json"
