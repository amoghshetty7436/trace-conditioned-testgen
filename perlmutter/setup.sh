#!/bin/bash
# One-time setup on a Perlmutter login node: Python envs, the gpt-oss-120b weights, coverage wheels.
# Run from the repo root:  bash perlmutter/setup.sh      (about 20 to 40 minutes, mostly the 65 GB download)
set -euo pipefail
ROOT="${TTG_ROOT:-$SCRATCH/ttg}"
mkdir -p "$ROOT"
module load python

# vLLM gets its own env: it pins torch and CUDA libraries the harness doesn't need.
if [ ! -x "$ROOT/vllm-env/bin/vllm" ]; then
  python -m venv "$ROOT/vllm-env"
  "$ROOT/vllm-env/bin/pip" install -q --upgrade pip
  "$ROOT/vllm-env/bin/pip" install -q "vllm==${VLLM_VERSION:-0.11.0}"
fi
if [ ! -x "$ROOT/ttg-env/bin/python" ]; then
  python -m venv "$ROOT/ttg-env"
  "$ROOT/ttg-env/bin/pip" install -q --upgrade pip
  "$ROOT/ttg-env/bin/pip" install -q -r requirements.txt
fi

# Model weights (about 65 GB). original/ and metal/ are other copies of the same weights.
export HF_HOME="$ROOT/hf"
"$ROOT/vllm-env/bin/python" -c "from huggingface_hub import snapshot_download as d; d('openai/gpt-oss-120b', \
  local_dir='$ROOT/models/gpt-oss-120b', ignore_patterns=['original/*', 'metal/*'])"

# coverage.py wheels for the sandbox; compute nodes can download these too, this just saves time.
"$ROOT/ttg-env/bin/python" -c "import sys; sys.path.insert(0, 'src'); import sandbox; print(sandbox.coverage_wheels())"

echo "Setup done in $ROOT. Next: sbatch perlmutter/gate.sbatch"
