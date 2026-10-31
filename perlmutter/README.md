# Running the tester on NERSC Perlmutter

gpt-oss-120b runs as the tester model, served by vLLM on one A100 node (4 GPUs). The task sandboxes run
on the same node's CPUs with `podman-hpc`, since Perlmutter has no Docker. None of this has been run on
Perlmutter yet, so the first job doubles as the test of the setup.

## Steps

1. Log in (with MFA): `ssh <user>@perlmutter.nersc.gov`, then clone the repo and `cd` into it.
2. Optional but worth it: `podman-hpc login docker.io` with a free Docker Hub account. Anonymous pulls
   share NERSC's address and can hit Docker Hub's rate limit.
3. `bash perlmutter/setup.sh` on the login node. It makes two Python envs and downloads the model
   (about 65 GB) into `$SCRATCH/ttg`. Takes 20 to 40 minutes and uses no allocation.
4. `sbatch perlmutter/gate.sbatch`. It charges `m4999_g` (the project's GPU allocation) and should take
   about 1 node-hour. Watch it with `squeue --me` and `tail -f ttg-gate-<jobid>.out`; per-task logs and
   the vLLM log are in `logs/`.
5. `python perlmutter/compare.py` prints which bugs gpt-oss-120b found next to the Azure model and
   flash-lite, and a go/no-go verdict.

If the job stops early (time limit, a crash), submit it again: finished runs are kept and skipped.

## The gate pilot

8 tasks x 2 patches (wrong and "correct") x 6 conditions = 96 runs, at medium reasoning effort (high effort used up the 16k-token budget thinking in most runs). The
tasks are picked to tell a strong tester from a weak one: 7 bugs that the Azure model found and
flash-lite missed (5 in wrong patches, 2 in patches the benchmark calls correct), plus 2 easy catches
that both found, which check the setup. **Go** if gpt-oss-120b finds 3 or more of the 7 hard bugs;
read the flagged tests if it finds 1 or 2; **stop** at 0.

## If something breaks

- vLLM won't start: see `logs/gpt_oss_120b-<jobid>/vllm.log`. On some vLLM versions A100s need
  `export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` before `vllm serve`. A newer vLLM
  (`VLLM_VERSION=... bash perlmutter/setup.sh`) is the other fix.
- Image pulls or cleaning fail: see `logs/.../shard-*.log`. Task images live in node-local storage,
  which shares the node's memory, so lower `SHARDS` if the node runs out.
- The account name is refused: GPU jobs on Perlmutter charge `<project>_g`; check with `iris` or
  `sacctmgr show assoc user=$USER`.

# The full study with Terra (CPU node, no GPU)

The tester is Terra (`gpt-5.6-terra` on Azure OpenAI) at its default reasoning effort, as in the pilot.
The model runs on Azure, so the job needs one CPU node for the task sandboxes and charges `m4999`
(the CPU allocation). Phase 1 is seed 0: 169 tasks x 2 patches x 6 conditions = 2,028 runs, about
2 to 3 node-hours and roughly $200 to $240 of Azure usage. Seeds 1 and 2 can follow later.

1. On a login node, in the repo: `git pull`.
2. Save the Azure settings outside the repo, readable only by you:
   ```bash
   cat > ~/.ttg_azure.env <<'ENV'
   TESTER_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
   TESTER_API_KEY=<key>
   TESTER_MODEL=gpt-5.6-terra
   ENV
   chmod 600 ~/.ttg_azure.env
   ```
3. Log in to Docker Hub where the compute node can see it (169 images; anonymous pulls are
   rate-limited): `podman-hpc login --authfile ~/.config/containers/auth.json docker.io`
4. Fetch every selected task and trace from Hugging Face (about 3 GB download, free on the login node):
   `HF_HOME=$SCRATCH/ttg/hf $SCRATCH/ttg/ttg-env/bin/python src/pilot.py export-study`.
   It should print `wrote 338 traces from 169 tasks`.
5. `sbatch perlmutter/terra.sbatch`. Later, `SEEDS=3 sbatch perlmutter/terra.sbatch` adds seeds 1 and 2.

Each task is checked before any model call (`--check`): one commit in the cleaned image, no leaked
fix, hidden tests fail on the base code and pass with the human fix. Tasks that fail are skipped and
logged in `data/task_checks_terra.jsonl`, which also records whether each agent patch passes the hidden
tests (each run carries it as `hidden_tests_pass`). Results: `data/tester_runs_terra.jsonl` and
`data/tester_summary_terra.json`.
