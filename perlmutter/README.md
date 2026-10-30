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

8 tasks x 2 patches (wrong and "correct") x 6 conditions = 96 runs, at high reasoning effort. The
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
