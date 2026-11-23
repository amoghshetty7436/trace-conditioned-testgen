"""Prepare isolated shards or launch them on one allocated Perlmutter CPU node."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, inputs, shards):
    meta = json.loads((inputs / "inputs.json").read_text())
    for name, expected in meta["files"].items():
        if digest(inputs / name) != expected:
            raise SystemExit(f"Input hash mismatch: {name}")
    rows = [json.loads(line) for line in (inputs / "pilot_tasks.jsonl").read_text().splitlines()]
    tasks = list(dict.fromkeys(r["instance_id"] for r in rows))
    if len(tasks) != 169 or len(rows) != 338:
        raise SystemExit("Expected 169 tasks and 338 trajectories")
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    manifest = {"inputs": meta, "shards": shards, "workers_per_shard": 4, "containers_per_shard": 2,
                "seed": 0, "conditions": ["C1", "C2", "C3", "C4", "C5", "X"],
                "prompt_size": "full", "rounds": 3, "validation_policy": "terra",
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for folder in ("src", "perlmutter")
                                  for p in sorted((ROOT / folder).glob("*")) if p.is_file()},
                "model_settings": {k: v for k, v in llm.settings().items() if k != "api_key"},
                "budget": {"cpu_nodes": 1, "qos": "regular", "walltime_seconds": 34200,
                           "total_authorized_node_hours": 10, "automatic_resubmission": False},
                "tasks_by_shard": [tasks[i::shards] for i in range(shards)]}
    path = run / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit("Run manifest changed; do not mix protocols or inputs")
    path.write_text(json.dumps(manifest, indent=2))
    for i in range(shards):
        dest = run / f"shard-{i + 1:02d}"
        dest.mkdir(exist_ok=True)
        for name in ("pilot_tasks.jsonl", "pilot_traces.jsonl.gz"):
            target = dest / name
            if not target.exists():
                target.symlink_to(inputs / name)
    print(f"Prepared {len(tasks)} tasks / 2028 planned cases in {shards} shards")


def launch(run):
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Launch requires a Slurm allocation")
    manifest = json.loads((run / "manifest.json").read_text())
    settings = {k: v for k, v in llm.settings().items() if k != "api_key"}
    if settings != manifest["model_settings"]:
        raise SystemExit("Runtime model settings differ from the frozen manifest")
    for name, sha in manifest["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise SystemExit(f"Frozen code changed: {name}")
    children, handles = [], []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.time()
    for i, tasks in enumerate(manifest["tasks_by_shard"]):
        log = (run / "logs" / f"shard-{i + 1:02d}.log").open("a")
        handles.append(log)
        cmd = [sys.executable, str(ROOT / "src/tester.py"), "run", "--data", str(run / f"shard-{i + 1:02d}"),
               "--tasks", ",".join(tasks), "--seeds", "1", "--prompt-size", "full", "--validate-images",
               "--validation-policy", "terra", "--workers", str(manifest["workers_per_shard"]),
               "--docker-workers", str(manifest["containers_per_shard"])]
        children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    while any(p.poll() is None for p in children) and not stopping:
        time.sleep(5)
    if stopping:
        deadline = time.monotonic() + 30
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(1)
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    codes = [p.wait() for p in children]
    for handle in handles:
        handle.close()
    (run / "execution.json").write_text(json.dumps({"job_id": os.environ["SLURM_JOB_ID"], "started": started,
        "ended": time.time(), "stopped": stopping, "shard_exit_codes": codes}, indent=2))
    return 1 if stopping or any(codes) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "launch"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()
    if args.step == "prepare":
        if not args.inputs or not 1 <= args.shards <= 32:
            ap.error("prepare needs --inputs and 1–32 shards")
        prepare(args.run.resolve(), args.inputs.resolve(), args.shards)
    else:
        raise SystemExit(launch(args.run.resolve()))


if __name__ == "__main__":
    main()
