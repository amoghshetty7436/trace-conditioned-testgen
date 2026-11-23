"""Submit once. A persistent ledger prevents accidental duplicate allocations."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--account", required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    if not (run / "manifest.json").is_file() or not args.key.is_file() or not args.python.is_file():
        raise SystemExit("Missing manifest, Python environment, or private key file")
    cmd = ["sbatch", "--parsable", "--account", args.account, "--constraint=cpu", "--qos=regular", "--nodes=1",
           "--time=09:30:00", "--no-requeue", "--export=NONE", "--chdir", str(root),
           "--output", str(run / "logs/slurm-%j.out"), str(root / "perlmutter/deepseek.sbatch"),
           str(run), str(args.python.resolve()), str(args.key.resolve())]
    # Keep the intent even on failure: an uncertain submission must be checked with squeue.
    with (run / "submission.json").open("x") as ledger:
        json.dump({"state": "submitting", "reserved_node_hours": 9.5, "command": cmd}, ledger, indent=2)
    result = subprocess.run(cmd, capture_output=True, text=True)
    (run / "submission.json").write_text(json.dumps({"state": "submitted" if result.returncode == 0 else "check_required",
        "reserved_node_hours": 9.5, "command": cmd, "stdout": result.stdout, "stderr": result.stderr}, indent=2))
    print(result.stdout.strip() or result.stderr.strip())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
