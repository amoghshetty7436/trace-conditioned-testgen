"""Export the selected 169 tasks from pinned HF revisions, before allocating a node."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pilot import EXPORT_COLUMNS, plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path, default=Path("data/selected_traces.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "inputs.json"
    selection_hash = hashlib.sha256(args.selected.read_bytes()).hexdigest()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["selected_sha256"] != selection_hash:
            raise SystemExit("Selection changed; choose a new output directory")
        for name, digest in old["files"].items():
            if hashlib.sha256((args.out / name).read_bytes()).hexdigest() != digest:
                raise SystemExit(f"Input changed: {name}")
        print("Existing input export verified")
        return
    selected = pd.read_csv(args.selected)
    if selected.instance_id.nunique() != 169 or len(selected) != 338 or selected.trajectory_id.duplicated().any():
        raise SystemExit("Expected exactly 169 tasks / 338 unique trajectories")
    if not all(set(g.resolved) == {0, 1} and len(g) == 2 for _, g in selected.groupby("instance_id")):
        raise SystemExit("Each task must have one resolved and one unresolved trajectory")
    api = HfApi()
    repos = ["nebius/SWE-rebench", "nebius/SWE-rebench-openhands-trajectories"]
    pin_path = args.out / "dataset_revisions.json"
    pins = json.loads(pin_path.read_text()) if pin_path.exists() else {r: api.dataset_info(r).sha for r in repos}
    pin_path.write_text(json.dumps(pins, indent=2))
    ids = set(selected.trajectory_id)
    trace_ds = load_dataset(repos[1], revision=pins[repos[1]], split="train")
    trace_ds = trace_ds.filter(lambda batch: [tid in ids for tid in batch], input_columns="trajectory_id", batched=True)
    trace_rows = trace_ds.select_columns(["trajectory_id", "model_patch", "trajectory"]).to_pandas()
    tasks = pd.concat([load_dataset(repos[0], revision=pins[repos[0]], split=split).to_pandas()
                       for split in ("test", "filtered")], ignore_index=True).drop_duplicates("instance_id")
    task_cols = ["instance_id", "base_commit", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config", "problem_statement"]
    keep = ["instance_id", "trajectory_id", "period", "resolved", "exit_status", "docker_image", "created_at"]
    rows = selected[keep].merge(trace_rows[["trajectory_id", "model_patch"]], on="trajectory_id", validate="one_to_one")
    rows = rows.merge(tasks[task_cols], on="instance_id", validate="many_to_one")
    if len(rows) != 338 or set(rows.trajectory_id) != ids:
        raise SystemExit("Dataset export lost selected trajectories")
    # Preserve the original selection order, including tie-breaking for X donors.
    rows = rows.set_index("trajectory_id").loc[selected.trajectory_id].reset_index()
    with (args.out / "pilot_tasks.jsonl").open("w") as f:
        for row in rows[EXPORT_COLUMNS].to_dict("records"):
            f.write(json.dumps(plain(row), default=str) + "\n")
    with gzip.open(args.out / "pilot_traces.jsonl.gz", "wt") as f:
        for row in trace_rows.to_dict("records"):
            f.write(json.dumps({k: plain(row[k]) for k in ("trajectory_id", "trajectory")}, default=str) + "\n")
    (args.out / "selected_traces.csv").write_bytes(args.selected.read_bytes())
    names = ["pilot_tasks.jsonl", "pilot_traces.jsonl.gz", "selected_traces.csv"]
    manifest_path.write_text(json.dumps({"tasks": 169, "trajectories": 338, "expected_cases": 2028,
        "selected_sha256": selection_hash, "dataset_revisions": pins,
        "files": {n: hashlib.sha256((args.out / n).read_bytes()).hexdigest() for n in names}}, indent=2))
    print("Exported and hashed 169 tasks / 338 trajectories; pilot_* filenames contain the FULL study")


if __name__ == "__main__":
    main()
