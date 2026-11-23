"""Export a reproducible newer-task sample using the already selected trace pairs.

Includes the newer pilot tasks for a matched comparison, then samples additional
newer tasks with a fixed seed. Reads only needed Parquet columns/row groups over
HTTP; does not repeat selection or overwrite the original pilot.
"""

import argparse
import csv
import gzip
import io
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from llm import SSL_CONTEXT
from pilot import EXPORT_COLUMNS


def fetch(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                  context=SSL_CONTEXT, timeout=180)


class RemoteParquet(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        if n <= 0:
            return b""
        with fetch(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}) as r:
            if r.status != 206:
                raise RuntimeError(f"Expected HTTP range response, got {r.status}")
            out = r.read()
        if len(out) != n:
            raise RuntimeError(f"Short range read: expected {n}, got {len(out)}")
        self.pos += len(out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = list(csv.DictReader(Path("data/selected_traces.csv").open()))
    newer = {r["instance_id"] for r in selected if r["period"] == "new"}
    pilot = sorted({r["instance_id"] for r in selected
                    if r["period"] == "new" and r["pilot"].lower() == "true"})
    if not len(pilot) <= args.tasks <= len(newer):
        raise SystemExit(f"--tasks must be between {len(pilot)} and {len(newer)}")
    ids = pilot + random.Random(args.seed).sample(sorted(newer - set(pilot)), args.tasks - len(pilot))
    pairs = [r for iid in ids for r in selected if r["instance_id"] == iid]
    metadata = {"tasks": ids, "selection_seed": args.seed, "matched_pilot_tasks": pilot,
                "period": "new", "trajectory_ids": [r["trajectory_id"] for r in pairs]}
    tasks = {r["trajectory_id"]: r for r in map(json.loads, Path("data/pilot_tasks.jsonl").open())
             if r["instance_id"] in ids}
    with gzip.open("data/pilot_traces.jsonl.gz", "rt") as f:
        trajectories = {r["trajectory_id"]: r for r in map(json.loads, f) if r["trajectory_id"] in tasks}

    # Cache the extra records so interrupted exports can resume without redownloading.
    cache = out / "source_records.jsonl"
    sources = {r["trajectory_id"]: r for r in map(json.loads, cache.open())} if cache.exists() else {}
    repos = ["nebius/SWE-rebench", "nebius/SWE-rebench-openhands-trajectories"]
    revisions = {}
    for repo in repos:
        with fetch(f"https://huggingface.co/api/datasets/{repo}") as r:
            revisions[repo] = json.load(r)["sha"]
    metadata["dataset_revisions"] = revisions
    (out / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    task_file = out / "source_tasks.parquet"
    if not task_file.exists():
        repo = repos[0]
        print("Downloading filtered task metadata", flush=True)
        with fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revisions[repo]}/data/filtered-00000-of-00001.parquet") as r:
            task_file.write_bytes(r.read())
    task_rows = {r["instance_id"]: r for r in pq.read_table(task_file).to_pylist() if r["instance_id"] in ids}
    needed = {r["trajectory_id"] for r in pairs} - set(tasks) - set(sources)
    if needed:
        repo = repos[1]
        with fetch(f"https://huggingface.co/api/datasets/{repo}/tree/{revisions[repo]}") as r:
            size = next(x["size"] for x in json.load(r) if x["path"] == "trajectories.parquet")
        remote = RemoteParquet(f"https://huggingface.co/datasets/{repo}/resolve/{revisions[repo]}/trajectories.parquet", size)
        parquet = pq.ParquetFile(remote)
        print(f"Scanning {parquet.num_row_groups} trace row groups for {len(needed)} selected traces", flush=True)
        for i in range(parquet.num_row_groups):
            tids = parquet.read_row_group(i, columns=["trajectory_id"]).column(0).to_pylist()
            hit = needed.intersection(tids)
            if not hit:
                continue
            print(f"Fetching group {i + 1}: {len(hit)} selected traces", flush=True)
            rows = parquet.read_row_group(i, columns=["trajectory_id", "instance_id", "model_patch", "resolved", "exit_status", "trajectory"]).to_pylist()
            with cache.open("a") as f:
                for row in rows:
                    if row["trajectory_id"] in hit:
                        sources[row["trajectory_id"]] = row
                        f.write(json.dumps(row) + "\n")
            needed -= hit
            if not needed:
                break
    if needed:
        raise RuntimeError(f"Missing {len(needed)} selected traces")
    for pair in pairs:
        tid, iid = pair["trajectory_id"], pair["instance_id"]
        if tid in tasks:
            continue
        src = sources[tid]
        assert src["instance_id"] == iid and int(src["resolved"]) == int(pair["resolved"])
        merged = {**task_rows[iid], **{k: src[k] for k in ("trajectory_id", "model_patch", "resolved", "exit_status")}, "period": "new"}
        tasks[tid] = {k: merged[k] for k in EXPORT_COLUMNS}
        trajectories[tid] = {"trajectory_id": tid, "trajectory": src["trajectory"]}
    with (out / "pilot_tasks.jsonl").open("w") as f, gzip.open(out / "pilot_traces.jsonl.gz", "wt") as g:
        for p in pairs:
            f.write(json.dumps(tasks[p["trajectory_id"]]) + "\n")
            g.write(json.dumps(trajectories[p["trajectory_id"]]) + "\n")
    print(f"Exported {len(ids)} tasks and {len(pairs)} traces to {out}", flush=True)


if __name__ == "__main__":
    main()
