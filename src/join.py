"""Build the combined study dataset.

Joins the Nebius OpenHands traces (trace + agent patch + resolved label) with the
SWE-rebench tasks (issue + golden patch + human tests + Docker image) on instance_id.

Outputs (in data/):
  combined.parquet      every trace joined to its task (~2 GB, keep out of git)
  combined_light.parquet same rows without the trajectory text (small, for quick analysis)
  join_report.json      sanity counts used to size the study

Usage:
  python src/join.py                 # full run
  python src/join.py --limit 2000    # quick smoke test on the first 2000 traces
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

TASKS_REPO = "nebius/SWE-rebench"
TRACES_REPO = "nebius/SWE-rebench-openhands-trajectories"

# Fresh-task window: after the main tester's knowledge cutoff (gpt-oss-20b, June 2024)
# plus a buffer, up to the end of the SWE-rebench release.
NEW_START = "2024-09-01"
NEW_END = "2025-04-30"
OLD_END = "2024-06-01"

TASK_COLUMNS = [
    "instance_id", "repo", "base_commit", "created_at", "problem_statement", "hints_text",
    "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "version", "install_config",
    "meta", "docker_image", "image_name", "license_name",
]


def load_tasks() -> pd.DataFrame:
    parts = []
    for split in ["test", "filtered"]:
        df = load_dataset(TASKS_REPO, split=split).to_pandas()
        df["split"] = split
        parts.append(df)
    tasks = pd.concat(parts, ignore_index=True)
    in_filtered = tasks.groupby("instance_id")["split"].agg(lambda s: "filtered" in set(s))
    tasks = tasks.drop_duplicates("instance_id").set_index("instance_id")
    tasks["in_filtered"] = in_filtered
    tasks = tasks.reset_index()
    keep = [c for c in TASK_COLUMNS if c in tasks.columns] + ["in_filtered"]
    return tasks[keep]


def load_traces(limit: int | None) -> pd.DataFrame:
    split = f"train[:{limit}]" if limit else "train"
    return load_dataset(TRACES_REPO, split=split).to_pandas()


def period(created_at: pd.Series) -> pd.Series:
    ts = pd.to_datetime(created_at, utc=True, errors="coerce")
    out = pd.Series("between", index=created_at.index)
    out[ts < pd.Timestamp(OLD_END, tz="UTC")] = "old"
    out[(ts >= pd.Timestamp(NEW_START, tz="UTC")) & (ts <= pd.Timestamp(NEW_END, tz="UTC") + pd.Timedelta(days=1))] = "new"
    return out


def report(traces: pd.DataFrame, tasks: pd.DataFrame, combined: pd.DataFrame) -> dict:
    per_task = combined.groupby("instance_id").agg(
        n_traces=("resolved", "size"),
        n_resolved=("resolved", "sum"),
        period=("period", "first"),
        in_filtered=("in_filtered", "first"),
    )
    per_task["has_both"] = (per_task.n_resolved > 0) & (per_task.n_resolved < per_task.n_traces)

    def by_period(mask):
        return {p: int(((per_task.period == p) & mask).sum()) for p in ["old", "between", "new"]}

    return {
        "traces_rows": len(traces),
        "tasks_rows_deduplicated": len(tasks),
        "joined_rows": len(combined),
        "traces_without_task": int((~traces.instance_id.isin(tasks.instance_id)).sum()),
        "distinct_tasks_with_trace": int(per_task.shape[0]),
        "resolved_share_of_traces": round(float(combined.resolved.mean()), 4),
        "tasks_by_period": by_period(pd.Series(True, index=per_task.index)),
        "tasks_with_resolved_and_unresolved_trace_by_period": by_period(per_task.has_both),
        "tasks_with_only_unresolved_traces_by_period": by_period(per_task.n_resolved == 0),
        "tasks_in_filtered_split": int(per_task.in_filtered.sum()),
        "traces_mentioning_git_history_cmds": int(combined["uses_git_history"].sum()),
        "trajectory_chars_quantiles": {
            q: int(combined["trajectory_chars"].quantile(q)) for q in [0.1, 0.5, 0.9, 0.99]
        },
        "gen_tests_correct_values": combined["gen_tests_correct"].describe().round(4).to_dict()
        if "gen_tests_correct" in combined else None,
        "pred_passes_gen_tests_values": combined["pred_passes_gen_tests"].describe().round(4).to_dict()
        if "pred_passes_gen_tests" in combined else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only load the first N traces")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks()
    traces = load_traces(args.limit)

    combined = traces.merge(tasks, on="instance_id", how="inner", suffixes=("_trace", "_task"))
    combined["period"] = period(combined["created_at"])
    traj_text = combined["trajectory"].astype(str)
    combined["trajectory_chars"] = traj_text.str.len()
    combined["uses_git_history"] = traj_text.str.contains(
        r"git (?:log|show|reflog|checkout [^\s]*tag)", regex=True
    )

    combined.to_parquet(out / "combined.parquet", index=False)
    combined.drop(columns=["trajectory", "tools"], errors="ignore").to_parquet(
        out / "combined_light.parquet", index=False
    )

    rep = report(traces, tasks, combined)
    (out / "join_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
