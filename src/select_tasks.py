"""Step A: pick the study tasks and pull their full traces.

Reads data/combined_light.parquet (from join.py). For each task we want one resolved
(correct) and one unresolved (wrong) agent trace, so right and wrong patches are compared
on the same task. New-set tasks are all kept; old-set tasks are sampled.

Outputs (in data/):
  selected_traces.csv      one row per chosen trace (small, committed to git)
  selected_traces.parquet  same rows plus the full trajectory text (git-ignored)
  selection_report.json    how many tasks survive each filter

Usage:
  python src/select_tasks.py
  python src/select_tasks.py --n-old 75 --n-pilot 20 --seed 0
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from join import TRACES_REPO

MAX_PATCH_FILES = 3          # skip giant tasks at first (agent and golden patch)
MAX_TRAJECTORY_CHARS = 400_000  # ~100k tokens, leaves room in a 131k context

CSV_COLUMNS = [
    "instance_id", "trajectory_id", "period", "resolved", "pilot", "in_filtered",
    "repo", "repo_task", "created_at", "docker_image", "trajectory_chars", "uses_git_history",
    "agent_patch_files", "golden_patch_files", "n_fail_to_pass",
]


def files_in_patch(patch) -> int:
    return str(patch or "").count("diff --git ")


def list_len(value) -> int:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return int(bool(value.strip()))
    try:
        return len(value)
    except TypeError:
        return 0


def eligible_traces(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
    df = df.copy()
    df["agent_patch_files"] = df["model_patch"].map(files_in_patch)
    df["golden_patch_files"] = df["patch"].map(files_in_patch)
    df["n_fail_to_pass"] = df["FAIL_TO_PASS"].map(list_len)

    steps = [
        ("task is old or new (not in the gap)", df.period.isin(["old", "new"])),
        ("task has a Docker image", df.docker_image.notna() & (df.docker_image.astype(str) != "")),
        ("task has FAIL_TO_PASS tests", df.n_fail_to_pass > 0),
        (f"golden patch touches <= {MAX_PATCH_FILES} files", df.golden_patch_files.between(1, MAX_PATCH_FILES)),
        ("agent patch is non-empty", df.agent_patch_files > 0),
        (f"agent patch touches <= {MAX_PATCH_FILES} files", df.agent_patch_files <= MAX_PATCH_FILES),
        (f"trace <= {MAX_TRAJECTORY_CHARS:,} chars", df.trajectory_chars <= MAX_TRAJECTORY_CHARS),
    ]
    if "exit_status" in df:
        steps.append(("agent finished normally (exit_status submit)", df.exit_status.astype(str).str.startswith("submit")))

    keep = pd.Series(True, index=df.index)
    for name, mask in steps:
        keep &= mask
        sub = df[keep]
        counts[name] = {
            "traces": int(len(sub)),
            "paired_tasks_by_period": paired_by_period(sub),
        }
    return df[keep]


def paired_by_period(df: pd.DataFrame) -> dict:
    g = df.groupby("instance_id").agg(period=("period", "first"), r=("resolved", "max"), u=("resolved", "min"))
    paired = g[(g.r == 1) & (g.u == 0)]
    return {p: int((paired.period == p).sum()) for p in ["old", "new"]}


def pick_pairs(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One resolved and one unresolved trace per task. Prefer traces that don't read git
    history (possible leak of the fix), then pick at random."""
    df = df.sample(frac=1, random_state=seed)
    df = df.sort_values("uses_git_history", kind="stable")
    picked = df.groupby(["instance_id", "resolved"], sort=False).head(1)
    both = picked.groupby("instance_id")["resolved"].transform("nunique") == 2
    return picked[both]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-old", type=int, default=75, help="old-set tasks to sample")
    ap.add_argument("--n-new", type=int, default=None, help="cap on new-set tasks (default: all)")
    ap.add_argument("--n-pilot", type=int, default=20, help="pilot tasks, split evenly new/old")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-trajectories", action="store_true", help="don't write selected_traces.parquet")
    args = ap.parse_args()
    data = Path(args.data)

    df = pd.read_parquet(data / "combined_light.parquet")
    df["resolved"] = df["resolved"].astype(int)
    counts = {"all traces": {"traces": int(len(df)), "paired_tasks_by_period": paired_by_period(df)}}
    pairs = pick_pairs(eligible_traces(df, counts), args.seed)

    # Choose tasks: prefer the cleaner "filtered" split, then random.
    tasks = pairs.drop_duplicates("instance_id")[["instance_id", "period", "in_filtered"]]
    tasks = tasks.sample(frac=1, random_state=args.seed).sort_values("in_filtered", ascending=False, kind="stable")
    new = tasks[tasks.period == "new"].head(args.n_new) if args.n_new else tasks[tasks.period == "new"]
    old = tasks[tasks.period == "old"].head(args.n_old)
    half = args.n_pilot // 2
    pilot = set(new.instance_id.head(args.n_pilot - half)) | set(old.instance_id.head(half))

    chosen = pairs[pairs.instance_id.isin(set(new.instance_id) | set(old.instance_id))].copy()
    chosen["pilot"] = chosen.instance_id.isin(pilot)
    chosen = chosen.sort_values(["pilot", "period", "instance_id", "resolved"], ascending=[False, True, True, True])
    cols = [c for c in CSV_COLUMNS if c in chosen.columns]
    chosen[cols].to_csv(data / "selected_traces.csv", index=False)

    if not args.skip_trajectories:
        ids = set(chosen.trajectory_id)
        traces = load_dataset(TRACES_REPO, split="train")
        traces = traces.filter(lambda b: [t in ids for t in b], input_columns="trajectory_id", batched=True)
        full = traces.to_pandas()[["trajectory_id", "trajectory", "tools"]]
        chosen.merge(full, on="trajectory_id", how="left").to_parquet(data / "selected_traces.parquet", index=False)

    report = {
        "filters": counts,
        "selected_tasks": {"new": int(len(new)), "old": int(len(old))},
        "selected_traces": int(len(chosen)),
        "pilot_tasks": len(pilot),
        "selected_in_filtered_split": int(chosen.drop_duplicates("instance_id").in_filtered.sum()),
        "selected_with_git_history_trace": int(chosen.uses_git_history.sum()),
        "settings": {"n_old": args.n_old, "n_new": args.n_new, "n_pilot": args.n_pilot, "seed": args.seed,
                     "max_patch_files": MAX_PATCH_FILES, "max_trajectory_chars": MAX_TRAJECTORY_CHARS},
    }
    (data / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
