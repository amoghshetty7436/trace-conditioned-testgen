"""Step A: pick the study tasks and pull their full traces.

Reads data/combined_light.parquet (from join.py). For each task we want one resolved
(correct) and one unresolved (wrong) agent trace, so right and wrong patches are compared
on the same task. New-set tasks are all kept; old-set tasks are sampled.

Outputs (in data/):
  selected_traces.csv      one row per chosen trace (small, committed to git)
  selected_traces.parquet  same rows plus the full trajectory text (git-ignored)
  selection_report.json    how many tasks survive each filter
  leak_examples.csv        every trace the leak check flagged, with a snippet (committed)

Leak check: before pairing, every candidate trace is scanned for signs the agent saw the
future fix. The task images are built at the base commit without the hidden tests, so the
fix can only leak through
  - future_git_history: a git command that looks beyond HEAD (--all, reflog, origin/...,
    fetch) and shows a commit dated after the task was created. The images do carry later
    commits (checked by hand: the fmralign image's `git log --all` runs months past the
    base commit), so such an agent may have copied the fix.
  - fetched_upstream: the agent downloaded from the task's own GitHub repo (curl, wget,
    git clone, pip install git+...).
Both drop just that trace. The task itself stays usable because the tester's sandbox strips
the git history down to the base commit.
Every flagged trace is written to leak_examples.csv with a snippet, for checking by hand.
(An earlier version also flagged hidden test names and golden-patch lines anywhere in the
trace; by hand those were almost all tests and code already in the repo, so they're gone.)

Usage:
  python src/select_tasks.py
  python src/select_tasks.py --n-old 75 --n-pilot 20 --seed 0
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from join import TRACES_REPO

MAX_PATCH_FILES = 3          # skip giant tasks at first (agent and golden patch)
MAX_TRAJECTORY_CHARS = 400_000  # ~100k tokens, leaves room in a 131k context
FUTURE_SLACK = pd.Timedelta(days=1)  # tolerance for time zones around created_at
# Plain `git log` only walks back from HEAD (the base commit), which can itself be dated a few
# days after created_at. Future commits are only reachable through other refs, so only commands
# that look beyond HEAD count.
GIT_HISTORY_CMD = re.compile(
    r"git\s+(?:(?:log|show|rev-list|shortlog|branch|tag)\b[^\n]*?(?:--all|--branches|--remotes|--tags|-a\b|"
    r"origin/|upstream/|FETCH_HEAD)|reflog\b|fetch\b|pull\b|(?:checkout|switch)\s+(?:origin/|upstream/|-b\s+\S+\s+origin/))")
GIT_DATE_LINE = re.compile(r"^\s*(?:Date|AuthorDate|CommitDate):\s+(.+?)\s*$", re.M)
DOWNLOAD_CMD = re.compile(r"curl|wget|git\s+clone|pip\s+install\s+\S*git\+|urllib|requests\.get")

CSV_COLUMNS = [
    "instance_id", "trajectory_id", "period", "resolved", "exit_status", "pilot", "in_filtered",
    "repo", "repo_task", "created_at", "docker_image", "trajectory_chars", "uses_git_history",
    "agent_patch_files", "agent_edited_files", "golden_patch_files", "n_fail_to_pass",
]


def files_in_patch(patch) -> int:
    return str(patch or "").count("diff --git ")


def edited_files_in_patch(patch) -> int:
    """Existing files the patch changes. New files (often the agent's reproduce scripts)
    don't make a task bigger, so they aren't counted."""
    blocks = str(patch or "").split("diff --git ")[1:]
    return sum("new file mode" not in b.split("@@", 1)[0] for b in blocks)


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


def eligible_traces(df: pd.DataFrame, counts: dict, keep_max_iterations: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["agent_patch_files"] = df["model_patch"].map(files_in_patch)
    df["agent_edited_files"] = df["model_patch"].map(edited_files_in_patch)
    df["golden_patch_files"] = df["patch"].map(files_in_patch)
    df["n_fail_to_pass"] = df["FAIL_TO_PASS"].map(list_len)

    steps = [
        ("task is old or new (not in the gap)", df.period.isin(["old", "new"])),
        ("task has a Docker image", df.docker_image.notna() & (df.docker_image.astype(str) != "")),
        ("task has FAIL_TO_PASS tests", df.n_fail_to_pass > 0),
        (f"golden patch touches <= {MAX_PATCH_FILES} files", df.golden_patch_files.between(1, MAX_PATCH_FILES)),
        ("agent patch is non-empty", df.agent_patch_files > 0),
        (f"agent patch edits <= {MAX_PATCH_FILES} existing files", df.agent_edited_files <= MAX_PATCH_FILES),
        (f"trace <= {MAX_TRAJECTORY_CHARS:,} chars", df.trajectory_chars <= MAX_TRAJECTORY_CHARS),
    ]
    if "exit_status" in df:
        status = df.exit_status.astype(str)
        if keep_max_iterations:
            # Running out of steps still leaves a (usually wrong) patch; only crashes are dropped.
            ok = status.str.startswith("submit") | status.str.contains("maximum iteration")
            steps.append(("agent submitted or hit its step limit (no crash)", ok))
        else:
            steps.append(("agent finished normally (exit_status submit)", status.str.startswith("submit")))

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


def message_text(msg) -> str:
    if isinstance(msg, dict):
        parts = [msg.get("content"), msg.get("tool_calls"), msg.get("function_call")]
        return " ".join(str(p) for p in parts if p)
    return str(msg)


def git_history_outputs(trajectory) -> str:
    """Text of tool outputs that follow an assistant step running a git history command."""
    out, after_git = [], False
    for msg in trajectory if isinstance(trajectory, (list, tuple)) else []:
        role = msg.get("role") if isinstance(msg, dict) else None
        text = message_text(msg)
        if role == "assistant":
            after_git = bool(GIT_HISTORY_CMD.search(text))
        elif role == "tool" and after_git:
            out.append(text)
    return "\n".join(out)


def snippet(text: str, start: int, end: int, width: int = 150) -> str:
    return text[max(0, start - width):end + width].replace("\n", " ")


def fetch_commands(trajectory) -> str:
    """Text of assistant steps that download something."""
    steps = [message_text(m) for m in trajectory if isinstance(m, dict) and m.get("role") == "assistant"] \
        if isinstance(trajectory, (list, tuple)) else []
    return "\n".join(t for t in steps if DOWNLOAD_CMD.search(t))


def leak_reasons(trajectory, created_at: pd.Timestamp, repo: str) -> list[tuple[str, str]]:
    """(reason, snippet of the trace around the match) for each kind of leak found."""
    found = []
    git_out = git_history_outputs(trajectory)
    for m in GIT_DATE_LINE.finditer(git_out):
        when = pd.to_datetime(m.group(1), utc=True, errors="coerce")
        if pd.notna(when) and pd.notna(created_at) and when > created_at + FUTURE_SLACK:
            found.append(("future_git_history", snippet(git_out, m.start(), m.end())))
            break
    fetched = fetch_commands(trajectory)
    m = re.search(rf"github\.com[/:]{re.escape(repo)}\b", fetched, re.I) if repo else None
    if m:
        found.append(("fetched_upstream", snippet(fetched, m.start(), m.end())))
    return found


def scan_for_leaks(traces, df: pd.DataFrame) -> pd.DataFrame:
    """Scan the candidate traces batch by batch. Returns trajectory_id, leak_reasons, leak_snippet."""
    repo_col = "repo_task" if "repo_task" in df else "repo"
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    keys = dict(zip(df["trajectory_id"], zip(created, df[repo_col].astype(str))))
    subset = traces.filter(lambda b: [t in keys for t in b], input_columns="trajectory_id", batched=True)

    def scan(batch):
        found = [leak_reasons(traj, *keys[tid]) for tid, traj in zip(batch["trajectory_id"], batch["trajectory"])]
        return {
            "leak_reasons": [",".join(r for r, _ in f) for f in found],
            "leak_snippet": [" | ".join(s for _, s in f) for f in found],
        }

    scanned = subset.map(scan, batched=True, batch_size=64, remove_columns=[
        c for c in subset.column_names if c != "trajectory_id"])
    return scanned.to_pandas()


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
    ap.add_argument("--exclude-max-iterations", action="store_true",
                    help="also drop runs where the agent hit its step limit")
    ap.add_argument("--no-leak-scan", action="store_true", help="skip the leak check (not for the real study)")
    args = ap.parse_args()
    data = Path(args.data)

    df = pd.read_parquet(data / "combined_light.parquet")
    df["resolved"] = df["resolved"].astype(int)
    counts = {"all traces": {"traces": int(len(df)), "paired_tasks_by_period": paired_by_period(df)}}
    eligible = eligible_traces(df, counts, keep_max_iterations=not args.exclude_max_iterations)
    traces = None if args.no_leak_scan and args.skip_trajectories else load_dataset(TRACES_REPO, split="train")

    leaks = {"future_git_history": 0, "fetched_upstream": 0}
    if not args.no_leak_scan:
        # Only tasks that still have a resolved and an unresolved trace can be picked.
        has_both = eligible.groupby("instance_id")["resolved"].transform("nunique") == 2
        eligible = eligible[has_both].merge(scan_for_leaks(traces, eligible[has_both]), on="trajectory_id", how="left")
        eligible["leak_reasons"] = eligible["leak_reasons"].fillna("")
        for reason in leaks:
            leaks[reason] = int(eligible.leak_reasons.str.contains(reason).sum())
        leaky = eligible[eligible.leak_reasons != ""]
        leaky[["instance_id", "trajectory_id", "resolved", "period", "leak_reasons", "leak_snippet"]].to_csv(
            data / "leak_examples.csv", index=False)
        # The tester's sandbox gets its git history stripped, so a leak only spoils this one run.
        eligible = eligible[eligible.leak_reasons == ""]
        counts["no leak found in the trace"] = {
            "traces": int(len(eligible)), "paired_tasks_by_period": paired_by_period(eligible)}
    pairs = pick_pairs(eligible, args.seed)

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
        traces = traces.filter(lambda b: [t in ids for t in b], input_columns="trajectory_id", batched=True)
        full = traces.to_pandas()[["trajectory_id", "trajectory", "tools"]]
        chosen.merge(full, on="trajectory_id", how="left").to_parquet(data / "selected_traces.parquet", index=False)

    report = {
        "filters": counts,
        "exit_status_values": df["exit_status"].astype(str).str[:40].value_counts().head(10).to_dict()
        if "exit_status" in df else None,
        "leaky_candidate_traces_dropped": leaks if not args.no_leak_scan else "leak scan skipped",
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
