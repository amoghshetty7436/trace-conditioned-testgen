"""Step A: pick the study tasks and pull their full traces.

Reads data/combined_light.parquet (from join.py). For each task we want one resolved
(correct) and one unresolved (wrong) agent trace, so right and wrong patches are compared
on the same task. New-set tasks are all kept; old-set tasks are sampled.

Outputs (in data/):
  selected_traces.csv      one row per chosen trace (small, committed to git)
  selected_traces.parquet  same rows plus the full trajectory text (git-ignored)
  selection_report.json    how many tasks survive each filter
  leak_examples.csv        every trace the leak check flagged, with a snippet (committed)

Leak check: before pairing, every candidate trace is scanned and dropped if it
  - names a test that the hidden human test patch adds (the agent must not have seen it), or
  - shows a line of the golden fix in the output of a git history command
    (git log/show/reflog/checkout/...), i.e. the agent peeked at the future fix.
  A git leak drops the whole task (its environment likely exposes the fix); a hidden test
  name drops only that trace. Every flagged trace is written to leak_examples.csv with a
  snippet around the match, so the flags can be checked by hand.

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
MIN_LEAK_LINE_CHARS = 25     # golden-patch lines shorter than this are too generic to count
MIN_TEST_NAME_CHARS = 10     # skip generic names like "test_init"
GIT_HISTORY_CMD = re.compile(r"git\s+(?:log|show|reflog|checkout|switch|cherry-pick|rev-list|blame|branch\s+-a|fetch|pull)\b")
NEW_TEST_DEF = re.compile(r"^\+\s*(?:async\s+)?def\s+(test\w+)", re.M)
OLD_TEST_DEF = re.compile(r"^-\s*(?:async\s+)?def\s+(test\w+)", re.M)

CSV_COLUMNS = [
    "instance_id", "trajectory_id", "period", "resolved", "pilot", "in_filtered",
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


def eligible_traces(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
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


def new_test_names(test_patch) -> list[str]:
    """Test functions the hidden human test patch adds (not ones it only edits)."""
    text = str(test_patch or "")
    names = set(NEW_TEST_DEF.findall(text)) - set(OLD_TEST_DEF.findall(text))
    return sorted(n for n in names if len(n) >= MIN_TEST_NAME_CHARS)


def golden_lines(patch) -> list[str]:
    """Distinctive lines the golden fix adds."""
    out = set()
    for line in str(patch or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            text = line[1:].strip()
            if len(text) >= MIN_LEAK_LINE_CHARS:
                out.add(text)
    return sorted(out)


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


def leak_reasons(trajectory, test_names: list[str], fix_lines: list[str]) -> list[tuple[str, str]]:
    """(reason, snippet of the trace around the match) for each kind of leak found."""
    found = []
    full = str(trajectory)
    for n in test_names:
        m = re.search(rf"\b{re.escape(n)}\b", full)
        if m:
            found.append(("hidden_test_name", snippet(full, m.start(), m.end())))
            break
    git_out = git_history_outputs(trajectory)
    for line in fix_lines if git_out else []:
        i = git_out.find(line)
        if i >= 0:
            found.append(("golden_fix_in_git_output", snippet(git_out, i, i + len(line))))
            break
    return found


def scan_for_leaks(traces, df: pd.DataFrame) -> pd.DataFrame:
    """Scan the candidate traces batch by batch. Returns trajectory_id, leak_reasons, leak_snippet."""
    keys = {
        r.trajectory_id: (new_test_names(r.test_patch), golden_lines(r.patch))
        for r in df[["trajectory_id", "test_patch", "patch"]].itertuples(index=False)
    }
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
    ap.add_argument("--no-leak-scan", action="store_true", help="skip the leak check (not for the real study)")
    args = ap.parse_args()
    data = Path(args.data)

    df = pd.read_parquet(data / "combined_light.parquet")
    df["resolved"] = df["resolved"].astype(int)
    counts = {"all traces": {"traces": int(len(df)), "paired_tasks_by_period": paired_by_period(df)}}
    eligible = eligible_traces(df, counts)
    traces = None if args.no_leak_scan and args.skip_trajectories else load_dataset(TRACES_REPO, split="train")

    leaks = {"hidden_test_name": 0, "golden_fix_in_git_output": 0}
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
        # The fix in git output means the task's environment exposes the answer (future commits
        # in the image), which the tester could find too, so drop the whole task. A hidden test
        # name can also be the agent picking the same name, so drop just that trace.
        env_leak_tasks = set(leaky.loc[leaky.leak_reasons.str.contains("golden_fix_in_git_output"), "instance_id"])
        leaks["tasks_dropped_for_git_leak"] = len(env_leak_tasks)
        eligible = eligible[(eligible.leak_reasons == "") & ~eligible.instance_id.isin(env_leak_tasks)]
        counts["no leak found (trace dropped on hidden test name, task dropped on git leak)"] = {
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
