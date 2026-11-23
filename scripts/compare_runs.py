"""Summarize an experiment and compare only matched runs with the saved pilot."""

import argparse
import json
from pathlib import Path

CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "X"]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def key(row):
    return row["trajectory_id"], row["condition"], row["seed"]


def rate(rows):
    return f"{sum(bool(r.get('patch_flagged')) for r in rows)}/{len(rows)}" if rows else "—"


def mean(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return f"{sum(values) / len(values):.3f}" if values else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", default="data/tester_runs.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    rows = read_rows(data / "tester_runs.jsonl")
    rows = list({key(r): r for r in rows if not r.get("grade_error")}.values())
    baseline = {key(r): r for r in read_rows(Path(args.baseline))}
    selection = json.loads((data / "selection.json").read_text())
    config = json.loads((data / "tester_config.json").read_text()) if (data / "tester_config.json").exists() else {}
    expected = len(selection["tasks"]) * 2 * len(CONDITIONS)
    counts = {iid: sum(r["instance_id"] == iid for r in rows) for iid in selection["tasks"]}
    complete = sum(n == 12 for n in counts.values())
    lines = ["# Azure newer-task experiment", "",
             f"Status: {len(rows)}/{expected} runs recorded; {complete}/{len(counts)} tasks complete.", "",
             f"Deployment: `{config.get('model', 'gpt-5.6-sol')}`; reasoning: `{config.get('reasoning_effort', 'high')}`.",
             "One seed, two agent patches per task, six conditions, full prompt budgets, up to three generation rounds.",
             "The sample includes the 10 newer pilot tasks and 20 additional newer tasks sampled with seed 0.", "",
             "## All completed Azure runs", "",
             "Counts use completed runs only. Denominators may differ while the experiment is running.", "",
             "| Condition | Runs | Wrong patches caught | False alarms | Wrong-patch runs with locked-in tests | Mean coverage | Coverage available |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        group = [r for r in rows if r["condition"] == c]
        wrong = [r for r in group if not r["resolved"]]
        right = [r for r in group if r["resolved"]]
        locked = f"{sum(r.get('locked_in', 0) > 0 for r in wrong)}/{len(wrong)}" if wrong else "—"
        lines.append(f"| {c} | {len(group)} | {rate(wrong)} | {rate(right)} | {locked} | {mean(group, 'coverage')} | {sum(r.get('coverage') is not None for r in group)}/{len(group)} |")
    lines += ["", "## Matched comparison against Flash-Lite", "",
              "Only identical trajectory, condition and seed pairs are compared below. The expanded 30-task cohort is not directly compared with the original mixed old/new cohort.", "",
              "| Condition | Matched wrong patches | Flash-Lite catches | Azure catches | Azure-only catches | Flash-Lite-only catches | Flash-Lite false alarms | Azure false alarms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        matched = [r for r in rows if r["condition"] == c and key(r) in baseline]
        wrong = [r for r in matched if not r["resolved"]]
        right = [r for r in matched if r["resolved"]]
        gains = sum(r["patch_flagged"] and not baseline[key(r)]["patch_flagged"] for r in wrong)
        losses = sum(not r["patch_flagged"] and baseline[key(r)]["patch_flagged"] for r in wrong)
        lines.append(f"| {c} | {len(wrong)} | {rate([baseline[key(r)] for r in wrong])} | {rate(wrong)} | {gains} | {losses} | {rate([baseline[key(r)] for r in right])} | {rate(right)} |")
    models = sorted({round_rec["actual_model"] for r in rows for round_rec in r.get("round_records", []) if round_rec.get("actual_model")})
    lines += ["", "## Execution details", "",
              f"Returned model versions: {', '.join(models) or 'No tester completions yet'}.",
              f"Recorded tester input tokens: {sum(r.get('prompt_tokens', 0) for r in rows):,}.",
              f"Recorded tester completion tokens (including reasoning where reported): {sum(r.get('completion_tokens', 0) for r in rows):,}.",
              "Trace-summary generation and API smoke checks are not included in these token totals.", "",
              "## Interpretation limits", "",
              "- This is a one-seed experiment. Small differences are exploratory.",
              "- The historical 'new' date window was defined for the original study; it does not establish that these tasks were absent from this tester's training data.",
              "- X selects a length-matched donor from the available trace pool. Expanding the pool can change its donor, so X is not an identical-prompt model comparison.",
              "- C3 summaries are generated by the selected tester model; its comparison changes both summarization and test generation.",
              "- Coverage excludes runs where instrumentation is unavailable.", "", "## Task completion", "",
              "| Task | Completed runs (of 12) |", "|---|---:|"]
    for iid, n in counts.items():
        lines.append(f"| {iid} | {n} |")
    blocked = read_rows(data / "task_errors.jsonl")
    invalid = read_rows(data / "tester_errors.jsonl")
    validations = read_rows(data / "image_validation.jsonl")
    lines += ["", f"Recorded task error events: {len(blocked)}; invalid grading attempts: {len(invalid)}.",
              f"Image validation attempts: {len(validations)}; passed: {sum(bool(r.get('ok')) for r in validations)}.", ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}: {len(rows)}/{expected} runs, {complete}/{len(counts)} complete tasks")


if __name__ == "__main__":
    main()
