"""Preserve raw rows and compare identical cases using explicit failure/pass statuses."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"Malformed record: {path}:{number}")
    return rows


def key(row):
    return row["trajectory_id"], row["condition"], row["seed"]


def outcome(row):
    if row.get("grade_error") or row.get("error"):
        return "execution_error", None
    if not row.get("test_code"):
        return "no_test_generated", False
    statuses = row.get("statuses") or {}
    if any(not statuses.get(v) for v in ("P0", "Pa", "Pg")):
        return "no_test_results", None
    pa, pg = statuses["Pa"], statuses["Pg"]
    gold_ok = {t for t, s in pg.items() if s in ("PASSED", "XFAIL")}
    if any(pa.get(t) not in ("PASSED", "XFAIL", "FAILED", "ERROR", "XPASS") for t in gold_ok):
        return "missing_or_skipped_agent_test", None
    caught = any(pa.get(t) in ("FAILED", "ERROR", "XPASS") for t in gold_ok)
    return "scored", caught


def stats(rows):
    out = {}
    for condition in ("C1", "C2", "C3", "C4", "C5", "X"):
        group = [r for r in rows if r["condition"] == condition]
        entry = {"recorded": len(group), "status_counts": dict(Counter(outcome(r)[0] for r in group))}
        for resolved, name in ((0, "bug_caught"), (1, "false_alarm")):
            scores = [outcome(r)[1] for r in group if int(r["resolved"]) == resolved and outcome(r)[1] is not None]
            entry[name] = {"numerator": sum(scores), "denominator": len(scores),
                           "rate": sum(scores) / len(scores) if scores else None}
        out[condition] = entry
    return out


def report(run, baseline_path):
    records = [r for shard in sorted(run.glob("shard-*")) for r in read_rows(shard / "tester_runs.jsonl")]
    errors = [r for shard in sorted(run.glob("shard-*")) for r in read_rows(shard / "tester_errors.jsonl")]
    latest = {key(r): r for r in records}
    if len(latest) != len(records):
        raise ValueError("Duplicate completed cases across shards")
    baseline = {key(r): r for r in read_rows(baseline_path) if r["seed"] == 0}
    paired = sorted(k for k in latest.keys() & baseline.keys()
                    if outcome(latest[k])[1] is not None and outcome(baseline[k])[1] is not None)
    for k in paired:
        if (latest[k]["instance_id"], latest[k]["resolved"]) != (baseline[k]["instance_id"], baseline[k]["resolved"]):
            raise ValueError(f"Case labels differ: {k}")
    (run / "tester_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in latest.values()))
    manifest = json.loads((run / "manifest.json").read_text())
    inputs = read_rows(run / "shard-01/pilot_tasks.jsonl")
    expected = {(r["trajectory_id"], c, 0) for r in inputs for c in manifest["conditions"]}
    checks = [r for shard in sorted(run.glob("shard-*")) for r in read_rows(shard / "image_validation.jsonl")]
    validated = {r["instance_id"]: r for r in checks}
    usage = Counter()
    # Count persisted API replies once, including C3 summarization and unfinished cases.
    for shard in sorted(run.glob("shard-*")):
        replies = [json.loads(p.read_text())["reply"] for p in (shard / "checkpoints").glob("*/round-*.json")]
        replies += [r["reply"] for r in read_rows(shard / "pilot_summaries.jsonl") if "reply" in r]
        for reply in replies:
            for field in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens"):
                usage[field] += reply.get(field) or 0
            usage["persisted_api_replies"] += 1
            usage["truncated_replies"] += reply.get("finish_reason") == "length"
    summary = {"planned_cases": len(expected), "completed_records": len(latest), "uncompleted_cases": len(expected - latest.keys()),
               "grading_error_attempts": len(errors), "validated_tasks": len(validated),
               "validation_excluded_tasks": sorted(k for k, r in validated.items() if not r["ok"]),
               "all_completed": stats(list(latest.values())), "paired_cases": len(paired),
               "paired_deepseek": stats([latest[k] for k in paired]), "paired_terra": stats([baseline[k] for k in paired]),
               "persisted_usage": dict(usage),
               "notes": ["No-test generation is scored as a miss; execution and missing-status failures are excluded and counted.",
                         "Pa must explicitly fail and Pg must pass to count as a flag; both models are re-scored identically.",
                         "API usage can undercount requests interrupted before their reply was persisted.",
                         "Seed 0 is an experiment label; the DeepSeek API has no documented seed parameter.",
                         "C3 summaries are generated by the respective tester model."]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    with (run / "cases.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory_id", "condition", "seed", "instance_id", "resolved", "status", "flagged", "paired"])
        by_id = {r["trajectory_id"]: r for r in inputs}
        for k in sorted(expected):
            source = by_id[k[0]]
            status, flagged = outcome(latest[k]) if k in latest else ("uncompleted", None)
            writer.writerow([*k, source["instance_id"], source["resolved"], status, flagged, k in paired])
    print(json.dumps({k: summary[k] for k in ("planned_cases", "completed_records", "uncompleted_cases", "paired_cases")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    args = ap.parse_args()
    report(args.run, args.baseline)
