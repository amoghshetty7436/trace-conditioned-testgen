"""Back up one pilot job and report the unchanged harness's saved scores. Never submits jobs."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

DEST = Path(__file__).resolve().parent
REMOTE = "/pscratch/sd/i/ics35/ttg-deepseek-pilot-amogh-20261031"
HOST = "ics35@perlmutter.nersc.gov"
SOCKET = "/Users/ishaan812/.ssh/ttg-perlmutter-ics35.sock"
JOB = "60629144"
SSH = ["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST]


def remote(command):
    return subprocess.run(SSH + [command], capture_output=True, text=True, timeout=90, check=True).stdout


def sync():
    subprocess.run(["rsync", "-az", "-e", f"ssh -S {SOCKET} -o BatchMode=yes -o ConnectTimeout=15",
                    "--exclude", "repo/", f"{HOST}:{REMOTE}/", str(DEST) + "/"], timeout=120, check=True)


def records(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def case(row):
    return row["trajectory_id"], row["condition"], row["seed"]


def scores(rows):
    result = {}
    for c in ("C1", "C2", "C3", "C4", "C5", "X"):
        result[c] = {}
        for label, resolved in (("caught", 0), ("false_alarm", 1)):
            group = [r for r in rows if r["condition"] == c and r["resolved"] == resolved]
            result[c][label] = {"flags": sum(bool(r.get("patch_flagged")) for r in group), "cases": len(group)}
    return result


def report(state):
    rows = records(DEST / "pilot-data/tester_runs_deepseek.jsonl")
    latest = {case(r): r for r in rows}
    baseline = {case(r): r for r in records(DEST / "terra_pilot_baseline.jsonl")}
    common = sorted(latest.keys() & baseline.keys())
    checks = {r["instance_id"]: r for r in records(DEST / "pilot-data/task_checks_deepseek.jsonl")}
    current = scores(list(latest.values()))
    paired_base = scores([baseline[k] for k in common])
    paired_new = scores([latest[k] for k in common])
    diagnostic = Counter()
    for row in latest.values():
        diagnostic["no_test_file"] += not bool(row.get("test_code"))
        diagnostic["no_parsed_tests"] += not bool(row.get("tests"))
        diagnostic["explicit_error"] += bool(row.get("error"))
    result = {"job_id": JOB, "state": state, "planned_cases": 240, "completed_cases": len(latest),
              "paired_cases": len(common), "completed_tasks": len({r["instance_id"] for r in latest.values()}),
              "validation_checks": len(checks), "excluded_tasks": [k for k, r in checks.items() if not r.get("ok")],
              "all_deepseek": current, "paired_deepseek": paired_new, "paired_terra": paired_base,
              "diagnostics": dict(diagnostic), "tokens": {field: sum(r.get(field) or 0 for r in latest.values())
              for field in ("prompt_tokens", "completion_tokens")},
              "scoring": "Amogh's unchanged patch_flagged values; diagnostics do not alter scores.",
              "full_study_authorized": False}
    (DEST / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = ["# DeepSeek pilot", "", f"Job `{JOB}`: **{state}**. {len(latest)}/240 cases saved; {len(common)} paired with Terra.",
             "", "Amogh's unchanged commit `44ec6b20afeb790a8c5c625497c89bf0f163967d`. CPU-only, seed 0, 20 pilot tasks.",
             "Azure deployment `DeepSeek-V4-Flash`, requested reasoning high, maximum output 65,536 tokens.",
             "", "| Condition | DeepSeek catches | Terra catches | DeepSeek false alarms | Terra false alarms |",
             "|---|---:|---:|---:|---:|"]
    fmt = lambda s: f"{s['flags']}/{s['cases']}"
    for c in paired_new:
        a, b = paired_new[c], paired_base[c]
        lines.append(f"| {c} | {fmt(a['caught'])} | {fmt(b['caught'])} | {fmt(a['false_alarm'])} | {fmt(b['false_alarm'])} |")
    lines += ["", "Rates above use only matching saved cases and the original harness's scoring.",
              f"Validation exclusions: {len(result['excluded_tasks'])}. No parsed tests: {diagnostic['no_parsed_tests']}.",
              "Missing cases are not treated as completed results. Review raw execution failures before drawing conclusions.",
              "", "**Do not start the full study until the user reviews these results and explicitly approves.**"]
    (DEST / "REPORT.md").write_text("\n".join(lines) + "\n")
    return len(latest)


def main():
    deadline = time.monotonic() + 3 * 3600
    previous = None
    while time.monotonic() < deadline:
        try:
            status = remote(f"squeue -h -j {JOB} -o '%T'").strip()
            sync()
            count = report(status or "FINISHING")
            if (status, count) != previous:
                print(json.dumps({"job_id": JOB, "state": status or "FINISHING", "saved_cases": count}), flush=True)
                previous = status, count
            if not status:
                remote(f"cd {REMOTE}/repo; /pscratch/sd/i/ics35/ttg-deepseek-20261031/env/bin/python src/tester.py summarize --data ../pilot-data --out-tag deepseek > ../logs/final-summary.log 2>&1; sacct -j {JOB} -X -P -o JobID,Account,State,ElapsedRaw,AllocNodes,Timelimit > ../accounting.txt; iris user ics35 > ../allocation-after.txt; git status --short > ../git-status-after.txt")
                sync()
                text = (DEST / "accounting.txt").read_text().strip().splitlines()
                state = text[1].split("|")[2] if len(text) > 1 else "ENDED_ACCOUNTING_PENDING"
                count = report(state)
                (DEST / "collection-complete.json").write_text(json.dumps({"job_id": JOB, "state": state, "saved_cases": count, "collected_at": time.time()}, indent=2))
                print(f"Collection complete: {state}, {count}/240 cases", flush=True)
                return
        except (subprocess.SubprocessError, OSError) as exc:
            print(type(exc).__name__ + ": " + str(exc), flush=True)
        time.sleep(60)
    raise SystemExit("Monitor deadline reached; job not resubmitted. Inspect saved logs and squeue.")


if __name__ == "__main__":
    main()
