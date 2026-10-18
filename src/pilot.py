"""Pilot check: do the chosen tasks run, and do the dataset's labels hold up?

Two steps, because the full traces live on the machine that ran select_tasks.py and the
Docker runs need an x86 Linux machine with plenty of disk (task images are 1 to 30 GB).

  python src/pilot.py export      # on the machine with data/selected_traces.parquet
      -> data/pilot_tasks.jsonl (pilot rows, patches and tests only, no trace text)

  python src/pilot.py validate    # on a machine with Docker
      -> data/pilot_validation.csv, data/pilot_validation.json

For each pilot task, `validate` starts the task's image and runs the hidden human tests:
  base    no code change          FAIL_TO_PASS should fail
  golden  the human fix           everything should pass (checks the harness itself)
  agent   each chosen agent patch passes or not -> compared with the dataset's `resolved`
It also counts commits reachable from other refs that are newer than HEAD (future history
the tester's sandbox must strip). Images are deleted after each task unless --keep-images.
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

EXPORT_COLUMNS = [
    "instance_id", "trajectory_id", "period", "resolved", "exit_status", "docker_image",
    "base_commit", "created_at", "model_patch", "patch", "test_patch", "FAIL_TO_PASS",
    "PASS_TO_PASS", "install_config",
]
DEFAULT_TEST_CMD = "pytest --no-header -rA --tb=no -p no:cacheprovider"
RESULT_LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.M)
RUN_TIMEOUT = 1800

# Runs inside the container. /work holds code.diff, tests.diff, test_files, test_cmd.
RUN_SH = r"""
source /opt/conda/bin/activate testbed 2>/dev/null || true
cd /testbed
git config --global --add safe.directory /testbed
# Some images were built with the repo at another path (e.g. /borgstore) and installed in
# editable mode from there; point that missing path at /testbed so the package imports.
site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
for p in $( (cat "$site"/*.pth "$site"/*.egg-link 2>/dev/null | grep '^/'; \
             grep -l '"editable": true' "$site"/*.dist-info/direct_url.json 2>/dev/null \
               | xargs -r grep -ho '"file://[^"]*' | cut -c9-) | sort -u); do
  top="/$(echo "$p" | cut -d/ -f2)"; [ -e "$p" ] || [ -e "$top" ] || ln -s /testbed "$top"
done
# Others carry a plain installed copy of the package that shadows /testbed; put the repo first.
shadowed=$(cd / && python - <<'PY'
import importlib.util, os
for root in ("/testbed/src", "/testbed"):
    pkgs = [n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, "__init__.py"))] if os.path.isdir(root) else []
    for n in pkgs:
        try:
            spec = importlib.util.find_spec(n)
        except Exception:
            continue
        if spec and spec.origin and not os.path.realpath(spec.origin).startswith("/testbed/"):
            print(root)
            raise SystemExit
PY
)
[ -n "$shadowed" ] && export PYTHONPATH="$shadowed${PYTHONPATH:+:$PYTHONPATH}"
# --recount: some stored patches lost their trailing context lines, leaving hunk counts off.
apply() { git apply --recount --whitespace=nowarn "$1" 2>/dev/null || patch -p1 --forward --fuzz=3 -i "$1" >/dev/null; }
# Drop untracked files the patch re-creates (the agent's copy of /testbed/issue.md, say).
while read -r f; do [ -n "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && rm -f "$f"; done < /work/code_new_files
if [ -s /work/code.diff ]; then apply /work/code.diff || { echo "CODE_PATCH_FAILED"; exit 0; }; fi
# Reset the test files to the base commit so an agent's own test edits can't clash.
while read -r f; do [ -n "$f" ] && { git checkout HEAD -- "$f" 2>/dev/null || rm -f "$f"; }; done < /work/test_patch_files
apply /work/tests.diff || { echo "TEST_PATCH_FAILED"; exit 0; }
timeout {timeout} bash -c "$(cat /work/test_cmd) $(tr '\n' ' ' < /work/test_files)" 2>&1 | tail -n 3000
"""

HISTORY_SH = r"""
cd /testbed && git config --global --add safe.directory /testbed
head_date=$(git log -1 --format=%ct HEAD)
echo "HEAD_DATE $(git log -1 --format=%cI HEAD)"
echo "HEAD_COMMITS $(git rev-list --count HEAD)"
echo "ALL_COMMITS $(git rev-list --count --all)"
echo "NEWER_THAN_HEAD $(git log --all --format=%ct | awk -v h=$head_date '$1 > h' | wc -l)"
echo "NEWEST $(git log --all -1 --format=%cI)"
"""


def as_list(value) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
    return [str(v) for v in value] if value is not None else []


def as_dict(value) -> dict:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if value else {}


def patch_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", patch or "", re.M)))


def new_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/\S+ b/(\S+)\nnew file mode", patch or "", re.M)


def with_newline(patch) -> str:
    """Stored patches are sometimes stripped; git and patch need the final newline."""
    return patch if not patch or patch.endswith("\n") else patch + "\n"


def test_command(install_config) -> str:
    cmd = as_dict(install_config).get("test_cmd") or DEFAULT_TEST_CMD
    cmd = " && ".join(cmd) if isinstance(cmd, list) else str(cmd)
    if "pytest" in cmd and "-rA" not in cmd:
        cmd += " -rA"
    if "pytest" in cmd:
        cmd += " --continue-on-collection-errors"  # one broken test file shouldn't hide the rest
    return cmd


def docker(*args, timeout=None, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=check)


def run_tests(image: str, code_patch: str, test_patch: str, test_cmd: str, test_ids: list[str]) -> dict:
    """Apply a code patch plus the hidden tests in a fresh container; return {test_id: status}."""
    with tempfile.TemporaryDirectory() as work:
        w = Path(work)
        (w / "code.diff").write_text(with_newline(code_patch))
        (w / "tests.diff").write_text(with_newline(test_patch))
        (w / "code_new_files").write_text("\n".join(new_files(code_patch)) + "\n")
        (w / "test_patch_files").write_text("\n".join(patch_files(test_patch)) + "\n")
        # Run the files the test ids name; the test patch can also touch data files (cassettes).
        files = sorted({t.split("::")[0] for t in test_ids if "::" in t}
                       or {f for f in patch_files(test_patch) if f.endswith(".py")})
        (w / "test_files").write_text("\n".join(files) + "\n")
        (w / "test_cmd").write_text(test_cmd)
        script = RUN_SH.replace("{timeout}", str(RUN_TIMEOUT))
        try:
            out = docker("run", "--rm", "--network", "none", "-v", f"{work}:/work:ro", image,
                         "bash", "-c", script, timeout=RUN_TIMEOUT + 300).stdout
        except subprocess.TimeoutExpired:
            return {"_error": "timeout"}
    if "CODE_PATCH_FAILED" in out:
        return {"_error": "code_patch_failed"}
    if "TEST_PATCH_FAILED" in out:
        return {"_error": "test_patch_failed"}
    return {name: status for status, name in RESULT_LINE.findall(out)}


def verdict(results: dict, f2p: list[str], p2p: list[str]) -> dict:
    if "_error" in results:
        return {"error": results["_error"], "f2p_passed": None, "p2p_passed": None, "passes": None}
    ok = lambda t: results.get(t) in ("PASSED", "XFAIL")
    f2p_ok, p2p_ok = sum(map(ok, f2p)), sum(map(ok, p2p))
    missing = sum(t not in results for t in f2p + p2p)
    return {"error": "", "f2p_passed": f"{f2p_ok}/{len(f2p)}", "p2p_passed": f"{p2p_ok}/{len(p2p)}",
            "tests_not_found": missing, "passes": f2p_ok == len(f2p) and p2p_ok == len(p2p)}


def history(image: str) -> dict:
    out = docker("run", "--rm", "--network", "none", image, "bash", "-c", HISTORY_SH, timeout=600).stdout
    vals = dict(line.split(" ", 1) for line in out.strip().splitlines() if " " in line)
    return {"head_date": vals.get("HEAD_DATE"), "commits_newer_than_head": int(vals.get("NEWER_THAN_HEAD", -1)),
            "newest_commit": vals.get("NEWEST"),
            "unreachable_from_head": int(vals.get("ALL_COMMITS", 0)) - int(vals.get("HEAD_COMMITS", 0))}


def export(args) -> None:
    df = pd.read_parquet(Path(args.data) / "selected_traces.parquet")
    df = df[df.pilot.astype(bool)]
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    out = Path(args.data) / "pilot_tasks.jsonl"
    df[cols].to_json(out, orient="records", lines=True, default_handler=str)
    print(f"wrote {len(df)} traces from {df.instance_id.nunique()} tasks to {out}")


def validate(args) -> None:
    data = Path(args.data)
    rows = pd.read_json(data / "pilot_tasks.jsonl", lines=True, dtype=False)
    if args.limit:
        rows = rows[rows.instance_id.isin(rows.instance_id.drop_duplicates().head(args.limit))]
    out_csv = data / "pilot_validation.csv"
    done = set(pd.read_csv(out_csv).instance_id) if out_csv.exists() and not args.restart else set()
    records = pd.read_csv(out_csv).to_dict("records") if done else []

    for instance_id, task in rows.groupby("instance_id", sort=False):
        if instance_id in done:
            continue
        t = task.iloc[0]
        image = t.docker_image
        f2p, p2p = as_list(t.FAIL_TO_PASS), as_list(t.PASS_TO_PASS)
        cmd = test_command(t.get("install_config"))
        print(f"[{instance_id}] pulling {image}", flush=True)
        for attempt in range(4):  # Docker Hub rate limits (429) clear after a while
            pull = docker("pull", image, timeout=3600)
            if pull.returncode == 0 or "429" not in pull.stderr:
                break
            time.sleep(300 * (attempt + 1))
        if pull.returncode != 0:
            records.append({"instance_id": instance_id, "run": "pull", "error": pull.stderr.strip()[-300:]})
            continue
        hist = history(image)
        runs = [("base", "", None), ("golden", t.patch, None)] + [
            (f"agent_{int(r.resolved)}", r.model_patch, r) for r in task.itertuples()]
        for name, code, trace in runs:
            print(f"[{instance_id}] {name}", flush=True)
            v = verdict(run_tests(image, code, t.test_patch, cmd, f2p + p2p), f2p, p2p)
            rec = {"instance_id": instance_id, "run": name, "period": t.period, **v, **hist}
            if trace is not None:
                rec.update(trajectory_id=trace.trajectory_id, exit_status=trace.exit_status,
                           dataset_resolved=int(trace.resolved),
                           label_agrees=None if v["passes"] is None else v["passes"] == bool(trace.resolved))
            records.append(rec)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        if not args.keep_images:
            docker("rmi", "-f", image, timeout=600)
            docker("system", "prune", "-f", timeout=600)

    res = pd.DataFrame(records)
    summary = summarize(res)
    (data / "pilot_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(res: pd.DataFrame) -> dict:
    def runs(name):
        return res[res.run == name] if "run" in res else res.iloc[0:0]
    agents = res[res.run.astype(str).str.startswith("agent")] if "run" in res else res
    base, golden = runs("base"), runs("golden")
    return {
        "tasks": int(res.instance_id.nunique()),
        "pull_failures": int((res.run == "pull").sum()),
        "base_f2p_all_fail": int((base.f2p_passed.astype(str).str.startswith("0/")).sum()),
        "golden_passes": int((golden.passes == True).sum()),  # noqa: E712
        "golden_runs": int(len(golden)),
        "agent_runs": int(len(agents)),
        "agent_label_agrees": int((agents.label_agrees == True).sum()),  # noqa: E712
        "agent_label_disagrees": int((agents.label_agrees == False).sum()),  # noqa: E712
        "agent_errors": agents.error.fillna("").replace("", pd.NA).dropna().value_counts().to_dict(),
        "tasks_with_future_history": int((base.commits_newer_than_head > 0).sum()) if len(base) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "validate"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=None, help="validate only the first N tasks")
    ap.add_argument("--keep-images", action="store_true", help="don't delete images after each task")
    ap.add_argument("--restart", action="store_true", help="ignore earlier results and start over")
    args = ap.parse_args()
    export(args) if args.step == "export" else validate(args)


if __name__ == "__main__":
    main()
