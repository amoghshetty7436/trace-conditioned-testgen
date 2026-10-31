"""Step C and D on the pilot: the tester writes tests for each agent patch, then we grade them.

For each pilot task: pull the image, clean it (sandbox.py), then for each of its two traces and
each condition:
  1. Start a fresh container from the cleaned image with the agent patch applied, and read the
     changed files (and, for C2 and up, related test files and the README).
  2. Prompt the tester with the issue, the agent patch, the changed files and the condition's extra
     information (see below). It answers with one pytest file.
  3. Run that file on the patched code and send back the failures and coverage of the changed lines.
     The tester may fix mistakes in its tests but is told never to weaken an assertion to match the
     patch. Up to ROUNDS rounds in total.
  4. Grade the final file on the original code (P0), the agent patch (Pa) and the human fix (Pg).
     A test that fails on Pa but passes on Pg flags the patch: a bug caught when the patch is wrong
     (resolved=0), a false alarm when it is right (resolved=1).

Conditions (each adds to C2, as the plan says):
  C1 issue + agent patch + changed files        C2 C1 + related test files + README
  C3 C2 + trace summary   C4 C2 + structured trace   C5 C2 + raw trace   X C2 + raw trace of another task

  python src/tester.py run [--set study] [--check] [--limit N] [--conditions C1,C5] [--seeds 1] [--restart]
      -> data/tester_runs.jsonl (every run with its prompts' sizes, final test file, grades)
         data/tester_results.csv, data/tester_summary.json
  python src/tester.py summarize

Needs data/pilot_tasks.jsonl and data/pilot_traces.jsonl.gz (`python src/pilot.py export`), Docker,
and a model endpoint (see llm.py).
"""

import argparse
import gzip
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

import llm
import traces
from pilot import (CONTAINER_CLI, PRELUDE_SH, RESULT_LINE, as_list, docker, image_name, new_files, pull_image, run_tests,
                   test_command, verdict, with_newline)
from sandbox import added_lines, check, clean, clean_tag

CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "X"]
ROUNDS = 3
FILE_KEEP = 25_000      # chars read per changed file / related test file; the prompt limits below cut further
RELATED_KEEP = 15_000
# Character budgets per prompt section. "small" keeps every call under ~13k input tokens, for
# endpoints with small per-minute token limits (Gemma on the free Gemini API allows 16k a minute).
# In "small" mode a revision round resends the issue, the patch, the changed files (and for C2 and
# up the related tests), the current test file and the failures, instead of the whole conversation;
# the trace is left out of revisions.
LIMITS = {
    "full": {"issue": 10_000, "diff": 30_000, "files": 60_000, "related": 30_000, "readme": 4_000,
             "trace": 60_000, "feedback": 6_000, "summary_input": 60_000},
    "small": {"issue": 4_000, "diff": 8_000, "files": 12_000, "related": 6_000, "readme": 1_500,
              "trace": 14_000, "feedback": 2_500, "summary_input": 36_000},
}
L = LIMITS["full"]
TEST_NAME = "test_ttg_generated.py"
RUN_TIMEOUT = 600

SYSTEM = """You are an expert Python tester. A coding agent wrote a patch for a GitHub issue. \
Your job is to write pytest tests that check whether the issue is really fixed, so that the tests \
pass on a correct fix and fail on a wrong or incomplete one. The patch may be wrong. Base your \
expectations on the issue, not on what the patched code happens to do. Test through the \
repository's public behavior, the way its existing tests do. Do not use the network."""

TASK = """# Issue
{issue}

# Agent patch (already applied in the repository at /testbed)
```diff
{diff}
```

# Changed files after the patch
{files}
{extra}
# What to write
Write one pytest file that will be saved as `{test_path}` and run from /testbed. Cover the \
behavior the issue asks for, including edge cases it implies. Reply with the complete file in a \
single ```python block."""

TRACE_HEADER = {
    "C3": "# Summary of the coding agent's work log\n",
    "C4": "# The coding agent's work log, one line per action\n",
    "C5": "# The coding agent's work log (long parts cut)\n",
    "X": "# The coding agent's work log (long parts cut)\n",
}

REVISE = """# Issue
{issue}

# Agent patch (applied at /testbed)
```diff
{diff}
```

# Changed files after the patch
{files}
{related}
# Your current test file `{test_path}`
```python
{code}
```

""" + "{feedback}"

FEEDBACK = """Your test file was run on the patched code.

```
{output}
```
{coverage}
Some tests failed or could not run. If a failure comes from a mistake in the test itself (imports, \
names, wrong use of the API, setup, fixtures, syntax), fix it. If an assertion fails because the \
patched code does something other than what the issue asks for, keep that assertion as it is: it \
may be a real bug in the patch. Never weaken an expectation just to make a test pass. Reply with \
the complete revised file in a single ```python block."""

# In the cleaned image: apply the code patch and print the files the tester gets to see.
CONTEXT_SH = PRELUDE_SH + r"""
while read -r f; do [ -n "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && rm -f "$f"; done < /work/code_new_files
if [ -s /work/code.diff ]; then apply /work/code.diff || { echo "CODE_PATCH_FAILED"; exit 0; }; fi
python - <<'PY'
import json, os, re, subprocess
want = json.load(open("/work/request.json"))
tracked = subprocess.run(["git", "ls-files"], stdout=subprocess.PIPE, universal_newlines=True).stdout.split()
tests = [t for t in tracked if t.endswith(".py") and re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", t)]
def read(p, keep):
    try:
        text = open(p, errors="replace").read()
    except OSError:
        return None
    return text if len(text) <= keep else text[:keep] + f"\n# [... {len(text) - keep} chars cut ...]\n"
out = {"files": {p: read(p, want["file_keep"]) for p in want["files"]}, "related": {}, "readme": ""}
for p in want["files"]:
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem == "__init__":
        stem = os.path.basename(os.path.dirname(p))
    hits = [t for t in tests if stem in os.path.basename(t)]
    if len(hits) < 2:
        pat = re.compile(r"^\s*(from|import)\s+[\w.]*\b" + re.escape(stem) + r"\b", re.M)
        hits += [t for t in tests if t not in hits and pat.search(read(t, 10**7) or "")][:2 - len(hits)]
    for t in hits[:2]:
        out["related"].setdefault(t, read(t, want["related_keep"]))
readme = next((f for f in sorted(os.listdir(".")) if f.lower().startswith("readme")), None)
out["readme"] = (read(readme, 4000) or "") if readme else ""
first = next(iter(out["related"]), None)
out["test_dir"] = os.path.dirname(first) if first else ("tests" if os.path.isdir("tests") else "")
print("CONTEXT_JSON " + json.dumps(out))
PY
"""

# In the cleaned image: apply the code patch, add the generated test file, run it.
TEST_SH = PRELUDE_SH + r"""
while read -r f; do [ -n "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && rm -f "$f"; done < /work/code_new_files
if [ -s /work/code.diff ]; then apply /work/code.diff || { echo "CODE_PATCH_FAILED"; exit 0; }; fi
target="$(cat /work/test_path)"
mkdir -p "$(dirname "$target")" && cp /work/generated.py "$target"
cmd="$(cat /work/test_cmd)"
if [ -s /work/cover ] && python -c "import coverage" 2>/dev/null; then
  case "$cmd" in
    pytest*) cmd="python -m coverage run --include=$(cat /work/cover) -m pytest ${cmd#pytest}" ;;
    "python -m pytest"*) cmd="python -m coverage run --include=$(cat /work/cover) -m pytest ${cmd#python -m pytest}" ;;
  esac
fi
timeout {timeout} bash -c "$cmd $target" 2>&1 | tail -n 400
if [ -s /work/cover ] && [ -f .coverage ]; then python -m coverage json -q -o /tmp/cov.json 2>/dev/null && \
  python -c "import json; print('COVERAGE_JSON ' + json.dumps({k: v['executed_lines'] for k, v in json.load(open('/tmp/cov.json'))['files'].items()}))"; fi
"""

# Anything short of "every test ran and passed" earns another round, up to ROUNDS.
NOT_CLEAN = re.compile(r"^(FAILED|ERROR) |errors? during collection|INTERNALERROR|no tests ran|"
                       r"CODE_PATCH_FAILED|RUN_ERROR", re.M)


def container(image: str, script: str, files: dict[str, str], timeout: int = RUN_TIMEOUT + 300) -> str:
    with tempfile.TemporaryDirectory() as work:
        for name, text in files.items():
            (Path(work) / name).write_text(text)
        try:
            return docker("run", "--rm", "--network", "none", "-v", f"{work}:/work:ro", image, "bash", "-c",
                          script.replace("{timeout}", str(RUN_TIMEOUT)), timeout=timeout).stdout
        except Exception as e:  # noqa: BLE001 - a hung container just counts as a failed run
            return f"RUN_ERROR {e}"


def edited_py_files(patch: str) -> list[str]:
    created = set(new_files(patch))
    files = re.findall(r"^diff --git a/\S+ b/(\S+)", patch or "", re.M)
    return [f for f in dict.fromkeys(files) if f.endswith(".py") and f not in created]


def added_line_numbers(patch: str) -> dict[str, list[int]]:
    """For each file, the new-file line numbers the patch adds."""
    out, path, line = {}, None, 0
    for l in (patch or "").splitlines():
        if l.startswith("+++ "):
            path = l[6:] if l.startswith("+++ b/") else None
        elif l.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", l)
            line = int(m.group(1)) if m else 0
        elif path and l.startswith("+"):
            out.setdefault(path, []).append(line)
            line += 1
        elif path and not l.startswith("-") and not l.startswith("\\"):
            line += 1
    return out


def code_files(code_patch: str) -> dict[str, str]:
    return {"code.diff": with_newline(code_patch), "code_new_files": "\n".join(new_files(code_patch)) + "\n"}


def context(image: str, model_patch: str) -> dict | None:
    req = {"files": edited_py_files(model_patch), "file_keep": FILE_KEEP, "related_keep": RELATED_KEEP}
    out = container(image, CONTEXT_SH, {**code_files(model_patch), "request.json": json.dumps(req)})
    m = re.search(r"^CONTEXT_JSON (.*)$", out, re.M)
    return json.loads(m.group(1)) if m else None


def run_file(image: str, code_patch: str, test_code: str, test_path: str, cmd: str,
             cover: dict[str, list[int]] | None = None) -> dict:
    files = {**code_files(code_patch), "generated.py": test_code, "test_path": test_path,
             "test_cmd": cmd + " --tb=short", "cover": ",".join(f"/testbed/{p}" for p in cover) if cover else ""}
    out = container(image, TEST_SH, files)
    results = {name: status for status, name in RESULT_LINE.findall(out) if name.startswith(test_path)}
    cov = None
    m = re.search(r"^COVERAGE_JSON (.*)$", out, re.M)
    if m and cover:
        executed = json.loads(m.group(1))
        hit = total = 0
        for path, lines in cover.items():
            ran = next((set(v) for k, v in executed.items() if k.endswith(path)), set())
            hit, total = hit + len(set(lines) & ran), total + len(lines)
        cov = round(hit / total, 3) if total else None
    text = re.sub(r"^COVERAGE_JSON .*$", "", out, flags=re.M).strip()
    return {"results": results, "output": text, "coverage": cov, "patch_failed": "CODE_PATCH_FAILED" in out}


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len).strip() + "\n" if blocks else None


def describe_files(files: dict[str, str | None], budget: int) -> str:
    files = {p: t for p, t in files.items() if t is not None}
    each = budget // max(len(files), 1)
    return "\n".join(f"## {p}\n```python\n{traces.cut(t, each)}\n```\n" for p, t in files.items()) or "(none)\n"


def build_prompt(task, trace, condition: str, ctx: dict, trace_text: str, test_path: str) -> str:
    extra = ""
    if condition != "C1":
        related = describe_files(ctx["related"], L["related"]) if ctx["related"] else "(none found)\n"
        extra += f"\n# Existing tests for these modules\n{related}"
        if ctx.get("readme"):
            extra += f"\n# README (start)\n```\n{ctx['readme'][:L['readme']]}\n```\n"
    if condition in TRACE_HEADER:
        extra += f"\n{TRACE_HEADER[condition]}```\n{traces.cut(trace_text, L['trace'])}\n```\n"
    return TASK.format(issue=traces.cut(task.problem_statement, L["issue"]), diff=traces.cut(trace.model_patch, L["diff"]),
                       files=describe_files(ctx["files"], L["files"]), extra=extra, test_path=test_path)


def summary_for(trajectory_id: str, trajectory: list, cache_path: Path) -> str:
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["trajectory_id"], r["model"])] = r["summary"]
    model = llm.settings()["model"]
    if (trajectory_id, model) not in cache:
        reply = llm.chat([{"role": "user", "content": traces.SUMMARY_PROMPT.format(trace=traces.render_raw(trajectory, L["summary_input"]))}],
                         temperature=0.0)
        append_line(cache_path, json.dumps({"trajectory_id": trajectory_id, "model": model, "summary": reply["text"]}))
        cache[(trajectory_id, model)] = reply["text"]
    return cache[(trajectory_id, model)]


def grade(image: str, task, trace, test_code: str, test_path: str, cmd: str) -> dict:
    runs = {v: run_file(image, code, test_code, test_path, cmd)
            for v, code in [("P0", ""), ("Pa", trace.model_patch), ("Pg", task.patch)]}
    ok = lambda v, t: runs[v]["results"].get(t) in ("PASSED", "XFAIL")  # noqa: E731
    tests = sorted(set().union(*(r["results"] for r in runs.values())))
    flags = [t for t in tests if not ok("Pa", t) and ok("Pg", t)]
    return {
        "tests": len(tests),
        "pass_pg": sum(ok("Pg", t) for t in tests),
        "flags": len(flags),                                                 # fail on Pa, pass on Pg
        "locked_in": sum(ok("Pa", t) and not ok("Pg", t) for t in tests),    # pass on Pa, fail on Pg
        "broken": sum(not ok("Pa", t) and not ok("Pg", t) for t in tests),   # fail on both
        "differential": any(ok("P0", t) != ok("Pa", t) for t in tests),      # notices the change at all
        "flagged_tests": flags,
        "statuses": {v: r["results"] for v, r in runs.items()},
    }


def append_line(path: Path, line: str) -> None:
    """One write() per record, so parallel shards appending to the same file never interleave."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def check_task(task, group, tag: str, cmd: str, log_path: Path) -> tuple[bool, dict]:
    """The sandbox checks of sandbox.py plus the agent labels of pilot.py validate, on the cleaned image.
    Returns whether the task is usable (one commit, no leaked fix, hidden tests fail on the base code
    and pass with the human fix) and, per trace, whether the agent patch passes the hidden tests."""
    rec = {"instance_id": task.instance_id, **check(tag, added_lines(task.patch) + added_lines(task.test_patch))}
    f2p, p2p = as_list(task.FAIL_TO_PASS), as_list(task.PASS_TO_PASS)
    base = verdict(run_tests(tag, "", task.test_patch, cmd, f2p + p2p), f2p, p2p)
    golden = verdict(run_tests(tag, task.patch, task.test_patch, cmd, f2p + p2p), f2p, p2p)
    rec.update(base_f2p=base["f2p_passed"], golden_passes=golden["passes"],
               works=bool(base["f2p_passed"] and base["f2p_passed"].startswith("0/") and golden["passes"]))
    rec["ok"] = rec["git_ok"] and rec["no_leaks"] and rec["works"]
    hidden = {}
    if rec["ok"]:
        for tr in group.itertuples():
            hidden[tr.trajectory_id] = verdict(run_tests(tag, tr.model_patch, task.test_patch, cmd, f2p + p2p), f2p, p2p)["passes"]
        rec["agent_labels_agree"] = all(hidden[tr.trajectory_id] == bool(tr.resolved) for tr in group.itertuples())
        rec["hidden_tests_pass"] = hidden
    append_line(log_path, json.dumps(rec, default=str))
    return rec["ok"], hidden


def run(args) -> None:
    data = Path(args.data)
    rows = pd.read_json(data / f"{args.set}_tasks.jsonl", lines=True, dtype=False)
    traces_path = data / f"{args.set}_traces.jsonl.gz"
    if not traces_path.exists():
        raise SystemExit(f"{traces_path} is missing: run `python src/pilot.py export` (pilot) or `export-study`")
    with gzip.open(traces_path, "rt") as f:
        trajs = {r["trajectory_id"]: r["trajectory"] for r in map(json.loads, f)}
    global L
    L = LIMITS[args.prompt_size]
    raw = {tid: traces.render_raw(t, L["trace"]) for tid, t in trajs.items()}
    lengths = {tid: sum(len(traces.text_of(m.get("content"))) for m in t if isinstance(m, dict)) for tid, t in trajs.items()}
    instance_of = dict(zip(rows.trajectory_id, rows.instance_id))
    conditions = args.conditions.split(",")
    out_path = data / f"tester_runs{suffix(args)}.jsonl"
    if args.restart and out_path.exists():
        out_path.unlink()
    # --redo-empty reruns runs that ended without a test file; summarize keeps only the latest record of a run.
    done = {(r["trajectory_id"], r["condition"], r["seed"]) for r in map(json.loads, out_path.open())
            if r.get("test_code") or not args.redo_empty} if out_path.exists() else set()

    task_ids = list(dict.fromkeys(rows.instance_id))
    if args.tasks:
        wanted = args.tasks.split(",")
        if missing := set(wanted) - set(task_ids):
            raise SystemExit(f"not {args.set} tasks: {', '.join(sorted(missing))}")
        task_ids = [t for t in task_ids if t in wanted]
    task_ids = task_ids[: args.limit or None]
    shard, shards = map(int, args.shard.split("/"))
    task_ids = task_ids[shard - 1::shards]
    for instance_id in task_ids:
        group = rows[rows.instance_id == instance_id]
        todo = [(tr, c, s) for tr in group.itertuples() for c in conditions for s in range(args.seeds)
                if (tr.trajectory_id, c, s) not in done]
        if not todo:
            continue
        task = group.iloc[0]
        tag = clean_tag(instance_id)
        print(f"[{instance_id}] preparing the cleaned image", flush=True)
        image = image_name(task.docker_image)
        pull = pull_image(image)
        error = ("pull failed: " + pull.stderr.strip()[-300:]) if pull.returncode != 0 else clean(image, tag)
        if error:
            print(f"[{instance_id}] could not prepare the image, skipping: {error}", flush=True)
            continue
        docker("rmi", "-f", image, timeout=600)
        cmd = test_command(task.install_config)
        hidden = {}
        if args.check:
            ok, hidden = check_task(task, group, tag, cmd, data / f"task_checks{suffix(args)}.jsonl")
            if not ok:
                print(f"[{instance_id}] sandbox check failed (see task_checks), skipping", flush=True)
                docker("rmi", "-f", tag, timeout=600)
                continue
        contexts = {}
        for trace, condition, seed in todo:
            if trace.trajectory_id not in contexts:
                contexts[trace.trajectory_id] = context(tag, trace.model_patch)
            ctx = contexts[trace.trajectory_id]
            label = f"[{instance_id}] {condition} seed {seed} resolved={int(trace.resolved)}"
            if ctx is None:
                print(f"{label}: agent patch did not apply, skipping", flush=True)
                continue
            test_path = f"{ctx['test_dir']}/{TEST_NAME}" if ctx["test_dir"] else TEST_NAME
            trace_text = {"C3": lambda: summary_for(trace.trajectory_id, trajs[trace.trajectory_id], data / "pilot_summaries.jsonl"),
                          "C4": lambda: traces.render_structured(trajs[trace.trajectory_id]),
                          "C5": lambda: raw[trace.trajectory_id],
                          "X": lambda: raw[traces.mismatched(trace.trajectory_id, instance_of, lengths)]}
            text = trace_text[condition]() if condition in trace_text else ""
            prompt = build_prompt(task, trace, condition, ctx, text, test_path)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            cover = {p: n for p, n in added_line_numbers(trace.model_patch).items() if p in ctx["files"]}
            rec = {"instance_id": instance_id, "trajectory_id": trace.trajectory_id, "condition": condition,
                   "seed": seed, "resolved": int(trace.resolved), "period": task.period,
                   "model": llm.settings()["model"], "prompt_size": args.prompt_size, "prompt_chars": len(prompt), "rounds": 0,
                   "prompt_tokens": 0, "completion_tokens": 0, "coverage": None, "feedback": [],
                   "hidden_tests_pass": hidden.get(trace.trajectory_id)}
            code = None
            try:
                for rnd in range(ROUNDS):
                    reply = llm.chat(messages, seed=seed)
                    rec["rounds"] = rnd + 1
                    rec["prompt_tokens"] += reply["prompt_tokens"] or 0
                    rec["completion_tokens"] += reply["completion_tokens"] or 0
                    code = extract_code(reply["text"]) or code
                    if code is None:
                        break
                    fb = run_file(tag, trace.model_patch, code, test_path, cmd, cover)
                    rec["coverage"] = fb["coverage"]
                    rec["feedback"].append(fb["output"][-3000:])
                    if not NOT_CLEAN.search(fb["output"]) or rnd == ROUNDS - 1:
                        break
                    cov = f"\nCoverage of the patch's added lines: {fb['coverage']:.0%}\n" if fb["coverage"] is not None else ""
                    feedback = FEEDBACK.format(output=fb["output"][-L["feedback"]:], coverage=cov)
                    if args.prompt_size == "small":
                        related = "" if condition == "C1" or not ctx["related"] else \
                            f"\n# Existing tests for these modules\n{describe_files(ctx['related'], L['related'])}"
                        messages = [messages[0], {"role": "user", "content": REVISE.format(
                            issue=traces.cut(task.problem_statement, L["issue"]), test_path=test_path, code=code,
                            diff=traces.cut(trace.model_patch, L["diff"]), feedback=feedback,
                            files=describe_files(ctx["files"], L["files"]), related=related)}]
                    else:
                        messages += [{"role": "assistant", "content": reply["text"]}, {"role": "user", "content": feedback}]
            except RuntimeError as e:
                # Nothing is saved for this run, so rerunning the same command picks it up again.
                print(f"{label}: {str(e)[:500]}", flush=True)
                if "HTTP 429" in str(e):
                    raise SystemExit("still rate-limited after retries (daily quota?); rerun later to resume")
                if "HTTP 4" in str(e):
                    raise SystemExit("the model endpoint refused the request; check the key and model name")
                continue
            rec["test_path"], rec["test_code"] = test_path, code
            if code:
                rec.update(grade(tag, task, trace, code, test_path, cmd))
            rec["patch_flagged"] = bool(rec.get("flags"))
            print(f"{label}: {rec.get('tests', 0)} tests, {rec.get('flags', 0)} flag the patch, "
                  f"{rec.get('broken', 0)} broken, coverage {rec['coverage']}", flush=True)
            append_line(out_path, json.dumps(rec))
        if not args.keep_images:
            docker("rmi", "-f", tag, timeout=600)
            if CONTAINER_CLI == "docker":  # with parallel podman-hpc shards a prune can hit another shard's container
                docker("system", "prune", "-f", timeout=600)
    summarize(args)


def suffix(args) -> str:
    """--out-tag gpt_oss_120b writes tester_runs_gpt_oss_120b.jsonl etc., keeping other models' runs apart."""
    return f"_{args.out_tag}" if args.out_tag else ""


def summarize(args) -> None:
    data = Path(args.data)
    path = data / f"tester_runs{suffix(args)}.jsonl"
    if not path.exists():
        print(f"no runs in {path} yet")
        return
    latest = {(r["trajectory_id"], r["condition"], r["seed"]): r for r in map(json.loads, path.open())}
    recs = list(latest.values())
    df = pd.DataFrame(recs)
    cols = ["instance_id", "trajectory_id", "period", "condition", "seed", "resolved", "model", "rounds", "tests",
            "pass_pg", "flags", "locked_in", "broken", "differential", "coverage", "patch_flagged",
            "prompt_tokens", "completion_tokens", "error"]
    df[[c for c in cols if c in df]].to_csv(data / f"tester_results{suffix(args)}.csv", index=False)

    def stats(g: pd.DataFrame) -> dict:
        wrong, right = g[g.resolved == 0], g[g.resolved == 1]
        return {"runs": int(len(g)),
                "bug_caught_rate": round(float(wrong.patch_flagged.mean()), 3) if len(wrong) else None,
                "false_alarm_rate": round(float(right.patch_flagged.mean()), 3) if len(right) else None,
                "no_test_file": int(g.tests.isna().sum()) if "tests" in g else int(len(g)),
                "mean_tests": round(float(g.tests.mean()), 2) if "tests" in g else None,
                "mean_broken": round(float(g.broken.mean()), 2) if "broken" in g else None,
                "differential_rate": round(float(g.differential.fillna(False).astype(bool).mean()), 3) if "differential" in g else None,
                "mean_coverage": round(float(g.coverage.dropna().mean()), 3) if g.coverage.notna().any() else None,
                "mean_prompt_tokens": int(g.prompt_tokens.mean())}

    summary = {"model": sorted(df.model.unique().tolist()),
               "by_condition": {c: stats(g) for c, g in df.groupby("condition")},
               "by_condition_and_period": {f"{c} {p}": stats(g) for (c, p), g in df.groupby(["condition", "period"])}}
    (data / f"tester_summary{suffix(args)}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_condition"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarize"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--set", default="pilot", choices=["pilot", "study"],
                    help="pilot: data/pilot_tasks.jsonl (20 tasks); study: data/study_tasks.jsonl (all selected tasks)")
    ap.add_argument("--check", action="store_true",
                    help="check each cleaned image first (no leaked fix, hidden tests behave) and skip tasks that fail")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--tasks", default="", help="comma-separated instance ids: only these pilot tasks")
    ap.add_argument("--out-tag", default="", help="name suffix for the output files, e.g. the model")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--prompt-size", choices=list(LIMITS), default="full",
                    help="small: keep each call under ~13k input tokens (for Gemma on the free Gemini API)")
    ap.add_argument("--shard", default="1/1", help="i/n: run every n-th task starting at the i-th, to run n processes at once")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--redo-empty", action="store_true", help="rerun runs that produced no test file")
    ap.add_argument("--restart", action="store_true", help="delete earlier runs and start over")
    args = ap.parse_args()
    run(args) if args.step == "run" else summarize(args)


if __name__ == "__main__":
    main()
