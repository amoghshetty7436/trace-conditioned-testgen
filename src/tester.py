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

  python src/tester.py run [--limit N] [--conditions C1,C5] [--seeds 1] [--restart]
      -> data/tester_runs.jsonl (every run with its prompts' sizes, final test file, grades)
         data/tester_results.csv, data/tester_summary.json
  python src/tester.py summarize

Needs data/pilot_tasks.jsonl and data/pilot_traces.jsonl.gz (`python src/pilot.py export`), Docker,
and a model endpoint (see llm.py).
"""

import argparse
import fcntl
import gzip
import hashlib
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import llm
import traces
from pilot import PRELUDE_SH, RESULT_LINE, as_list, docker, image_name, new_files, pull_image, run_tests, test_command, verdict, with_newline
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
DOCKER_SLOTS = threading.BoundedSemaphore(1)
RECORD_LOCK = threading.Lock()
SUMMARY_LOCKS = {}


def append_record(path: Path, record: dict) -> None:
    with RECORD_LOCK, path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def cached_chat(path: Path, messages: list[dict], seed: int) -> dict:
    """Persist each paid reply before executing tests; reuse only identical requests."""
    settings = {k: v for k, v in llm.settings().items() if k != "api_key"}
    fingerprint = hashlib.sha256(json.dumps([settings, messages, seed], sort_keys=True).encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["request_sha256"] == fingerprint:
            return {**saved["reply"], "local_cache_hit": True}
    reply = llm.chat(messages, seed=seed)
    atomic_json(path, {"request_sha256": fingerprint, "request_messages": messages, "reply": reply})
    return {**reply, "local_cache_hit": False}

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
tracked = subprocess.run(["git", "ls-files"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.split()
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
        name = "ttg-run-" + uuid.uuid4().hex
        with DOCKER_SLOTS:
            try:
                result = docker("run", "--rm", "--name", name, "--network", "none", "-v", f"{work}:/work:ro", image, "bash", "-c",
                                script.replace("{timeout}", str(RUN_TIMEOUT)), timeout=timeout)
                if result.returncode:
                    return f"RUN_ERROR exit={result.returncode}\n{result.stdout}\n{result.stderr}"
                return result.stdout
            except Exception as e:
                # Killing the Docker CLI alone can leave the container running.
                docker("rm", "-f", name, timeout=30)
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
    if not m:
        raise RuntimeError("Context extraction failed: " + out[-3000:])
    return json.loads(m.group(1))


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
    with RECORD_LOCK:
        lock = SUMMARY_LOCKS.setdefault((str(cache_path), trajectory_id, llm.settings()["model"]), threading.Lock())
    with lock:
        return _summary_for(trajectory_id, trajectory, cache_path)


def _summary_for(trajectory_id: str, trajectory: list, cache_path: Path) -> str:
    cache = {}
    with RECORD_LOCK:
        lines = cache_path.read_text().splitlines() if cache_path.exists() else []
    for line in lines:
        r = json.loads(line)
        cache[(r["trajectory_id"], r["model"])] = r["summary"]
    model = llm.settings()["model"]
    if (trajectory_id, model) not in cache:
        reply = llm.chat([{"role": "user", "content": traces.SUMMARY_PROMPT.format(trace=traces.render_raw(trajectory, L["summary_input"]))}],
                         temperature=0.0)
        append_record(cache_path, {"trajectory_id": trajectory_id, "model": model, "summary": reply["text"], "reply": reply})
        cache[(trajectory_id, model)] = reply["text"]
    return cache[(trajectory_id, model)]


def grade(image: str, task, trace, test_code: str, test_path: str, cmd: str) -> dict:
    runs = {v: run_file(image, code, test_code, test_path, cmd)
            for v, code in [("P0", ""), ("Pa", trace.model_patch), ("Pg", task.patch)]}
    invalid = {v: r["output"] for v, r in runs.items()
               if r["patch_failed"] or "RUN_ERROR" in r["output"] or not r["results"]}
    if invalid:
        return {"grade_error": invalid, "statuses": {v: r["results"] for v, r in runs.items()}}
    ok = lambda v, t: runs[v]["results"].get(t) in ("PASSED", "XFAIL")  # noqa: E731
    tests = sorted(set().union(*(r["results"] for r in runs.values())))
    flags = [t for t in tests if not ok("Pa", t) and ok("Pg", t)]
    return {
        "grading_outputs": {v: r["output"] for v, r in runs.items()},
        "tests": len(tests),
        "pass_pg": sum(ok("Pg", t) for t in tests),
        "flags": len(flags),                                                 # fail on Pa, pass on Pg
        "locked_in": sum(ok("Pa", t) and not ok("Pg", t) for t in tests),    # pass on Pa, fail on Pg
        "broken": sum(not ok("Pa", t) and not ok("Pg", t) for t in tests),   # fail on both
        "differential": any(ok("P0", t) != ok("Pa", t) for t in tests),      # notices the change at all
        "flagged_tests": flags,
        "statuses": {v: r["results"] for v, r in runs.items()},
    }


def run_case(task, trace, condition, seed, ctx, tag, cmd, data, prompt_size,
             trajs, raw, instance_of, lengths, source_version):
    started = time.monotonic()
    instance_id = task.instance_id
    label = f"[{instance_id}] {condition} seed {seed} resolved={int(trace.resolved)}"
    case_id = hashlib.sha256(json.dumps([trace.trajectory_id, condition, seed]).encode()).hexdigest()
    checkpoint = data / "checkpoints" / case_id
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
           "model": llm.settings()["model"], "prompt_size": prompt_size, "prompt_chars": len(prompt), "rounds": 0,
           "prompt_tokens": 0, "completion_tokens": 0, "coverage": None, "feedback": [],
           "reasoning_effort": llm.settings()["reasoning_effort"], "round_records": []}
    if condition == "X":
        rec["mismatched_trajectory_id"] = traces.mismatched(trace.trajectory_id, instance_of, lengths)
    code = None
    state_path = checkpoint / "generation.json"
    state_key = hashlib.sha256(json.dumps([
        {k: v for k, v in llm.settings().items() if k != "api_key"}, messages, seed,
        trace.model_patch, test_path, cmd, prompt_size, ROUNDS, FEEDBACK, REVISE, TEST_SH,
    ], sort_keys=True).encode()).hexdigest()
    first_round = 0
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved["state_key"] == state_key:
            rec, messages, code, first_round = saved["record"], saved["messages"], saved["code"], saved["next_round"]
            rec["resumed_rounds"] = rec["rounds"]
            print(f"{label}: resuming after {rec['rounds']} saved rounds", flush=True)
    for rnd in range(first_round, ROUNDS):
        print(f"{label}: round {rnd + 1}/{ROUNDS}", flush=True)
        reply = cached_chat(checkpoint / f"round-{rnd}.json", messages, seed)
        rec["rounds"] = rnd + 1
        rec["prompt_tokens"] += reply["prompt_tokens"] or 0
        rec["completion_tokens"] += reply["completion_tokens"] or 0
        code = extract_code(reply["text"]) or code
        rec["round_records"].append({**reply, "test_code": code})
        if code is None:
            break
        fb = run_file(tag, trace.model_patch, code, test_path, cmd, cover)
        rec["coverage"] = fb["coverage"]
        rec["feedback"].append(fb["output"][-3000:])
        finished = not NOT_CLEAN.search(fb["output"]) or rnd == ROUNDS - 1
        if finished:
            atomic_json(state_path, {"state_key": state_key, "record": rec, "messages": messages,
                                    "code": code, "next_round": ROUNDS})
            break
        cov = f"\nCoverage of the patch's added lines: {fb['coverage']:.0%}\n" if fb["coverage"] is not None else ""
        feedback = FEEDBACK.format(output=fb["output"][-L["feedback"]:], coverage=cov)
        if prompt_size == "small":
            related = "" if condition == "C1" or not ctx["related"] else \
                f"\n# Existing tests for these modules\n{describe_files(ctx['related'], L['related'])}"
            messages = [messages[0], {"role": "user", "content": REVISE.format(
                issue=traces.cut(task.problem_statement, L["issue"]), test_path=test_path, code=code,
                diff=traces.cut(trace.model_patch, L["diff"]), feedback=feedback,
                files=describe_files(ctx["files"], L["files"]), related=related)}]
        else:
            messages += [{"role": "assistant", "content": reply["text"]}, {"role": "user", "content": feedback}]
        atomic_json(state_path, {"state_key": state_key, "record": rec, "messages": messages,
                                "code": code, "next_round": rnd + 1})
    rec["test_path"], rec["test_code"] = test_path, code
    grading_started = time.monotonic()
    if code:
        rec.update(grade(tag, task, trace, code, test_path, cmd))
    rec["grade_elapsed_seconds"] = round(time.monotonic() - grading_started, 3)
    rec["elapsed_seconds"] = round(time.monotonic() - started, 3)
    rec["source_version"] = source_version
    rec["patch_flagged"] = bool(rec.get("flags"))
    return rec


def check_manifest(previous, current, allow_code_change=False):
    """Explicit source upgrades may resume; model/data/protocol changes never mix."""
    if previous == current:
        return
    old_protocol = {k: v for k, v in previous.items() if k != "source_sha256"}
    new_protocol = {k: v for k, v in current.items() if k != "source_sha256"}
    if old_protocol != new_protocol or not allow_code_change:
        raise SystemExit("Experiment configuration changed; use a separate --data directory. "
                         "For a reviewed source-only upgrade, use --allow-code-change.")


def run(args) -> None:
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    with (data / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another runner is already using {data}")
        _run(args)


def _run(args) -> None:
    data = Path(args.data)
    global DOCKER_SLOTS
    DOCKER_SLOTS = threading.BoundedSemaphore(args.docker_workers)
    config = {k: v for k, v in llm.settings().items() if k != "api_key"}
    config.update(prompt_size=args.prompt_size, rounds=ROUNDS, validate_images=args.validate_images,
                  validation_policy=args.validation_policy, task_filter=args.tasks,
                  tasks_sha256=hashlib.sha256((data / "pilot_tasks.jsonl").read_bytes()).hexdigest(),
                  traces_sha256=hashlib.sha256((data / "pilot_traces.jsonl.gz").read_bytes()).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*.py")})
    manifest = data / "tester_config.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and not args.restart:
        check_manifest(previous, config, args.allow_code_change)
    if not manifest.exists() and (data / "tester_runs.jsonl").exists() and not args.restart:
        raise SystemExit("Existing runs have no configuration manifest; use a separate --data directory.")
    if previous and previous != config:
        append_record(data / "config_history.jsonl", {"changed_at": time.time(), "previous": previous, "current": config})
    atomic_json(manifest, config)
    source_version = hashlib.sha256(json.dumps(config["source_sha256"], sort_keys=True).encode()).hexdigest()
    append_record(data / "execution_history.jsonl", {"started_at": time.time(), "source_version": source_version,
                  "workers": args.workers, "docker_workers": args.docker_workers})
    rows = pd.read_json(data / "pilot_tasks.jsonl", lines=True, dtype=False)
    traces_path = data / "pilot_traces.jsonl.gz"
    if not traces_path.exists():
        raise SystemExit(f"{traces_path} is missing: run `python src/pilot.py export` where selected_traces.parquet is")
    with gzip.open(traces_path, "rt") as f:
        trajs = {r["trajectory_id"]: r["trajectory"] for r in map(json.loads, f)}
    global L
    L = LIMITS[args.prompt_size]
    raw = {tid: traces.render_raw(t, L["trace"]) for tid, t in trajs.items()}
    lengths = {tid: sum(len(traces.text_of(m.get("content"))) for m in t if isinstance(m, dict)) for tid, t in trajs.items()}
    instance_of = dict(zip(rows.trajectory_id, rows.instance_id))
    conditions = args.conditions.split(",")
    out_path = data / "tester_runs.jsonl"
    if args.restart and out_path.exists():
        out_path.unlink()
    if args.restart and (data / "checkpoints").exists():
        shutil.rmtree(data / "checkpoints")
    done = {(r["trajectory_id"], r["condition"], r["seed"]) for r in map(json.loads, out_path.open())} \
        if out_path.exists() else set()

    task_ids = list(dict.fromkeys(rows.instance_id))[: args.limit or None]
    if args.tasks:
        wanted = args.tasks.split(",")
        if set(wanted) - set(task_ids):
            raise SystemExit("Task filter contains unknown tasks")
        task_ids = [t for t in task_ids if t in wanted]
    sizes_path = data / "image_sizes.json"
    sizes = {r["instance_id"]: r.get("compressed_gb", 0) for r in json.loads(sizes_path.read_text())} if sizes_path.exists() else {}
    shard, shards = map(int, args.shard.split("/"))
    task_ids = task_ids[shard - 1::shards]
    for instance_id in task_ids:
        group = rows[rows.instance_id == instance_id]
        todo = [(tr, c, s) for tr in group.itertuples() for c in conditions for s in range(args.seeds)
                if (tr.trajectory_id, c, s) not in done]
        if not todo:
            continue
        task = group.iloc[0]
        image = image_name(task.docker_image)
        tag = clean_tag(instance_id)
        free_gb = shutil.disk_usage(data).free / 1e9
        already_pulled = docker("image", "inspect", image).returncode == 0
        needed_gb = 3 if already_pulled else 4 * sizes.get(instance_id, 0) + 3
        if free_gb < needed_gb:
            failure = {"instance_id": instance_id, "error": "insufficient_disk", "free_gb": round(free_gb, 2),
                       "estimated_required_gb": round(needed_gb, 2)}
            with (data / "task_errors.jsonl").open("a") as f:
                f.write(json.dumps(failure) + "\n")
            print(f"[{instance_id}] insufficient disk ({free_gb:.1f} GB free, reserving {needed_gb:.1f} GB); skipping", flush=True)
            continue
        print(f"[{instance_id}] preparing the cleaned image", flush=True)
        if (not already_pulled and pull_image(image).returncode != 0) or clean(image, tag):
            append_record(data / "task_errors.jsonl", {"instance_id": instance_id, "error": "image_preparation_failed"})
            print(f"[{instance_id}] could not prepare the image, skipping", flush=True)
            continue
        docker("rmi", "-f", image, timeout=600)
        cmd = test_command(task.install_config)
        if args.validate_images:
            print(f"[{instance_id}] validating cleaned image and both agent labels", flush=True)
            validation = {"instance_id": instance_id, **check(tag, added_lines(task.patch) + added_lines(task.test_patch))}
            f2p, p2p = as_list(task.FAIL_TO_PASS), as_list(task.PASS_TO_PASS)
            variants = [("base", ""), ("golden", task.patch)] + [
                (f"agent_{int(r.resolved)}", r.model_patch) for r in group.itertuples()]
            for name, patch in variants:
                validation[name] = verdict(run_tests(tag, patch, task.test_patch, cmd, f2p + p2p), f2p, p2p)
            validation["ok"] = bool(validation["git_ok"] and validation["no_leaks"]
                                    and validation["base"].get("f2p_passed") == f"0/{len(f2p)}"
                                    and validation["golden"].get("passes") is True
                                    and validation["agent_0"].get("passes") is False
                                    and validation["agent_1"].get("passes") is True
                                    and all(not validation[n].get("error") and not validation[n].get("tests_not_found")
                                            for n, _ in variants))
            if args.validation_policy == "terra":
                # Match Amogh's inclusion gate; retain stricter diagnostics separately.
                validation["strict_ok"] = validation["ok"]
                validation["ok"] = bool(validation["git_ok"] and validation["no_leaks"]
                                        and (validation["base"].get("f2p_passed") or "").startswith("0/")
                                        and validation["golden"].get("passes"))
            with (data / "image_validation.jsonl").open("a") as f:
                f.write(json.dumps(validation) + "\n")
            if not validation["ok"]:
                print(f"[{instance_id}] validation failed; see image_validation.jsonl", flush=True)
                if not args.keep_images:
                    docker("rmi", "-f", tag, timeout=600)
                continue
        task_started = time.monotonic()
        contexts = {}
        # Read each patched repository once, before sharing immutable context with workers.
        needed_traces = {tr.trajectory_id: tr for tr, _, _ in todo}
        with ThreadPoolExecutor(max_workers=args.docker_workers) as pool:
            futures = {pool.submit(context, tag, tr.model_patch): tid for tid, tr in needed_traces.items()}
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    contexts[tid] = future.result()
                except RuntimeError as e:
                    append_record(data / "task_errors.jsonl", {"instance_id": instance_id, "trajectory_id": tid, "error": str(e)})
                    print(f"[{instance_id}] {e}", flush=True)
        jobs = [(tr, c, seed) for tr, c, seed in todo if tr.trajectory_id in contexts]
        fatal_error = None
        completed = 0
        print(f"[{instance_id}] running {len(jobs)} cases with {args.workers} workers / {args.docker_workers} Docker slots", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, task, tr, c, seed, contexts[tr.trajectory_id], tag, cmd, data,
                                   args.prompt_size, trajs, raw, instance_of, lengths, source_version): (tr, c, seed)
                       for tr, c, seed in jobs}
            for future in as_completed(futures):
                tr, condition, seed = futures[future]
                label = f"[{instance_id}] {condition} seed {seed} resolved={int(tr.resolved)}"
                try:
                    rec = future.result()
                except CancelledError:
                    continue
                except Exception as e:
                    append_record(data / "tester_errors.jsonl", {"instance_id": instance_id,
                                  "trajectory_id": tr.trajectory_id, "condition": condition, "seed": seed, "error": str(e)})
                    print(f"{label}: {str(e)[:500]}", flush=True)
                    if "HTTP 4" in str(e):
                        fatal_error = str(e)
                        for pending in futures:
                            pending.cancel()
                    continue
                if rec.get("grade_error"):
                    append_record(data / "tester_errors.jsonl", rec)
                    print(f"{label}: invalid grading run; saved for diagnosis and retry", flush=True)
                    continue
                append_record(out_path, rec)
                completed += 1
                print(f"{label}: {rec.get('tests', 0)} tests, {rec.get('flags', 0)} flag the patch, "
                      f"{rec.get('broken', 0)} broken, coverage {rec['coverage']} "
                      f"({completed}/{len(jobs)} completed, {rec['elapsed_seconds']:.1f}s)", flush=True)
        append_record(data / "timings.jsonl", {"instance_id": instance_id, "completed": completed,
                      "workers": args.workers, "docker_workers": args.docker_workers,
                      "case_batch_seconds": round(time.monotonic() - task_started, 3)})
        summarize(args, quiet=True)
        if fatal_error:
            raise SystemExit(f"Endpoint refused requests; completed runs saved. {fatal_error[:300]}")
        if not args.keep_images:
            docker("rmi", "-f", tag, timeout=600)
    summarize(args)


def summarize(args, quiet=False) -> None:
    data = Path(args.data)
    if not (data / "tester_runs.jsonl").exists():
        print("No completed tester runs yet", flush=True)
        return
    recs = [json.loads(l) for l in (data / "tester_runs.jsonl").open()]
    if not recs:
        return
    df = pd.DataFrame(recs)
    cols = ["instance_id", "trajectory_id", "period", "condition", "seed", "resolved", "model", "rounds", "tests",
            "pass_pg", "flags", "locked_in", "broken", "differential", "coverage", "patch_flagged",
            "prompt_tokens", "completion_tokens", "error", "grade_error"]
    df[[c for c in cols if c in df]].to_csv(data / "tester_results.csv", index=False)

    def stats(g: pd.DataFrame) -> dict:
        grade_errors = int(g.grade_error.notna().sum()) if "grade_error" in g else 0
        attempted = len(g)
        if "grade_error" in g:
            g = g[g.grade_error.isna()]
        wrong, right = g[g.resolved == 0], g[g.resolved == 1]
        return {"runs": int(len(g)), "attempted_runs": attempted, "grade_errors": grade_errors,
                "bug_caught_rate": round(float(wrong.patch_flagged.mean()), 3) if len(wrong) else None,
                "false_alarm_rate": round(float(right.patch_flagged.mean()), 3) if len(right) else None,
                "no_test_file": int(g.tests.isna().sum()) if "tests" in g else int(len(g)),
                "mean_tests": round(float(g.tests.mean()), 2) if "tests" in g else None,
                "mean_broken": round(float(g.broken.mean()), 2) if "broken" in g else None,
                "differential_rate": round(float(g.differential.fillna(False).astype(bool).mean()), 3) if "differential" in g else None,
                "mean_coverage": round(float(g.coverage.dropna().mean()), 3) if g.coverage.notna().any() else None,
                "mean_prompt_tokens": int(g.prompt_tokens.mean()) if len(g) else None}

    summary = {"model": sorted(df.model.unique().tolist()),
               "by_condition": {c: stats(g) for c, g in df.groupby("condition")},
               "by_condition_and_period": {f"{c} {p}": stats(g) for (c, p), g in df.groupby(["condition", "period"])}}
    (data / "tester_summary.json").write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(json.dumps(summary["by_condition"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarize"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pilot tasks")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--prompt-size", choices=list(LIMITS), default="full",
                    help="small: keep each call under ~13k input tokens (for Gemma on the free Gemini API)")
    ap.add_argument("--shard", default="1/1", help="i/n: run every n-th task starting at the i-th, to run n processes at once")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="concurrent tester cases within one task image")
    ap.add_argument("--docker-workers", type=int, default=1, help="maximum simultaneous test/context containers")
    ap.add_argument("--allow-code-change", action="store_true", help="record a reviewed source-only upgrade when resuming")
    ap.add_argument("--validate-images", action="store_true", help="check leaks, hidden-test baselines and both agent labels before generation")
    ap.add_argument("--validation-policy", choices=["strict", "terra"], default="strict")
    ap.add_argument("--tasks", help="comma-separated task IDs; retain the full trace pool for condition X")
    ap.add_argument("--restart", action="store_true", help="delete earlier runs and start over")
    args = ap.parse_args()
    if args.workers < 1 or args.docker_workers < 1:
        ap.error("worker counts must be positive")
    run(args) if args.step == "run" else summarize(args)


if __name__ == "__main__":
    main()
