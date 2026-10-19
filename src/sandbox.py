"""Clean task images for the tester's sandbox, and check that the cleaning worked.

The SWE-rebench images carry git history from after each task was created (the pilot found
newer commits in all 20), which can contain the real fix. For each task this makes a cleaned
image whose /testbed has a single commit holding exactly the base files, then checks it:

  git     one commit, one ref, no other .git directories on the filesystem
  leaks   no file anywhere holds many of the lines the human fix or the hidden tests add
          (at least 30% of them, and 2 or more; lines already in the base repo don't count)
  works   the hidden tests still fail on the base code and pass with the human fix

  python src/sandbox.py prepare [--limit N] [--keep-images] [--restart]
      -> cleaned images ttg-clean/<instance_id>, data/sandbox_report.csv

Each tester run then starts its own throwaway container from the cleaned image with
`docker run --rm --network none`, so runs can't see each other or the internet.
Cleaned images are deleted after the checks unless --keep-images.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from pilot import as_list, docker, pull_image, run_tests, test_command, verdict

MIN_NEEDLE = 30  # shorter added lines (`return x`, `)`) are too common to mean anything

# Runs inside the original image: rebuild /testbed/.git with only the base commit.
CLEAN_SH = r"""
set -e
cd /testbed
git config --global --add safe.directory /testbed
date=$(git log -1 --format=%cI HEAD)
git ls-files -z > /tmp/tracked
rm -rf .git
git init -q -b main
git config user.name base && git config user.email base@example.com
xargs -0 git add -f < /tmp/tracked
GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -q -m base
git gc -q --prune=now
rm -f /tmp/tracked
# coverage.py, for the tester's coverage feedback; installed offline from wheels on the host.
(source /opt/conda/bin/activate testbed 2>/dev/null; pip install -q --no-index --find-links /ttg_wheels coverage) \
  >/dev/null 2>&1 || echo "COVERAGE_NOT_INSTALLED"
"""

WHEELS = Path(__file__).resolve().parent.parent / "data" / "wheels"
PYTHONS = ["3.6", "3.7", "3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]


def coverage_wheels() -> Path:
    """Download coverage.py wheels for every Python an image might use (once)."""
    if not any(WHEELS.glob("coverage-*.whl")):
        WHEELS.mkdir(parents=True, exist_ok=True)
        for v in PYTHONS:
            subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:",
                            "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_17_x86_64",
                            "--platform", "manylinux1_x86_64", "--python-version", v, "coverage", "-d", str(WHEELS)],
                           capture_output=True)
    return WHEELS

# Runs inside the cleaned image. /ttg_check/needles holds the lines to look for; the check's own
# files live in /ttg_check and /ttg_scratch, which the filesystem search skips.
CHECK_SH = r"""
cd /testbed
git config --global --add safe.directory /testbed
echo "COMMITS $(git rev-list --all --count)"
echo "REFS $(git for-each-ref | wc -l)"
echo "REFLOG $(git reflog 2>/dev/null | wc -l)"
echo "OTHER_GIT $(find / -xdev -name .git -not -path /testbed/.git -not -path '/proc/*' 2>/dev/null | head -20 | tr '\n' ' ')"
# Lines already in the base repo (moved code, say) don't count as leaks.
mkdir -p /ttg_scratch
grep -rhoIF -f /ttg_check/needles /testbed --exclude-dir=.git 2>/dev/null | sort -u > /ttg_scratch/in_repo
grep -vxF -f /ttg_scratch/in_repo /ttg_check/needles > /ttg_scratch/needles || true
echo "NEEDLES $(grep -c . /ttg_scratch/needles)"
if [ -s /ttg_scratch/needles ]; then
  grep -rlIF -f /ttg_scratch/needles / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev \
    --exclude-dir=ttg_check --exclude-dir=ttg_scratch 2>/dev/null > /ttg_scratch/hits
  # A copy of the fix holds many of its lines; a stray shared line (`with pytest.raises(TypeError):`)
  # in some other package doesn't. Count how many distinct lines each file holds.
  cat > /ttg_scratch/count.py <<'PY'
import math
needles = [l.rstrip("\n") for l in open("/ttg_scratch/needles") if l.strip()]
need = min(len(needles), max(2, math.ceil(0.3 * len(needles))))
found = []
for path in open("/ttg_scratch/hits").read().split("\n"):
    try:
        text = open(path, errors="replace").read()
    except Exception:
        continue
    n = sum(x in text for x in needles)
    found.append((n, path))
found.sort(reverse=True)
print("LEAK_FILES " + " ".join(f"{p}:{n}/{len(needles)}" for n, p in found if n >= need))
print("TOP_MATCH " + (f"{found[0][1]}:{found[0][0]}/{len(needles)}" if found else ""))
PY
  for py in /opt/conda/bin/python python3 python; do command -v $py >/dev/null && { $py /ttg_scratch/count.py; break; }; done
else
  echo "LEAK_FILES "
fi
"""


def clean_tag(instance_id: str) -> str:
    return "ttg-clean/" + re.sub(r"[^a-z0-9._-]", "-", instance_id.lower())


def added_lines(patch: str) -> list[str]:
    lines = (l[1:].strip() for l in (patch or "").splitlines() if l.startswith("+") and not l.startswith("+++"))
    return sorted({l for l in lines if len(l) >= MIN_NEEDLE and sum(c.isalnum() for c in l) >= 15})


def clean(image: str, tag: str) -> str:
    """Make the cleaned image; return an error message, or "" on success."""
    name = "ttg-clean-" + tag.split("/")[1]
    docker("rm", "-f", name)
    run = docker("run", "--name", name, "--network", "none", "-v", f"{coverage_wheels()}:/ttg_wheels:ro",
                 image, "bash", "-c", CLEAN_SH, timeout=1800)
    if run.returncode != 0:
        docker("rm", "-f", name)
        return "clean_failed: " + (run.stderr or run.stdout).strip()[-300:]
    commit = docker("commit", name, tag, timeout=1800)
    if commit.returncode != 0:  # some images trip a Docker layer-diff bug; flattening works
        commit = flatten(name, image, tag)
    docker("rm", "-f", name)
    return "" if commit.returncode == 0 else "commit_failed: " + commit.stderr.strip()[-300:]


def flatten(container_name: str, image: str, tag: str) -> subprocess.CompletedProcess:
    """`docker export | docker import`, keeping the original image's env, workdir and command."""
    cfg = json.loads(docker("inspect", image, "--format", "{{json .Config}}").stdout or "{}")
    changes = [f"ENV {e.split('=', 1)[0]}={json.dumps(e.split('=', 1)[1])}" for e in cfg.get("Env") or [] if "=" in e]
    if cfg.get("WorkingDir"):
        changes.append(f"WORKDIR {cfg['WorkingDir']}")
    if cfg.get("Cmd"):
        changes.append(f"CMD {json.dumps(cfg['Cmd'])}")
    args = [a for c in changes for a in ("--change", c)]
    export = subprocess.Popen(["docker", "export", container_name], stdout=subprocess.PIPE)
    out = subprocess.run(["docker", "import", *args, "-", tag], stdin=export.stdout, capture_output=True, text=True)
    export.wait()
    return out


def check(tag: str, needles: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "needles").write_text("\n".join(needles) + "\n")
        out = docker("run", "--rm", "--network", "none", "-v", f"{work}:/ttg_check:ro", tag,
                     "bash", "-c", CHECK_SH, timeout=3600).stdout
    vals = dict((line.split(" ", 1) + [""])[:2] for line in out.strip().splitlines())
    res = {"commits": int(vals.get("COMMITS") or -1), "refs": int(vals.get("REFS") or -1),
           "reflog": int(vals.get("REFLOG") or -1), "other_git": vals.get("OTHER_GIT", "").strip(),
           "needles": int(vals.get("NEEDLES") or 0), "leak_files": vals.get("LEAK_FILES", "").strip(),
           "top_match": vals.get("TOP_MATCH", "").strip()}
    res["git_ok"] = res["commits"] == 1 and res["refs"] == 1 and not res["other_git"]
    res["no_leaks"] = not res["leak_files"]
    return res


def prepare(args) -> None:
    data = Path(args.data)
    rows = pd.read_json(data / "pilot_tasks.jsonl", lines=True, dtype=False)
    tasks = rows.drop_duplicates("instance_id")
    if args.limit:
        tasks = tasks.head(args.limit)
    out_csv = data / "sandbox_report.csv"
    records = [] if args.restart or not out_csv.exists() else pd.read_csv(out_csv).to_dict("records")
    records = [r for r in records if r.get("ok") == True]  # noqa: E712  # retry the tasks that failed
    done = {r["instance_id"] for r in records}

    for t in tasks.itertuples():
        if t.instance_id in done:
            continue
        tag = clean_tag(t.instance_id)
        rec = {"instance_id": t.instance_id, "period": t.period, "clean_image": tag}
        print(f"[{t.instance_id}] pulling", flush=True)
        if pull_image(t.docker_image).returncode != 0:
            records.append({**rec, "error": "pull_failed"})
            continue
        print(f"[{t.instance_id}] cleaning", flush=True)
        rec["error"] = clean(t.docker_image, tag)
        docker("rmi", "-f", t.docker_image, timeout=600)  # the cleaned image keeps the layers it needs
        if not rec["error"]:
            print(f"[{t.instance_id}] checking", flush=True)
            rec.update(check(tag, added_lines(t.patch) + added_lines(t.test_patch)))
            f2p, p2p = as_list(t.FAIL_TO_PASS), as_list(t.PASS_TO_PASS)
            cmd = test_command(t.install_config)
            base = verdict(run_tests(tag, "", t.test_patch, cmd, f2p + p2p), f2p, p2p)
            golden = verdict(run_tests(tag, t.patch, t.test_patch, cmd, f2p + p2p), f2p, p2p)
            rec.update(base_f2p=base["f2p_passed"], golden_passes=golden["passes"],
                       works=bool(base["f2p_passed"] and base["f2p_passed"].startswith("0/") and golden["passes"]))
            rec["ok"] = rec["git_ok"] and rec["no_leaks"] and rec["works"]
        records.append(rec)
        pd.DataFrame(records).to_csv(out_csv, index=False)
        if not args.keep_images:
            docker("rmi", "-f", tag, timeout=600)
            docker("system", "prune", "-f", timeout=600)

    res = pd.DataFrame(records)
    print(f"{int(res.get('ok', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}/{len(res)} tasks cleaned and checked OK")
    cols = [c for c in ["instance_id", "error", "commits", "other_git", "leak_files", "base_f2p", "golden_passes", "ok"]
            if c in res]
    print(res[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pilot tasks")
    ap.add_argument("--keep-images", action="store_true", help="keep the cleaned images")
    ap.add_argument("--restart", action="store_true", help="ignore earlier results and start over")
    args = ap.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
