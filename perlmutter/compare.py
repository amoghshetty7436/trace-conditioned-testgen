"""Go/no-go for the gate pilot: does gpt-oss-120b find the bugs the Azure model found?

  python perlmutter/compare.py        # after perlmutter/gate.sbatch, from the repo root

On the 8 gate tasks the Azure model (gpt-5.6-terra) found 9 real bugs and flash-lite 2. Seven of them
are "hard": flash-lite missed them. A patch counts as found when any condition flags it.
"""

import json
from pathlib import Path

DATA = Path("data")
RUNS = {"gpt-oss-120b": "tester_runs_gpt_oss_120b.jsonl", "azure terra": "tester_runs.jsonl",
        "flash-lite": "tester_runs_flash_lite.jsonl"}
# (task, resolved): resolved=1 entries are real bugs in patches the benchmark calls correct (docs/pilot_azure.md).
HARD = [("asottile__add-trailing-comma-20", 0), ("streamlink__streamlink-2765", 0), ("pypa__setuptools_scm-803", 0),
        ("lzakharov__csv2md-22", 0), ("scrapy__parsel-308", 0),
        ("asottile__add-trailing-comma-20", 1), ("probabl-ai__skore-914", 1)]
EASY = [("reagento__dishka-299", 0), ("radiocosmology__alpenhorn-226", 0)]
TASKS = {t for t, _ in HARD + EASY}


def load(name: str) -> list[dict]:
    path = DATA / name
    return [r for r in map(json.loads, path.open()) if r["instance_id"] in TASKS] if path.exists() else []


def main() -> None:
    runs = {m: load(f) for m, f in RUNS.items()}
    if not runs["gpt-oss-120b"]:
        raise SystemExit("no gpt-oss-120b runs yet: data/tester_runs_gpt_oss_120b.jsonl is missing")
    print(f"{'bug (task, patch)':48} " + " ".join(f"{m:>14}" for m in runs))
    found = {m: 0 for m in runs}
    for key in HARD + EASY:
        cells = []
        for m, rs in runs.items():
            conds = sorted({r["condition"] for r in rs if (r["instance_id"], r["resolved"]) == key and r.get("patch_flagged")})
            ran = any((r["instance_id"], r["resolved"]) == key for r in rs)
            found[m] += bool(conds) and key in HARD
            cells.append(",".join(conds) if conds else ("-" if ran else "not run"))
        label = f"{key[0].split('__')[1]} ({'correct' if key[1] else 'wrong'} patch){' easy' if key in EASY else ''}"
        print(f"{label:48} " + " ".join(f"{c:>14}" for c in cells))
    print("\nhard bugs found (of 7): " + ", ".join(f"{m} {n}" for m, n in found.items()))
    other = [r for r in runs["gpt-oss-120b"] if r["resolved"] == 1 and r.get("patch_flagged")
             and r["instance_id"] not in ("asottile__add-trailing-comma-20", "probabl-ai__skore-914")]
    print(f"gpt-oss-120b flags on other correct patches (possible false alarms, check by hand): {len(other)}")
    n = found["gpt-oss-120b"]
    print("\nverdict:", "GO: worth the full run" if n >= 3 else
          "BORDERLINE: read the flagged tests before deciding" if n >= 1 else "STOP: it finds no more than flash-lite")


if __name__ == "__main__":
    main()
