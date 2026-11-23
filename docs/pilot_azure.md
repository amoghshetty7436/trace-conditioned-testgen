# Pilot results: stronger model (Azure OpenAI), 20 tasks

Same setup as [the free-model pilot](pilot_flash_lite.md): one run per condition, on one wrong agent patch
(resolved=0) and one correct one (resolved=1) per task, which makes 240 runs. Raw data: `data/tester_runs.jsonl`,
`data/tester_results.csv`, `data/tester_summary.json`. The free model's numbers are in brackets. They cover 19 tasks,
because starlette-1240 was skipped there. That was a harness bug (a Python 3.7-only call inside a Python 3.6
sandbox), fixed for this run.

## By condition

| Condition | Wrong patches caught (of 20) | False alarms (of 20) | Wrong-patch runs with locked-in tests (of 20) | Tests per run | Broken tests per run | Coverage of new lines | Prompt tokens |
|---|---|---|---|---|---|---|---|
| C1 issue + diff         | 5 (2) | 1 (0) | 8 (11) | 4.0 (2.5) | 0.45 (0.45) | 0.37 (0.33) | 18k |
| C2 + related tests      | 6 (2) | 1 (0) | 8 (8)  | 3.9 (2.6) | 0.45 (0.21) | 0.38 (0.35) | 28k |
| C3 trace summary        | 8 (3) | 3 (0) | 7 (10) | 4.2 (3.6) | 0.45 (0.55) | 0.37 (0.37) | 30k |
| C4 structured trace     | 6 (3) | 3 (0) | 7 (9)  | 4.0 (3.1) | 0.35 (0.24) | 0.37 (0.36) | 35k |
| C5 raw trace            | 6 (1) | 2 (0) | 5 (11) | 3.7 (3.8) | 0.35 (0.16) | 0.41 (0.40) | 55k |
| X  another task's trace | 5 (4) | 1 (0) | 7 (8)  | 3.6 (2.6) | 0.28 (0.45) | 0.37 (0.37) | 49k |

The terms mean the same as in the free-model write-up. Caught: a test fails on the agent patch and passes on the
human fix. False alarm: the same, on a patch the hidden tests accept. Locked-in: a test passes on the agent patch
and fails on the human fix.

## The false alarms, checked by hand

All 11 false alarms come from three tasks:

- **add-trailing-comma-20** (C1, C3, C4, C5, X): **real bug**. The "correct" agent patch only fixes calls whose
  callee is a plain name. The flagged tests use `callbacks[0]('long' 'literal')` and `factory()('long' 'literal')`,
  which still need two passes on the agent's patch and one on the human fix. The hidden tests only check `f(...)`.
- **skore-914** (C3, C4, C5): **real bug**, confirmed by rerunning the C5 test. When runs with different scores
  are combined, the agent's patch numbers each fold as its own run in the chart (0 to 5) instead of grouping folds
  by run (0, 0, 0, 1, 1, 1). The hidden tests don't check the run numbers.
- **gspread-1229** (C2, C3, C4): **true false alarm**. The tests insist that `None` options are passed through to
  the API instead of dropped. The issue says this should make no difference to users.

Counting the first two as catches gives:

| Condition | Real bugs found | True false alarms |
|---|---|---|
| C1 | 6 | 0 |
| C2 | 6 | 1 |
| C3 | **10** | 1 |
| C4 | 8 | 1 |
| C5 | 8 | 0 |
| X  | 6 | 0 |

## What it says

1. **The stronger model catches about twice as many bugs** as the free one (5 to 8 per condition, against 1 to 4).
2. **The trace conditions now come out ahead.** Counting the bugs found on "correct" patches, C3 to C5 find 8 to 10
   real bugs, against 6 for C1, C2 and X. C5 also locks in the agent's mistake least often (5, against 8 for C1).
   This is the direction the study predicts, but each gap is only 2 to 4 tasks, from one run per condition.
3. **The tester also finds bugs that the benchmark's hidden tests miss**, in patches the benchmark calls
   correct. Two of the 20 "correct" patches have a real bug that some condition found.
4. **Locking in the agent's mistake is still common** (5 to 8 of 20 runs). Borgstore-88 is locked in by all six
   conditions again, because the issue text points to the agent's reading.

## Next step

To tell whether the gaps are real, run several seeds per condition, on more tasks. The harness now does both
(`--seeds N`, and `--shard i/n` to run several processes at once). On this model a run costs about 18k to 55k prompt
tokens, depending on the condition, and 20 tasks take about 1.5 hours with 3 processes.
