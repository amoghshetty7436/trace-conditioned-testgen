# Pilot results: free model (Gemini Flash-Lite), 19 tasks

One seed per run. Each task has one wrong agent patch (resolved=0) and one correct one (resolved=1), and every
condition runs on both, so each condition has 38 runs: 19 on wrong patches, 19 on correct ones.
`encode__starlette-1240` is missing because its agent patches did not apply in the sandbox (still to look into).
Raw data: `data/tester_runs.jsonl`, `data/tester_results.csv`, `data/tester_summary.json`.

## By condition

| Condition | Wrong patches caught (of 19) | False alarms (of 19) | Wrong-patch runs with locked-in tests (of 19) | Tests per run | Broken tests per run | Runs with every test broken (of 38) | Coverage of new lines |
|---|---|---|---|---|---|---|---|
| C1 issue + diff         | 2 | 0 | 11 | 2.5 | 0.45 | 5 | 0.33 |
| C2 + related tests      | 2 | 0 |  8 | 2.6 | 0.21 | 5 | 0.35 |
| C3 trace summary        | 3 | 0 | 10 | 3.6 | 0.55 | 3 | 0.37 |
| C4 structured trace     | 3 | 0 |  9 | 3.1 | 0.24 | 2 | 0.36 |
| C5 raw trace            | 1 | 0 | 11 | 3.8 | 0.16 | 1 | 0.40 |
| X  another task's trace | 4 | 0 |  8 | 2.6 | 0.45 | 4 | 0.37 |

- Caught: at least one generated test fails on the agent patch and passes on the human fix.
- False alarm: the same, on a patch the hidden tests accept.
- Locked-in: a test passes on the agent patch and fails on the human fix (it encodes the agent's behavior).
- Broken: fails on both the agent patch and the human fix. Coverage was recorded on 30 of the 38 runs per condition.

## What it says

1. **No condition is clearly better at catching bugs.** All counts are between 1 and 4 out of 19, on one seed.
   X, which gets an unrelated trace, caught the most. So these catches come from the issue and the diff, not from
   the trace.
2. **Caught bugs cluster on a few tasks:** alpenhorn-226 (C2, C3, C4, X), dishka-299 (C1 to C4),
   textual-3896 (C1, C3, C5, X), pyjwt-670 (C4, X), supervision-1739 (X).
3. **No false alarms anywhere.** The tester never failed a correct patch in a way the human fix passes.
4. **Locking in the agent's mistake is common:** 8 to 11 of the 19 wrong-patch runs in every condition wrote a
   test that passes on the wrong patch and fails on the human fix. borgstore-88 is the clearest case, because the
   issue text points to the agent's reading.
5. **The raw trace (C5) gives the cleanest tests:** the most tests per run (3.8), the fewest broken ones (0.16 per run, and only 1 run where
   every test was broken) and the highest coverage (0.40). It also costs about 2.5x the prompt tokens of C1.

## Is it worth more compute?

This small free model rarely catches bugs in any condition, so the pilot can't separate the conditions on
catch rate. Before reading anything into the condition differences, the next run needs a stronger tester model,
several seeds per run, and the full task set. The harness, sandbox and grading all worked end to end on 19 of 20
tasks.
