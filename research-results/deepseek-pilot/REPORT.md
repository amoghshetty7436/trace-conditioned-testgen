# DeepSeek pilot

Job `60629144`: **COMPLETED**. 240/240 cases saved; 240 paired with Terra.

Amogh's unchanged commit `44ec6b20afeb790a8c5c625497c89bf0f163967d`. CPU-only, seed 0, 20 pilot tasks.
Azure deployment `DeepSeek-V4-Flash`, requested reasoning high, maximum output 65,536 tokens.

| Condition | DeepSeek catches | Terra catches | DeepSeek false alarms | Terra false alarms |
|---|---:|---:|---:|---:|
| C1 | 4/20 | 5/20 | 1/20 | 1/20 |
| C2 | 4/20 | 6/20 | 0/20 | 1/20 |
| C3 | 2/20 | 8/20 | 0/20 | 3/20 |
| C4 | 6/20 | 6/20 | 0/20 | 3/20 |
| C5 | 5/20 | 6/20 | 0/20 | 2/20 |
| X | 5/20 | 5/20 | 0/20 | 1/20 |

Rates above use only matching saved cases and the original harness's scoring.
Validation exclusions: 0. No parsed tests: 2.
All 240 unique cases match the Terra baseline case identifiers. All 20 task validations passed; input and archived source hashes match the manifest, and the recorded final remote git status is clean.

**Do not start the full study until the user reviews these results and explicitly approves.**

Final audit: job completed in 3,415 seconds (56 minutes 55 seconds), using one CPU node: 0.9486 CPU node-hours. The allocation was released. Approximately 9.05 of the original 10 node-hours remain assuming no other CPU jobs; the captured Iris charge had not yet updated.

Across six conditions, DeepSeek flagged 26/120 incorrect-patch cases (21.7%) versus Terra's 36/120 (30.0%). It flagged 1/120 correct-patch cases (0.8%) versus Terra's 11/120 (9.2%). These are repeated condition-level cases from 20 tasks, not 120 independent tasks. This pilot shows fewer catches and fewer false alarms; it does not establish overall superiority.

Diagnostic caveats, without changing upstream scores:

- One DeepSeek C3 catch for `radiocosmology__alpenhorn-226` comes from five skipped tests with missing parsed Pa statuses, rather than explicit failures that pass on Pg. Excluding that ambiguous case diagnostically gives 25/120 catches. The primary table preserves Amogh's original score.
- Two C3 generations (`dalito__ucumvert-25`, `roboflow__supervision-1739`) reached the 65,536-token completion cap without producing test code. They remain included in the denominators.
- All generated tests were classified broken in 13 DeepSeek cases versus 12 Terra cases. No rows contain an explicit error field value.
- Recorded test-generation usage: 10,493,368 prompt tokens and 2,166,917 completion tokens. Upstream summary generation does not record usage, so these are not total billable usage.
- Model and reasoning settings differ; this comparison does not imply equal inference compute.

Raw outputs, execution logs, inputs, model configuration, exact source archive, accounting and Terra baseline are saved in this directory and were copied from Perlmutter. The final local audit is in `diagnostics.json` and is reproducible with `audit_pilot.py`. No full-study job was submitted.
