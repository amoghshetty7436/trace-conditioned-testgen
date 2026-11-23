# Experiment results

Kimi and DeepSeek results, validation checks, configuration, analysis and extracted reasoning responses. The original Amogh source commit and model configuration are recorded in each manifest. Kimi pilot and study reasoning traces are compressed JSONL, keyed by API attempt.

Original Kimi pilot and main-run API request and SSE streams are archived under `raw-traces/`, together with the original pilot and full-study coding-agent trajectories. The archive manifest lists checksums and attempt counts; each tarball contains complete attempt directories. Other container inputs, archived source copies, and unrelated prototype output remain in the local ignored `results/` directory. The pilot X condition uses a smaller trace pool than the full study; do not pool them without qualification.
