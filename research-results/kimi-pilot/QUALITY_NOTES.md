# Early Kimi pilot observations

The Azure deployment accepts the requested high reasoning setting, 65,536 output-token cap, streaming, temperature 0.2 and seed 0. A two-turn smoke test and a separate smoke through Amogh's unmodified client succeeded. The provider returns the model name `Kimi-K3.1` for requests to deployment `Kimi-K3.1-Pro`; the immutable underlying model revision has not been independently verified.

Two early cases returned tool-call markup in ordinary `content` instead of fenced Python, despite the request not specifying tools. Both ended with `finish_reason=stop`, not an output-token limit. Amogh's parser correctly found no Python test file; these are retained as generation failures, without retrying under changed prompts or scoring.

- canonical__charmcraft-2042 / C1 / seed 0 / incorrect patch: API attempt `1794785475586187622-4f42e0ea` (111 completion tokens).
- dalito__ucumvert-25 / C2 / seed 0 / incorrect patch: API attempt `1794785477734538412-6b01047a` (273 completion tokens).

Raw replies and reasoning are preserved in each attempt's `response.sse` and `parsed.json`. These observations concern this Azure deployment and harness protocol; they do not establish the model's best possible performance under a different agent protocol.

The upstream harness replays only final assistant content during revisions. This controlled pilot preserves that behavior rather than applying the model provider's recommendation to replay reasoning. No source or prompts were changed during this run.
