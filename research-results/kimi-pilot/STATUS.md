# Kimi Azure pilot

Prepared for the user-authorized 20-task pilot only. Full 169-task study remains gated on review.

Amogh source commit: `44ec6b20afeb790a8c5c625497c89bf0f163967d`, unchanged.
Azure deployment: `Kimi-K3.1-Pro`; smoke responses identify `Kimi-K3.1`.
High reasoning; 65,536 output-token cap; seed 0; six conditions; 240 planned cases.
One Perlmutter CPU node, eight shards, two-hour hard cap, no automatic resubmission.

An external recorder preserves request/response bodies and saves reasoning text. No reasoning is injected into prompts or feedback. Source, inputs, configuration, raw results, API streams and accounting are archived. Credentials remain outside the artifact directory.
