# J02 — Complementary candidate retrieval

Prerequisite: J01 passes. Review S09/S10/S11/T02/T05/I01 full methods and official implementations before selecting an extension. Preserve the strongest specialized dense retriever as the control.

Using the allowed development panel, compare SkillRet dense retrieval with lexical, field-aware lexical/dense, and one complementary generic embedding channel. Reproduce field-aware TF-IDF/dense before calling BM25F/learned fusion an exact paper reproduction. Test basic rank fusion and a small learned/weighted variant; tune only on declared development data.

Save immutable top-20/50/100 candidate artifacts and per-query channel overlap, candidate recall/completeness and marginal coverage. Diagnose whether additional channels rescue misses or merely displace specialized-model hits. A method that loses useful hits is retained as a negative result.

Only if coverage remains limiting, test one extension at a time: SkillSight; supported pseudo-query views; SPLADE; or late interaction. Review model/encoder compatibility before late chunking. Generated views may use source skill contents but not final queries/qrels; log generation costs and provenance. Never change source skills or train on test-derived pseudo-pairs without a separate transductive label.

Acceptance: at most two candidate generators promoted, fixed candidate files for J03, controlled ablations and costs, and an explanation of why greater K helps or does not help. Do not proceed to expensive model training simply because hybrid retrieval was proposed.

Also screen the released SkillRouter-Embedding-0.6B and SkillReason-embedding-0.6b specialists. Keep their native input adapters separate from the equal-evidence comparison. SkillReason's published SkillRet scores use the older 6,660/4,997 pool; never copy those numbers into the corrected benchmark leaderboard. Only promote its 4B version through a new approved configuration. [S02,S37,E16,E18]

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
