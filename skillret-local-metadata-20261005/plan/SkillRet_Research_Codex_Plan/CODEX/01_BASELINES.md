# J01 — Baseline reproduction

Prerequisite: J00 approved and metric/exposure gates resolved. Work only within allocated hardware and spending limits. Fetch/pin the actual corrected dataset and official code; validate all counts, IDs, qrels joins, representation and train/test boundaries.

Implement/reuse the exact official BM25, Qwen3-Embedding-0.6B/8B and SKILLRET-Embedding-0.6B/8B adapters. Reproduce the native SKILLRET-Reranker-0.6B on the top 20 from the 8B SkillRet embedder. First use tiny hand-checked fixtures, then the allowed development panel. Compare metric output with official pytrec_eval settings, not an independently invented MAP definition.

Retain existing BGE/CLM/hosted-JeV controls and distinguish their old runs from reruns. Create a separate Track-B name+description view for full-catalog prompt, BM25, vector, hard-domain vector and hybrid. Full-catalog means all entries in the specified library: overflow is explicit, not a hidden retrieval step.

Cache corpus/query embeddings and candidate rankings with code/text/model hashes. Start with exact dense search. Record indexing and serving separately. Time a representative batch on actual hardware, including long-skill cases; estimate the full workload before scaling.

Acceptance: passing metric and adapter tests; actual data/version manifest; comparable baseline outputs; explanation of discrepancies from the published anchor; actual RAM/VRAM/token/time data; proposed J02/J03 shortlist with caps. No inference success claim unless the model really ran.

## Shared requirements

Apply MASTER_HANDOFF.md and BENCHMARK_CONTRACT.md. Preserve existing code/results and pin all new inputs. Respect explicit resource approvals; null/absent budget means no paid operations. Save exact commands, versions, input hashes and raw per-query results. Any artifact handoff is one ZIP with START_HERE, including every related patch, test, log and report. No claim of model quality is valid merely because an adapter smoke test passed.
