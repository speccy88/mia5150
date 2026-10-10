# Frozen Jev context preparation

Truncation is required by the frozen conservative byte estimator, not proven native Jev overflow. No provider acceptance or accuracy effect is measured.

The original blocked plans remain unchanged. Every selected query retains its original dense top20 pool. Smoke requests are an exact subset of the pilot requests, permitting cache reuse.

| Scope | Queries prepared | Full-text pairs | Truncated pairs | Queries with truncation | Requests | Body bytes retained |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 10/10 | 195 | 5 | 4 | 40 | 97.59% |
| pilot | 200/200 | 3857 | 143 | 96 | 779 | 95.72% |

Batching is greedy in frozen dense-rank order. Each original candidate is tested alone first. Full text is unchanged when admitted. The optional policy otherwise retains the longest Unicode-codepoint prefix of skill_md that passes both canonical-JSON byte checks. Query, name, description, rubric, IDs and ranks are unchanged. No relevance label chooses a cut.

The study interprets documented 64k/32k as conservative local 64,000/32,000 thresholds. It adds 2,048 framing units and reserves 2,048 output units. These are study choices. Exact native Jev token counts and boundary behavior remain unknown. See [primary-source audit](../CONTEXT_SOURCE_AUDIT.md).

Per-pair source/prepared hashes, counts, retained fractions and reasons are saved in context_candidate_audit.csv and the frozen manifests. Gold-relevance flags appear only in this after-the-fact diagnostic CSV; they are absent from requests and preparation decisions.

Any later quality result must distinguish full-text and text-reduced query subsets while keeping all selected queries in the overall denominator. No Jev accuracy, latency, usage, or cost has been measured. No credential or endpoint was accessed.
