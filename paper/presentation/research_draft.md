# Research presentation draft

##1. Discovery question
Five methods, four catalog sizes, one common selector. Separate retrieval coverage from final selected sets.

##2. Frozen primary design
200 exposed queries. Seed 42. Original size grid. Pinned metadata and model hashes. Gold-conditioned small catalogs.

##3. Existing evidence and scope drift
Historical16 studies remain separate. Original/revised proposals and 80 hosted cells have a versioned crosswalk.

##4. Measured selection
Use report/selection-heatmap.png and retain planned/valid denominators.

##5. Retrieval versus selection
At 6006, hybrid Recall@5 is57.17% while exact sets are11/200. BM25 yields9/200. This tests a weak0.6B deployment.

##6. Context and latency
Use input-growth.png and latency.png. Distinguish preflight tokens, successful serving, startup and failure latency.

##7. Paired uncertainty
Use paired-differences.png. The full-catalog hybrid difference has an interval spanning zero.

##8. Scenario and model sensitivity
Keep balanced 24-query and generated12-probe results separate. Show actual execution status; do not invent pending values.

##9. Limitations
Exposed synthetic queries, incomplete qrels, small sensitivity panel, no human adjudication or ordering/execution success.

##10. Reproduction and next evidence
Source snapshots, raw ledgers, tests and regeneration commands. Hosted stage obeys explicit caps and credential availability. Team roles and speaking allocation await confirmation.

## Hosted follow-up diagnostic

The separate Qwen/SiliconFlow diagnostic uses 24 exposed balanced queries at 100 skills. Exact sets: listing 5/24; bm25 13/24; vector 7/24; domain vector 4/24; hybrid 10/24. Known reported charges including successful canaries and routing total US$0.02411946, all from Qwen in this completed stage. Other failed provider requests retain US$0.19 UNKNOWN liability holds. Actual total cost remains unknown. There are 150 cumulative generation attempts. These results are not official-prevalence estimates, a causal local-versus-hosted model comparison, native task execution, or completion of the original hosted matrix. The later Mistral/Parasail BF16 arm returned an upstream429 at its first canary, with no usage or known charge, and attempted no benchmark cases. Original planned selectors:360, completed120; later added selectors:120, completed0. Unattempted cells remain unmeasured. Only Qwen provides successful benchmark quality; multi-model comparison remains incomplete. No further generation is planned.
