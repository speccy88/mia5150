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
