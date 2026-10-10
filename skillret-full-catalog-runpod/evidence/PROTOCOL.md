# Frozen scientific protocol and implementation checks

## Primary experiment

One official SKILLRET test catalog and all its evaluation queries, one canonical description-only text representation, five scorers:

| ID | Input to scoring | Adaptation using SKILLRET labels |
|---|---|---|
| bm25_full | Lexical query/document statistics | None |
| bge_m3_dense_full | BGE-M3 dense vectors, normalized | None |
| qwen3_raw_full | Pre-head Qwen3-8B last-token vectors, normalized | None |
| clm_pretrained_full | Same Qwen vectors through original heads | None in this experiment |
| clm_skillret_full | Same Qwen vectors through adapted heads | Train partition only |

All methods exhaustively score the same catalog. There is no generative selector after retrieval. The controlled head ablation is the last three rows; differences between BGE/BM25 and Qwen include model size, pretraining and architecture, so do not attribute them exclusively to one training objective.

The published CLM heads have already been trained. The baseline is not randomly initialized. Exact output/input dimensions must be read from the pinned checkpoint. The currently published implementation uses a 4096-dimensional encoder representation and normally 512-dimensional projected representation.

## Input and normalization contract

Skill: `f"{name}: {description}"`; query: dataset `query` only. Stable opaque IDs remain separate from text. No gold-derived annotations appear in input.

Pre-head does not mean unnormalized: the official CLM Embedder normalizes its pooled Qwen vectors before passing them to nonlinear heads. Cache these reference-compatible vectors. The raw-Qwen row is cosine on this pre-head space. Each CLM head output is normalized again; multiply by the original positive temperature scale only when reproducing logits/probabilities. A positive global scale cannot change exact rank ordering.

The pretrained and fine-tuned arms must reuse identical pre-head arrays and exact row/ID mappings. Changing the head checkpoint requires recomputing both state and skill projections. Checkpoint-specific cache keys prevent old projected vectors from being mixed with new heads.

## Multi-positive contrastive adaptation

Let z_q and z_s be normalized projected query/skill vectors. Let L_ij = exp(scale) * z_q[i] dot z_s[j], with the original model's scale handling. Define P_ij = 1 exactly when training qrels mark skill j relevant to query i. Candidate IDs are unique within a batch; queries are unique within a batch.

An explicit suitable loss is uniform-positive bidirectional cross entropy:

- Query direction: each row's target is P_ij divided by the number of positive skills in that row. Average negative log-softmax over its positives.
- Skill direction: for each column containing positives, target queries are P_ij divided by that column's positive count. Average negative log-softmax over relevant queries.
- Average the two direction losses. Columns with no positive query in the batch are excluded ONLY from the skill-to-query loss. They still supply negatives in the query-to-skill denominator.

This reduces to symmetric InfoNCE when P is an identity matrix. It is a documented multi-gold adaptation, not a claim that the upstream trajectory trainer already implements full SKILLRET qrel masking. Masking only same-query trajectory groups does not prevent a shared relevant skill appearing as a false negative across different queries.

The supplied helper implements this definition. Reusing it does not validate the surrounding training pipeline; Codex must still test qrel construction, batches, gradient flow, encoder freezing, epoch-0 parity and checkpoint loading.

## Split and selection

Validation comes from training queries, grouped by original query family when available. Test has no role in hyperparameter choice, representation choice, negative mining or checkpoint selection. A query-disjoint validation split may share skill IDs with training; say so. Final official train/test skill-ID disjointness must be checked, not assumed to remove all semantic overlap.

Choose the best trained epoch using full-training-catalog validation Recall@5 (MRR tiebreak), not the upstream within-task trajectory metric. Keep epoch 0 as the pretrained control. If all trained epochs are worse, still report the best trained epoch's outcome honestly and retain the baseline unchanged.

## Metrics

For query q, G is its nonempty set of relevant skill IDs and R is its unique full ranking.

- Hit@1: 1 if R[0] is in G, else 0.
- Recall@k: number of distinct positives among R[:k], divided by |G|.
- MRR: reciprocal of the first relevant skill's 1-based position in the FULL ranking; 0 if none.
- NDCG@5: binary relevance DCG with discount 1/log2(rank+1), divided by the ideal DCG containing min(5, |G|) positives.
- Completeness@5: 1 only if all of G is included in R[:5], else 0.

Aggregate by macro mean over query IDs. Report single-gold and multi-gold slices if useful. For three positives and only one at rank 1, Hit@1 is 1 but Recall@1 is 1/3. An empty or failed result is never silently removed from the denominator; distinguish infrastructure failure from a completed low-quality ranking.

Reference example: G={A,C}, R=[X,A,Y,C]. Hit@1=0, Recall@1=0, Recall@5=1, MRR=1/2, Completeness@5=1. The NDCG denominator has two ideal positives. Duplicate predictions/IDs are input errors, not bonus credit.

Use exact scores with stable skill-ID ties. Do not use argpartition alone where arbitrary tie ordering can change BM25 metrics. Store enough scores/IDs to reproduce ties identically.

## Timing

Three different measurements must not be confused:

1. Encoder preprocessing throughput, over batched inputs.
2. Online text-to-ranking latency: fresh query encoding, optional head, and catalog search with model/index already loaded.
3. Offline replay latency: a query vector already exists; only projection/scoring/ranking runs.

Training and evaluation can reuse cached query vectors for scientific efficiency, but that does not eliminate query encoding cost in deployment. A combined time from separately observed components is an estimate, not a directly observed p95. Small timing samples and differing CPU/GPU devices must be disclosed.

## Optional catalog-size sweep

Main result is all test queries against N_test skills. Optional smaller sizes include all positives and seeded nested distractors. Catalog size is conditioned on gold presence, unlike arbitrary production libraries; state this selection protocol. Gold labels are used only by the evaluation-set constructor, never by rankers.

A fixed-index score-mask sweep measures extra distractor competition. It does NOT recreate a smaller BM25 corpus: IDF and average document length would change. Either label the cheaper fixed-index design or implement an additional separately named reindexed design. Neither cached-score masking nor offline reused vectors establish end-to-end latency scaling.

## Reproducibility and conclusions

Bootstrap query groups for uncertainty, paired across methods; do not treat three nested-catalog seeds as three independent datasets or claim a bootstrap measures training-seed uncertainty. Synthetic query provenance and incomplete qrels are limitations. A comparison supports statements about this dataset, representation and run—not that one architecture is universally superior.

Archive primary scores/rankings, all generated vector arrays and ordered IDs, exact text, labels, selected training groups, checkpoints, code/configs/versions, tests and cost records. Never report a paper's published result as a result of this experiment.
