# What was recovered from Fred's CLM experiment

Source located in Fred's Library: `model_adaptation_6_lessons_CLM8B_TTM.md`, created October 1, 2026. Relevant snippets record the earlier experiment. Conversation retrieval also supplied method details. **The original completed raw training archive and the exact split-ID manifest were not recovered here; this is not an independent re-audit of those results.**

Recorded test Recall@5, 4,392 queries over the 6,006-skill test catalog:

| Historical method | Reported Recall@5 |
|---|---:|
| Frozen raw Qwen embeddings | .0132 |
| Original CLM heads | .0582 |
| SkillRet-adapted CLM heads | .3174 |
| BGE-M3 | .5545 |

The notes report 56,950 train queries / 6,309 validation queries, query-family holdout, validation Recall@5 with MRR tiebreak, and epoch 10 selected. They describe frozen Qwen3-8B last-token embeddings, `name: description` skill text, and training the small query/action projection heads. This was head-only adaptation, not a Qwen LoRA.

What carries over:

1. **BGE-M3 must stay in the comparison.** CLM adaptation improved its own baseline by 25.92 percentage points, but remained below BGE-M3. A trained system need not be the best system.
2. **Reuse a verified split, not remembered counts.** The new scripts accept the old split-ID manifest when Hermes recovers it. Without it, create a new split and disclose it.
3. **Same representation first.** Start with metadata-only text; full SKILL.md is a separate experiment and will be more expensive.
4. **Multi-positive labels stay multi-positive.** Do not expand the labels into independent contradictory one-positive training examples.
5. **Preserve baselines and raw predictions.** Freeze catalog/order/tokenization, use validation for selection, and score the test only after the procedure is chosen.
6. **Learn from Charlie's grader audit.** Real pytest execution, explicit test counts, known-bad controls, no claim of functional failure merely because a wrapper metric said so.

What cannot be reused mechanically:

- Cached CLM query/action vectors cannot replace token-level JevK5 forwards. LoRA changes intermediate representations; gradients must flow through the adapted network.
- The CLM optimizer, learning rate and very cheap head-training runtime are not automatically appropriate for a 4B adapter experiment.
- Head-only CLM compares each state to a full cached catalog. This package reranks a BGE shortlist. The test catalog may be the same, but they are different systems with different retrieval ceilings and costs.
- The separate 200-query BM25-top20 CLM inference trial is not the later full-catalog head-adaptation result. Do not merge these histories.

Desired new result: not "LoRA loss decreased", but "BGE + adapted JevK5 improves held-out skill retrieval over both BGE alone and the same unadapted reranker at an acceptable measured cost."
