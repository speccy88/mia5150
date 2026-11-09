# Source and implementation audit

The experiment starts from repository commit
`9d534137eb1e5c03408f2f8946f45e0825ede114`, fetched from public `main`.
The historical SkillRet/Jev, Qwen8B and both BGE studies remain byte-preserved.
The failed BGE adapter improved a short validation proxy while harming even
metadata-only official retrieval. That evidence motivates preserving the strong
SkillRet document space and using full-corpus development; it does not identify
a single cause for the earlier failure.

The pinned [SkillRet embedding](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B)
revision is `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d`.
The pinned [specialized reranker](https://huggingface.co/ThakiCloud/SKILLRET-Reranker-0.6B)
revision is `0c9a1acc32d3c8ceb506ef008c011566762bff6a`.
Per-file SHA256, upstream blob identifiers and sizes are recorded in source
manifests. The fallback [Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
was inspected as a reference; it is not a silently substituted starting model.
No model card or SKILL.md instruction is executed.

Both checkpoint configurations define a Qwen3 backbone with 28 layers, hidden
size 1,024, 16 query heads and eight key/value heads of dimension 128. Actual
query/value attention projections are discovered by module name and their count
is checked before LoRA injection. Rank eight adds 1,146,880 trainable parameters
to those 56 projections; runtime receipts verify the actual count. Base weights
remain FP16 and adapter weights FP32.

The saved configuration uses `rope_parameters` from a newer Transformers
version. The pinned Transformers 4.57.1 loader explicitly copies the saved
1,000,000 theta and default RoPE type, disables scaling, and checks the effective
rotary frequencies. It must not silently inherit a library default theta.
CUDA SDPA uses repeated key/value heads and the memory-efficient kernel, with
math/Flash/cuDNN fallbacks disabled. Actual forward and backward admission is
required on SM75; model size alone is not evidence of fit.

The embedding protocol retains the original instruction, raw query whitespace,
stripped document fields, left padding, last-nonpadding-token pooling, float32
L2 normalization and right truncation at 32,768 tokens. Exactly two historical
test documents exceed this cap. The two long training documents are separately
recorded. Query anchoring uses frozen vectors of complete instructed queries;
all training queries fit 1,024 tokens without truncation. The original document
index remains unchanged during query-only updates.

The reranker retains its released yes/no prompt and assistant suffix. Its score
is the yes-minus-no logit difference. Only the final hidden state and those two
output-head rows are needed. Before use, a small non-test fixture compares
scores and adapter gradients with the full-vocabulary, all-token reference,
including heterogeneous left-padded batches. BCE applies to skill-level labels.
For full-window max aggregation, exact first-argmax recomputation replays the
winning window's dropout RNG and restores the post-scoring RNG. A separate
scaled-gradient comparison against the naive retained-graph implementation is
required before that path may train.

[Sentence Transformers encoder losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
and [cross-encoder losses](https://sbert.net/docs/package_reference/cross_encoder/losses.html)
provide context for contrastive and BCE objectives. This experiment implements
its declared multi-positive full-index cross-entropy and anchored query loss
directly, so accumulation is not mistaken for in-batch negative competition.
[PEFT troubleshooting](https://huggingface.co/docs/peft/developer_guides/troubleshooting)
informs explicit dtype and save/reload checks.
[Gradient caching research](https://arxiv.org/abs/2101.06983) was inspected;
the query-only route does not need two-sided contrastive gradient caching.
Window-max recomputation is a narrower exact operation, with its own parity
checks, rather than an unverified claim of implementing that full method.

The released models may already have seen the public training partition.
Development is held out only from our local updates. The official test is
historically exposed. Source repository/content grouping, multi-positive
exclusions and identical candidate/input rules reduce avoidable leakage; they
do not establish unseen-pretraining or fresh-blind-test performance.

The additional exposure audit found **zero** shared exact skill IDs, source URLs
or normalized body hashes across the official train/test skills, but **1,054**
source repositories occur in both original partitions. Thus official test
performance must not be presented as new-source-family generalization. Our
training-derived development grouping is stricter within TRAIN. No test labels
were used for this audit or to change the split. Identical normalized training
query texts have no conflicting positive sets. The archived official instructed
queries range up to 862 tokens, compared with 307 in TRAIN; neither distribution
is truncated by the 1,024-token query budget.

The initial real checkpoint fixture diverged in gradient norm before either
branch reloaded. A separate parameter-only probe reloaded all 112 FP32 adapter
tensors exactly. This points to the default efficient CUDA attention backward
path, which [PyTorch documents as nondeterministic by default](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md#cuda-scaled-dot-product-attention).
Enabling strict deterministic algorithms preserved the efficient kernel and
made an uninterrupted four-update run identical to two updates plus checkpoint
reload and two more. Both fixture attempts are preserved and their weights
were discarded before any model-selection pilot.

The skill-group split also changes the relevance-label mix: 468 selection
queries and 473 confirmation queries have one positive, compared with a much
more balanced one/two/three-positive mix in raw TRAIN. The exclusion and
membership manifests account for this; no retained query loses any known
positive. Full-corpus, full-document validation is a stronger retrieval check
than the earlier short proxy, but this label-mix shift still limits its
representativeness.
