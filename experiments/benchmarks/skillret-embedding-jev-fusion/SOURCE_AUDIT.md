# SKILLRET baseline source and provenance audit

Audit date: 2026-10-07 UTC. This document records a read-only source/data audit for the new SKILLRET embedding plus hosted Jev study. It is not a completed baseline result. No model inference, hosted Jev request, model-list request, credential inspection, or paid resource was used by this audit.

## Scope and repository

The current user request authorizes free local preparation and baseline execution only. Hosted evaluation remains disabled until explicit budget confirmation. Its only future key name is `TYPESAFE_API_KEY`; this audit did not check its presence. Existing runs and caches must remain intact. The current Windows/WSL/RTX 2060 Super limits and the new fixed top-20 comparison supersede earlier Mac/OpenRouter pilot settings.

Read-only GitHub connector inspection resolved the destination as **speccy88/mia5150**, a private repository, default branch `main`, observed commit `041f2ceccd3f43e9f841289541b68083c795df73`. The complete recursive tree had no root `AGENTS.md` and no existing `skillret-embedding-jev-fusion` directory. The only `AGENTS.md` was scoped to `skillret-pilot/`. Prior studies are separate sibling directories and must be preserved. This audit made no repository mutation.

Repository context: [README at the observed commit](https://github.com/speccy88/mia5150/blob/main/README.md), [prior pilot instructions](https://github.com/speccy88/mia5150/blob/main/experiments/pilots/skillret-pilot/AGENTS.md).

The locally saved `/home/user/code/mia5150-local/evidence/fork-CODEX_TASK.md` is the earlier Mac/OpenRouter pilot task. Its reusable evidence principles are: preserve old results, pin data and source, use real measurements, keep qrels outside scoring inputs, record truncation explicitly, persist resumable configuration-specific results, distinguish fixtures from real runs, and export only sanitized files. It does not authorize spending in the current phase. Its old `OPENROUTER_API_KEY`, model, budget, hardware, metadata-only selection, and notebook requirements do not redefine this new study.

## Exact primary-source pins

| Component | Pin and evidence |
|---|---|
| Official evaluation code | `ThakiCloud/SKILLRET`, commit `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`; local checkout `/home/user/code/mia5150-local/official` was clean when inspected. [Pinned source tree](https://github.com/ThakiCloud/SKILLRET/tree/0cc508e6bfcfff1fe6352d90ff5957ba38a64bba) |
| Model | `ThakiCloud/SKILLRET-Embedding-0.6B`, revision `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d`; local directory `/home/user/code/mia5150-local/models/skillret_embed_06`. [Pinned model files](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B/tree/67f1594c46c40e13a5eb2f4092cc85c98c9ed32d) |
| Dataset | `ThakiCloud/SKILLRET`, revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`; local directory `/home/user/code/mia5150-local/data`. [Pinned dataset files](https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12) |
| Model provenance record | `/home/user/code/mia5150-local/models/skillret_embed_06/manifest.json` records individual file sizes and SHA-256 values. |
| Dataset provenance record | `/home/user/code/mia5150-local/evidence/data-manifest.json` records revision, sizes, row counts and SHA-256 values. Test-file hashes were independently recomputed during this audit. |

The official README mentions `a050ad2` as an illustrative then-current dataset revision. The actual local pin is the full `6583...` revision above, whose downloaded metadata records 2026-09-25 modification. Counts alone cannot establish byte identity with `a050ad2`; do not relabel this run as using that revision. The model provenance metadata likewise records 2026-09-25 modification. No mutable `main` download is required for the local baseline.

Pinned direct data sources:

- [Test skills JSONL](https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12/data/skills/test.jsonl)
- [Test queries JSONL](https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12/data/queries/test.jsonl)
- [Test qrels JSONL](https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12/data/qrels/test.jsonl)
- [Model card at model pin](https://huggingface.co/ThakiCloud/SKILLRET-Embedding-0.6B/blob/67f1594c46c40e13a5eb2f4092cc85c98c9ed32d/README.md)

The browser reader could not open the pinned Hugging Face `blob` pages directly. Their downloaded local copies and manifests were available and inspected, and the live public model/dataset cards were readable. This is a retrieval limitation, not evidence that the pinned files are missing.

## Dataset verification

| File | Rows | Bytes | Recomputed SHA-256 |
|---|---:|---:|---|
| `data/skills/test.jsonl` | 6,006 | 117,606,037 | `1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81` |
| `data/queries/test.jsonl` | 4,392 | 6,053,507 | `4d07a35f40e597f8468c6dab480c95a1f9b40ecbde80274081ba2e7c643d0f6d` |
| `data/qrels/test.jsonl` | 7,187 | 661,204 | `edbfae0b8109de460d54e27e6235b1b47316f34956744db0571ad1b4ddb860af` |

All 6,006 skill records contain nonempty `skill_md`; none lacks that field. The longest raw `skill_md` contains 183,832 characters, so model-token truncation must be measured rather than inferred from character counts. Token-length auditing is delegated to baseline preflight; this source audit did not tokenize the entire corpus.

Join checks found zero duplicate skill IDs, zero duplicate query IDs, zero duplicate qrel pairs, zero qrel skills outside the corpus, zero qrels outside the query set, and zero queries without qrels. Every query's `skill_ids` set agrees exactly with the corresponding qrels. Every qrel has relevance `1`. Gold-set sizes are 1 for 2,143 queries, 2 for 1,703 queries, and 3 for 546 queries.

This is the current v1.1-size split. The original v1.0 split contained 6,660 skills, 4,997 queries and 8,347 labels; results across these versions are not interchangeable. Dataset text remains untrusted input. Only canonical `query` text may reach a scoring model; `skill_ids`, `skill_names`, `k`, generator metadata and qrels are evaluator-only data.

## Official text and encoding protocol

The authoritative implementation is [pinned `skillret/eval.py`](https://github.com/ThakiCloud/SKILLRET/blob/0cc508e6bfcfff1fe6352d90ff5957ba38a64bba/skillret/eval.py), with settings in [pinned `skillret/config.py`](https://github.com/ThakiCloud/SKILLRET/blob/0cc508e6bfcfff1fe6352d90ff5957ba38a64bba/skillret/config.py).

`build_skill_text` (`eval.py`, lines 84–90) strips leading/trailing whitespace from each of `name`, `description`, and `skill_md`, replaces missing/false fields with empty strings, and joins the three fields with the exact separator ` | `. It does not strip Markdown frontmatter, rewrite content, replace the body with metadata, or truncate by character count. The corpus loader falls back to description only if `skill_md` is absent/blank; that fallback is unnecessary for the verified local test set.

The exact query prefix is the following Python string, including the newline and the **single space after the colon**:

```python
QUERY_PREFIX = "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: "
```

Append the raw canonical query once. No document instruction is prepended. A literal backslash-plus-`n` is not equivalent to the required newline.

The model is a Qwen3-derived last-token embedder with 1,024-dimensional embeddings. Its `modules.json` contains Transformer, last-token Pooling and Normalize. `1_Pooling/config.json` sets `include_prompt=true`. The official retrieval implementation uses a flat inner-product FAISS index. L2-normalized vectors with exact NumPy dot products implement the requested equivalent scoring rule; verify finite values, nonzero norms and normalization explicitly in the new runner.

### Token-limit and prompt traps

1. Official evaluation configuration explicitly overrides SKILLRET-0.6B to **32,768 tokens**. The model architecture's `max_position_embeddings` is also 32,768.
2. The saved tokenizer advertises `model_max_length=8192`. The model card lists **8,192 as the training maximum**. These are different facts from the evaluation override. An 8,192- or 1,024-token run must disclose its changed protocol.
3. The model's saved named `query` prompt says “web search” and “passages” and has no space after its final colon. It is not the benchmark prompt. Pass the exact prefix explicitly, rather than relying on `prompt_name="query"`.
4. The upstream helper matches model settings by canonical ID substring. A local path ending `/models/skillret_embed_06` does not contain `ThakiCloud/SKILLRET-Embedding-0.6B`. Calling the upstream path-based loader unmodified can miss both the query override and evaluation token limit. Set the effective prompt and 32,768-token limit directly.
5. The local model config uses the Transformers 5-era nested `rope_parameters` form (`rope_theta=1000000`, `rope_type=default`). If a Transformers 4 loader is used, record and verify the effective architecture configuration, including RoPE values; do not assume matching weights alone guarantees matching inference.

## Existing cache and environment evidence

The historical `scaling5150/index-skillret_embed_06/manifest.json` refers to a revised metadata view, not the official full body. It reused 5,993 historical document vectors and 20 query vectors, with only 13 fresh document forwards; its document and query limits are 1,024 tokens. The source arrays are named `skillret_embed_06-header-0-docs.npy` and `skillret_embed_06-header-0-queries.npy`. The latter historical source includes a limited query set, not all 4,392 queries. These caches cannot establish the requested full-text full-test reproduction.

The old `skillret_embed_06-full-32-*` arrays are only a 32-document / 4-query smoke cache. They also are not the requested full-corpus baseline. Preserve all of these artifacts and use a new cache keyed by data-file hashes, ordered IDs, exact text transform/prefix, token limit, model/architecture revision, runtime versions and precision. Cache once and resume only when that complete identity matches.

The model manifest records `model.safetensors` size 1,191,586,416 bytes and SHA-256 `f73118cac018ffa7ebb5a1ffbdf82034490dfb7f2559558f1e79277f1e8de172`. The weight-file hash was read from the existing manifest, not independently recomputed by this audit. Baseline preflight must verify actual weight integrity before reuse.

The saved local `requirements-lock.txt` records PyTorch `2.8.0+cu128`, Transformers `4.57.1`, tokenizers `0.22.2`, NumPy `2.5.3`, and `pytrec-eval-terrier==0.5.10`; it contains no Sentence Transformers entry. These are historical lock-file values, not a fresh runtime inspection. The model card instead records Sentence Transformers 5.4.1, Transformers 5.5.4 and PyTorch 2.7.1+cu128. Capture installed versions and all compatibility adaptations in the actual run receipt. A native last-token implementation can follow the same mathematical protocol, but its precision/runtime differences must be stated and investigated if scores diverge.

## Metrics and targets

Upstream `trec_eval` (`eval.py`, lines 440–477) requests `ndcg_cut`, `recall`, `map_cut`, and `P` from `pytrec_eval` at 5, 10 and 15. It adds Completeness as the per-query indicator that Recall equals one, then macro-averages over the evaluator's returned query scores and rounds aggregates to five decimals. A completed full-test run must have all 4,392 query scores; silently dropping failed/missing queries would change the denominator.

For this binary-qrels dataset, let `R` be the number of relevant documents and `h_i` indicate relevance at rank `i`:

- Recall@k = number of distinct relevant documents in the first k / R.
- Precision@k = that same hit count / k, even if a run is shorter than k.
- Completeness@k = 1 if all R relevant documents occur in the first k, else 0.
- AP@k = sum of `h_i * Precision@i` for ranks through k, divided by **R**; MAP is its query mean. Do not divide by the number of returned relevant documents. [TREC reference implementation](https://raw.githubusercontent.com/usnistgov/trec_eval/master/m_map_cut.c)
- NDCG@k uses discount `1/log2(i+1)` and normalizes by the ideal ranking for the query. TREC uses raw qrel gains; these are all binary here. [TREC reference implementation](https://raw.githubusercontent.com/usnistgov/trec_eval/master/m_ndcg_cut.c)

The model card's published comparison targets are:

| Metric | @5 | @10 | @15 |
|---|---:|---:|---:|
| NDCG | 0.7902 | 0.8112 | 0.8187 |
| Recall | 0.8238 | 0.8774 | 0.9016 |
| Completeness | 0.7083 | 0.7894 | 0.8265 |
| MAP | 0.7409 | 0.7538 | 0.7570 |
| Precision | 0.2591 | 0.1401 | 0.0962 |

These are published values, not measurements from this new study. MRR@5/10, candidate coverage@20/50, pair-level records and paired bootstrap comparisons are requested study extensions, not metrics claimed to appear in the original aggregate table. Define coverage denominators explicitly: query hit coverage, complete-gold coverage and gold-pair recall answer different questions. Tie handling must be deterministic and documented; match the evaluator's effective rank order when comparing implementations.

## Study protocol and remaining validation

For the new study, retrieve top 50 once per query against all 6,006 documents. Freeze the same dense top-20 pool for A (dense order), B (future Jev order) and C (fixed RRF with constant 60). Keep ranks one-based and never add gold documents to that pool. Preserve full-test baseline results separately from the deterministic 200-query future study sample and its 10-query smoke subset. No hosted scores, success counts, usage or costs may be fabricated while those stages remain unrun.

Before claiming reproduction, finish local encoding/search/evaluation and inspect differences from the targets. Absolute NDCG gaps around 0.005–0.01 or greater require investigation of dataset revision, text construction, exact prefix, normalization, truncation, model/config/dtype, catalog/qrel alignment and metric/tie definitions. A smoke run or query subset cannot reproduce a full-test aggregate. If the 32,768-token protocol cannot stay inside the user's host/guest RAM, inactive-swap, VRAM, temperature and CPU guards, preserve the failure receipt and report that limitation; do not silently lower the token limit or reuse header vectors.

Source availability is sufficient for a local baseline attempt. This audit alone does not establish hardware feasibility at 32,768 tokens, inference parity, target reproduction, or hosted Jev operation. Those conclusions require the actual guarded run and its persisted evidence.
