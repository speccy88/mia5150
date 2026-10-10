# SkillRet CPU input recreation protocol

This is an exact recreation of the specified input protocol from public pinned sources. It is not recovery of any lost historical model outputs. No model inference is performed here.

## Source and integrity

- Dataset: https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12
- Revision: `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`
- Test split: 6,006 skills, 4,392 queries, 7,187 binary-positive qrels
- Raw file byte sizes, URLs and SHA-256 values: `data/raw/source_manifest.json`
- Processed artifact sizes and SHA-256 values: `data/processed/manifest.json`
- Checks: `protocol/data_audit.json`
- Reproduce online: `python src/prepare_dataset.py`
- Reproduce from verified local raw data: `python src/prepare_dataset.py --offline`
- Only the Python standard library is needed. Existing raw files are hash-checked, never silently trusted.

## Frozen query memberships

Sort all query IDs by `(SHA256(UTF8('skillret-cpu-query-subset-v1' + NUL + query_id)), query_id)`, ascending. The NUL is one zero code point, not two printable characters. SHA digest byte ordering equals hexadecimal lexical ordering.

- `selected_query_ids.json`: the first 64 IDs, the main cohort
- `latency_query_ids.json`: the first 32 IDs, frozen timing and fallback cohort
- `pilot32_query_ids.json`: the identical first-32 alias
- `expensive_query_ids.json`: the original first-256 membership preserved for traceability, not a claim that 256 queries were evaluated
- `queries.jsonl`: selected 64 rows in hash order
- `allqueries.jsonl`: all test queries in lexical query-ID order
- `qrels.jsonl` and `qrels_selected64.jsonl`: selected 64 only, 106 positive rows
- `qrels_all.jsonl`: all 7,187 positive rows

## Frozen candidate memberships and ranking safety

For each selected query, concatenate the complete lexically sorted gold skill IDs with all non-gold IDs sorted by `(SHA256(UTF8('skillret-cpu-catalog-v1' + NUL + query_id + NUL + skill_id)), skill_id)`. Store the first 1,000 IDs in `catalogs.jsonl`.

The 10, 100 and 1,000 pools are prefixes of that stored list. The full pool is all 6,006 sorted test skill IDs from `full_catalog.json`. All pools are nested and contain every positive (at most three). This is a gold-preserving controlled candidate-set experiment; small pools are not unassisted full-catalog retrieval.

Crucially, the positive-first arrangement is storage only. Every scorer must sort the selected candidate IDs lexically before scoring. Never treat storage order as a rank. Sort final results by descending score and ascending skill ID for deterministic ties. Any scorer must verify this itself; the preparation script does not run inference.

## Semantic inputs and metadata restrictions

Every processed skill has exactly `skill_id`, `name`, `description`, `text`, `major`, and `domain`. Semantic `text` is exactly `name + '\n' + description[:1024]`, with slicing in Unicode code points and no stripping or other normalization. The preserved description field is not the semantic input if it exceeds 1,024 characters. Full `body` and `skill_md` are excluded.

Query semantic input is raw query text only. Query `gold_skill_ids`, `major`, and `domain` are evaluation/audit labels, derived from the joined gold skills, and must never be used to route, score, rank, or build a query prompt. Pre-existing skill `major` tags are allowed for index construction only, not as added semantic text. Any routing decision must come from the query text and eligible index, never query ground-truth labels.

## Validation performed

Verified all source checksums and expected counts; no duplicate skill IDs, query IDs, or qrel pairs; all qrels join; query skill sets exactly match qrels; `k` equals the gold count; each query has 1–3 positives; all selected catalogs contain valid unique IDs, preserve all gold, and nest correctly; cohort 32/64/256 nesting; exact field whitelist and semantic formula in written artifacts; correct output row order.

All-data gold-count histogram: {"1": 2143, "2": 1703, "3": 546}. Selected-64 histogram: {"1": 32, "2": 22, "3": 10}.

Decision-reranker ties retain the frozen first-stage rank; retrieval ties use canonical skill-ID order. Both are label-independent.
