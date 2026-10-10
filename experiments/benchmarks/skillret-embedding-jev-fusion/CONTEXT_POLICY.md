# Frozen offline Jev context policy

Preparation defaults to `--oversize-policy preserve_and_block`. The explicitly selected alternative is `--oversize-policy truncate_skill_md_right`, version `skill-md-longest-unicode-prefix-v1`. This policy is fixed before any API response is available. Neither preparation path reads a credential or calls the provider.

The provider documentation describes a 64k total request context and a 32k state-plus-longest-question context. The implementation uses conservative decimal limits of 64,000 and 32,000 because the documentation does not settle the decimal versus binary interpretation. The native tokenizer and a counting endpoint have not been verified. The local admission estimate is therefore **canonical serialized UTF-8 bytes plus 2,048 units of overhead**, not a measured native token count. Both limits additionally reserve 2,048 output units. JSON escaping, the exact query, model wrapper, actual question identifier, fixed question prefix, and five criteria are included in each estimate.

For the individual-pair estimate, preparation serializes the same state with exactly one complete question and its rubric. Its estimate plus the output reservation must fit 32,000. The complete multi-question request estimate plus the output reservation must fit 64,000. These estimates are admission heuristics; they are not a provider billing guarantee. Any truncation described below is required by this conservative local policy, **not proven necessary under the unpublished native tokenizer**.

## Deterministic preparation

1. Freeze the SHA256-selected query IDs and each query's existing dense top 20, in dense rank order. Do not consult qrels or labels when constructing a request.
2. Test every complete `name | description | skill_md` candidate individually with the exact original query and fixed rubric. Preserve the complete text whenever it fits; never shorten a fitting candidate merely to reduce the number of batches.
3. If an individual candidate exceeds either conservative limit, `preserve_and_block` blocks its whole query. With `truncate_skill_md_right`, binary-search for the **longest Unicode-codepoint prefix of the original `skill_md`** that fits both limits. Remove only the right suffix. Add no marker, summary, normalization, or replacement text. Preserve the name, description, query, question prefix, rubric, candidate ID, and dense rank byte for byte.
4. Retain at least one non-whitespace character from `skill_md`. If no such prefix fits, block the whole selected query with an explicit receipt. Never replace a candidate, omit it, silently remove a query, or reduce the selected evaluation denominator.
5. Greedily assemble the prepared candidates in their original dense order, opening a new batch before adding a candidate would exceed either limit. Every prepared query partitions exactly the same 20 candidates across its batches. Multiple questions can share the identical query state within a request.

The longest-prefix search counts the canonical JSON representation of each trial prefix, including escaped quotes, backslashes, control characters and multibyte UTF-8 characters. It never cuts a UTF-8 byte sequence. The prepared prefix may end within a line or code block: this is a mechanical context limit treatment, not a semantic summary. The chosen length can differ for the same skill under different queries because the original query is part of the context budget.

## Audit and immutability

The manifest freezes `context.oversize_policy`, the complete `context.truncation_policy` object, its `context.truncation_policy_sha256`, the estimator version, all supplied limits and reserves, source hashes, query selection, rubric and all exact request hashes. Existing differing artifacts cannot be overwritten; use a new output directory when the policy or inputs change.

Each query-candidate audit records original and prepared document/body character counts, UTF-8 byte counts, SHA256 hashes, both original and prepared admission estimates, the original limits exceeded, the policy/version and reason, removed body characters/bytes, and retained character/byte fractions. Native token count fields stay null. `document_sha256` is the prepared document hash; `original_document_sha256` and `prepared_document_sha256` remove any ambiguity. All 20 audit rows remain present even when the whole query is blocked.

Manifest counts explicitly report selected/prepared/blocked queries, prepared request batches, prepared candidate pairs, queries with truncation, and truncated query-candidate pairs. These are preparation counts, not model results. A document truncated for two queries counts as two pairs because its exact budget depends on its query.

Example, using a new output directory:

```text
python run_jev_rerank.py prepare --queries ../data/queries/test.jsonl --skills ../data/skills/test.jsonl --dense results/dense_top50.jsonl --output-dir results/jev-plan-pilot-prefix-v1 --scope pilot --model jev-latest --context-limit 64000 --pair-context-limit 32000 --overhead-tokens 2048 --reserved-output-tokens 2048 --oversize-policy truncate_skill_md_right
```

The dependency-free offline tests cover the unchanged preservation default, complete fixed candidate pools, the maximal fitting prefix, JSON escaping and multibyte text, unchanged fitting skills, immutable query/metadata overflow, refusal of whitespace-only body evidence, immutable policy receipts, and complete **synthetic** 10/200-query plans. Synthetic test success is not a claim that the real study's plans or hosted calls have run.
