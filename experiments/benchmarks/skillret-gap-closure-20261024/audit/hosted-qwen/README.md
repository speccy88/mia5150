# Independent hosted Qwen validation

**Result: passed, with one reporting-label correction.** All 120 selections on 24 previously exposed balanced queries were independently rescored from exported raw returned text and official positive qrels. The exact-set counts are listing 5, BM25 13, vector 7, domain vector 4, hybrid 10 (denominator 24 each).

This audit reconstructs all 144 selection/routing request hashes, verifies response-to-HTTP-ledger joins, parses all 24 routing decisions, recomputes 96 retrieval metric rows, verifies 72 candidate-list overlaps with the independently validated full-retrieval evidence, and reproduces five summary cells and eight paired bootstrap intervals. It uses no project scoring imports and makes no network/model calls.

## Costs and calls

- 147 successful Qwen generation requests = 3 canaries + 24 routing + 120 selection.
- All 147 reported charges also equal recorded input/output tokens times the frozen endpoint rates: **US$0.02411946**, comprising US$0.02410356 benchmark calls and US$0.00001590 successful canaries.
- Actual Qwen usage: 253,574 input tokens and 4,326 output tokens; none missing.
- Two additional failed generation canaries (DeepSeek and Gemma) have unknown actual costs and missing token usage. Their approved conservative holds remain US$0.14 and US$0.02.
- Cumulative generation attempts: **149**. Known charges plus holds: **US$0.18411946**. Actual total spend remains **unknown**.
- The public ledger has 155 merged request rows: 149 generation attempts and six logged read-only GET requests (four authentication, two endpoint availability). One legacy generation row is reconstructed from preserved source. The earlier read-only HTTP total is unknown; the prior GET404 reconciliation receipt is not another generation attempt or another row in this count.

## Correction and limits

At the audited commit, `diverse-execution.json` incorrectly labels the final successful Qwen routing-canary charge as `known_charged_cost_for_failed_canary_usd`. It must be null/removed/renamed. Main cost totals and performance tables are correct. Neither failed canary has a known charge.

This is a balanced exposed N=100, gold-conditioned, qrels-based selection diagnostic, not an official-prevalence estimate, fresh held-out test, human semantic annotation, native skill execution, or original hosted-matrix completion. Five methods do not turn 24 queries into 120 independent observations. The hosted object-wrapper contract, backend/quantization, context and timing differ from the local arms. Reported method-body latency excludes offline retrieval and embedding preparation. Paired intervals are descriptive, nominal and not multiplicity adjusted. This audit verifies consistency of sanitized owner-exported provider receipts, not account invoices. Public raw canary content is absent, so canary format acceptance is owner evidence; the canary identity/usage/cost receipts were audited.

## Source and reproduction

Repository snapshot: `497ca150798d3a4ee3852d0f915e2c1ab276a418` at https://github.com/speccy88/mia5150

Source folder: `experiments/benchmarks/skillret-gap-closure-20261024/hosted/`.

Input packet: `MIA5150_Hosted_Qwen_Evidence_2026-10-24.zip`, Library `libfile_dcf5cfd747e08191baa551b8a27aa109`, 315,783 bytes, SHA-256 `9b211bcec32cf0cb649691590452fae6fa1ca95fd5b99a04a21769bf1e67497d`. All 43 manifest file hashes passed. Public input and audited report file SHA-256 values are in the JSON receipt.

Official public inputs are `skills-metadata.jsonl`, `queries-test.jsonl`, and `qrels-test.jsonl`, available in the previously published local evidence. The optional overlap source is `MIA5150_Full4392_Retrieval_2026-10-24.jsonl.gz`, SHA-256 `a55bfecfa99508cebcc0d6c0238575c60bfbd919d46c2e69654f66dcd0b53a35`.

```sh
python independent_hosted_audit.py \
  --evidence /path/to/extracted/hosted-evidence \
  --public-inputs /path/to/public-inputs \
  --full-retrieval /path/to/MIA5150_Full4392_Retrieval_2026-10-24.jsonl.gz \
  --output independent_hosted_validation.json
```

Python 3 and NumPy are required. This packet contains only this README, the independent audit script and sanitized validation receipt; no credentials, private provider generation identifiers, runtime ledger files, or original student-identifying documents are included.
