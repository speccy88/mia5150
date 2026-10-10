# Jev reranking development trial

## Answer

Worth retaining as an additional experimental condition. On **200 of 200 fixed pilot queries**, BM25 Recall@5 was **0.605833**, BM25 Recall@20 was **0.727500**, and BM25 → Jev Recall@5 was **0.709167**. Absolute change: **0.103333** (fraction units); relative change: **0.170564**.

Improved / worsened / unchanged: **37 / 6 / 157**. Service failures: **0**. Failures retain null probabilities/rankings, never invented zero scores. Valid-only Jev Recall@5 is 0.709167; treating service failures as no delivered result gives an explicitly operational lower bound of 0.709167.

## Protocol and provenance

- Run: `f2dc6fc3f49ca24c4327`; started 2026-09-27T03:08:57.462735+00:00; Python 3.12.6 on `darwin`. Local computation and artifacts are on the Mac; only HTTP judgments run at TypeSafe.
- Dataset: `ThakiCloud/SKILLRET` revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`; corpus 6,006 skills. Exactly the existing 200 pilot IDs, in saved order. The 4,192 final IDs are disjoint and unchanged; no final-query inference or evaluation.
- Existing `pilot.bm25_rank` and `reference_metrics` are reused. Recomputed BM25 top five IDs **and scores** match all 200 historical rows exactly. Relevance is binary positive qrels; per-query recall is retrieved positives / all positives, macro-averaged across queries.
- Cached `data/metadata.jsonl` is reused byte-for-byte in each skill's `name_and_description`; hash `e41da252aadcbd6474465cd790047ce35c00f777bf47033ce77324300feb8c9d`. Original name + description underwent the existing 1,024-character cap, BGE tokenization/decoding of 126 tokens, and a final 128-token check including special tokens. No full bodies, uncapped source names, labels, answer counts, retrieval scores, or ranking positions enter requests.
- One request per query with 20 independent Noul questions over shared state, then highest five probabilities. Several skills may independently be relevant. Candidate display is UUID-sorted; equal probabilities retain original BM25 rank. No prompt tuning after looking at pilot outcomes.
- Pinned `jev-1.13.0`, `POST https://api.typesafe.ai/v1/systemone`, HTTP through existing httpx; no SDK installation, fallbacks, retries, context truncation, skill execution, or unrelated model benchmarks. This evaluates retrieval, not Deep Agents operation or skill execution.
- Repository has no HEAD commit; exact source hashes and installed package versions are recorded in the manifest instead of inventing a git SHA.

## Where gains and failures occur

Jev rescued previously lower-ranked gold skills in **39 queries**, moving **42 gold IDs** from BM25 ranks 6–20 into the top five. It removed BM25 top-five gold skills in **8 queries**, losing **8 gold IDs**. A query can both gain and lose skills; improved/worsened refers to its net recall difference.

BM25 top 20 omitted at least one positive in **80 queries**, so complete recall was impossible there; **32 queries** had no positives available at all. Candidate Recall@20 (0.727500) is the reranker's recall ceiling. Diagnostics overlap for multi-positive queries: missing upstream positives, available positives outside Jev top five, rescued positives, and retained positives are saved separately. They must not be summed as exclusive categories.

## Paired comparison and supporting metrics

Each CSV row contains the paired query difference. The descriptive 95% paired percentile bootstrap interval is **[0.065, 0.14583333333333331]**, using 10,000 paired-query resamples and seed 42; it covers 200 valid pairs. The fixed pilot is a development sample, not a population-random or untouched final test; this interval is exploratory, not a final-paper significance claim. With service failures, valid-pair intervals exclude technical failures and the all-attempt comparison remains explicitly incomplete.

Existing precision, MRR, nDCG, completeness and set metrics are in `summary-200.json`. Fixed top five can include additional nongold skills; exact-set match is not the primary endpoint. No threshold or number-of-answers selection uses hidden gold counts.

## Usage, latency and cost

- API calls: **200**; independent judgments requested / valid: **4000 / 4000**. Stages at 5 and 20 reuse calls within the 200 total.
- Observed provider input / output tokens: **1290762 / 187977**; 0 calls missing usage.
- Estimated cost from observed input tokens × published price: **USD 0.05421200**, approximately **CAD 0.08131801** at the labelled 1.50 planning factor. **Actual billed USD is unavailable**: the API returns token usage, not a currency-denominated charge.
- Pre-run maximum reserved estimate for 200 calls: **USD 0.688128** (full 65,536-token capacity per call × USD 0.042/M input tokens × 1.25 margin). Output tokens are free under the saved pricing snapshot. USD 2 and CAD 3 ceilings, a persistent 200-call cap, sequential execution and a 30-second timeout are enforced before calls. Unknown charges retain reserves; currently reserved USD 0.688128. These guards do not control unrelated account activity.
- Measured request latency: total **35.586s**, median **0.173s**, p95 **0.223s**. This includes network/server time; it is not model-only throughput. Initial local BM25 preparation took 0.329s.

## Evidence and reproduction

Artifacts are under `results/jev/f2dc6fc3f49ca24c4327/`: immutable config/source/provenance manifests, candidate cache, append-only raw request/response rows, staged summaries, summary CSV, and paired-query CSV. `results/jev/budget.jsonl` persists across run versions. Offline fixture tests establish implementation behavior only; live rows establish API operation. Official source snapshots and preservation audit are in `evidence/jev/`. The two existing notebooks are unchanged. The optional vector → Jev condition was not run; the primary comparison is the only new condition.

From the repository on the Mac, with `TYPESAFE_API_KEY` exported privately:

```bash
ALLOW_PAID_RUNS=1 .venv/bin/python jev_reranker.py run --queries 200 --live
.venv/bin/python jev_reranker.py report --run-dir results/jev/f2dc6fc3f49ca24c4327
.venv/bin/python -m pytest -q tests/test_jev_reranker.py tests/test_reference_helpers.py
```

The first command runs the 5 → 20 → 200 technical gates and resumes exact completed cases without another charge. To launch with the locally saved key, prepend `.venv/bin/python -m dotenv -f .env run --no-override --` to the runner command (put `ALLOW_PAID_RUNS=1` before that prefix). The experiment itself only reads `os.environ`; no credential is included in artifacts. Failures are not resubmitted on resume. Preserve the ledger. Changes to source/configuration produce a new version and are subject to the same cumulative cap.

## Interpretation

This is an additional MIA5150 development/pilot condition. Public skills, synthetic queries, capped metadata, incomplete relevance labels and prior consumption of these 200 queries limit external validity. A semantically useful but unlabelled skill counts as incorrect under the existing qrels. The recommendation above concerns carrying the condition forward for a separately authorized, prespecified study; it does not authorize consuming the reserved final split.

Sources: [HTTP API](https://docs.typesafe.ai/api), [Noul](https://docs.typesafe.ai/primitives/noul), [reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe), [batching](https://docs.typesafe.ai/patterns/fan-out), [current pricing and limits](https://docs.typesafe.ai/models). The cookbook's old example model was not copied; the current documented version is pinned.
