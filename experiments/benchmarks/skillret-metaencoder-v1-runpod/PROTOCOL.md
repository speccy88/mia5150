# MetaEncoder v1 / SkillRet protocol

Execution restriction: inference, vector manipulation, evaluation, calibration, plots and statistical calculations run on RunPod. Local source inspection and authoring are not benchmark measurements.

## Inputs
Dataset `ThakiCloud/SKILLRET@6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`, test files only: 6,006 skills, 4,392 queries, 7,187 positive relevance pairs. The runner validates exact historical SHA-256 hashes, counts, identifiers, pair uniqueness, and references. Documents concatenate stripped name, description and skill_md with ` | `. Only the original `query` field is encoded. Relevance labels enter evaluation or separately labeled calibration, never neural inputs.

Model `facebook/meta-encoder-v1@ed04ebd4defb8e837b5eaea164143177272696ac`, official wrapper, BF16, no training. Fixed experiment instructions are `Represent the document for retrieval.` and `Represent the query for retrieving relevant documents.` These are experimental retrieval instructions, not a claim of a SkillRet-specific Meta recommendation. No instruction search uses test labels.

## Length handling
Count exact formatted text tokens with the model processor chat template and tokenizer. Primary input is content-preserving: for inputs exceeding 16,384 formatted tokens, bisect original character substrings into nonoverlapping pieces each at most 16,384 tokens, without decoding/re-encoding loss. All original characters are retained. Mean chunk vectors, then L2 normalize. A separate first-chunk arm measures a fixed-length truncated protocol. Per-item audits record lengths, chunk count, latency and memory. FP32 normalized per-chunk and aggregated vectors are retained.

## Retrieval and metrics
Cosine over all 6,006 normalized document vectors, stable descending sort with original catalog row order resolving exact ties. Save top100 and top50 IDs/scores. Metric implementation is copied verbatim from the existing JEV-fusion study evaluator; source audit also reads official SkillRet eval.py revision `0cc508e6bfcfff1fe6352d90ff5957ba38a64bba`. NDCG uses linear positive gain; MAP divides by total relevant count; precision divides by cutoff; completeness requires all relevant skills. Unlike the official reporting layer's five-digit rounding, retain full precision. Preserve all qrels query IDs, including missing results as zero. Coverage@20/@50 equals macro recall, also report micro and query-any/all coverage. Stratify by one/two/three positives. Compare exact paired query IDs to saved historical baseline rankings using 1,000 percentile-bootstrap resamples, seed42, 95% CI. No historical baseline inference is repeated.

## Native retrieve and dimensions
Native retrieve re-encodes queries and uses cached corpus tensors, then cosine argsort. Its latency includes query encoding; manual cached matrix search does not. Compare ranking order and set, noting unstable native ties/numerical precision. Truncated-prefix dimensions 256/512/1024/2048/4096 with renormalization are exploratory, not native supported features; full6656 remains retained. Predefine close quality as NDCG@10 within0.01 absolute of full; comparisons do not establish unseen-test generalization.

## Decisions and calibration
Pilot is the first200 query IDs sorted by SHA256(query ID), matching existing deterministic selection. Score up to top50 once and reuse nested top10/top20. Official types are noul (boolean), choice (one option), score (expected ordinal index). Save raw score/probability/confidence, decision, latency and errors. Stable decision-score ranking preserves dense candidate order on ties. Candidate pools cannot recover omitted relevant skills. Classification is conditional on successfully scored candidate pairs; retrieval denominator includes all intended queries. Choice uses5/10/20 candidates and does not fabricate a ranking from one choice. Reports stratify single/multi-positive queries and condition on relevant-candidate availability. Choice schema can exceed comfortable context with long documents; record unsupported/OOM scope honestly rather than silently truncate.

Secondary temperature calibration: first half of frozen pilot query IDs are fit-only, second half are held out. Official scalar temperature calibrate method, no weight training. Evaluate ECE (ten equal-width probability bins), Brier and NLL on held-out pairs. The primary retrieval experiment remains zero-shot; calibrated pilot is not an untouched full4392-query test.

## Checkpoints and cost
Per-input atomic NPZ and JSON checkpoints verify shape, finite values, matching ID and input hash before reuse. Stages accept wall-time bounds; no automatic expansion/spending. Record actual RunPod hardware/billing separately. Mark every unavailable or incomplete arm explicitly.
