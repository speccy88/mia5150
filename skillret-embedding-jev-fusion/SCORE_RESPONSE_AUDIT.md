# Native Score fields with arithmetic diagnostics

The current frozen policy is **`native-fields-arithmetic-diagnostics-v1`**, implemented on 2026-10-07. Structurally valid provider scores and confidence are used exactly as reported. Probability normalization, weighted-score agreement, documented MAD-confidence agreement, and the existing joint nearest-cent calculation are diagnostic only. The parser does not normalize probabilities, substitute recomputed values, drop candidates or queries, or widen an arithmetic tolerance. The historical acceptance rules below are superseded by this policy.

## Amendment trigger and timing

The first 20 saved actual responses were compatible with the previous strict or joint nearest-cent rule. The 21st HTTP 200 response contained probabilities `(0, .15, .83, .02, 0)`, native score **1.87**, and confidence **.84**. Its unmodified body is pinned at SHA256 `62c8c20b1156425d54f8f7bd88c344099efd2c8381a67f84cedabac7d310a170`.

The displayed distribution sums to one and has weighted mean 1.87. Its documented mode-based MAD confidence is `1 - .17/1.2 = .858333...`. Closed half-cent intervals cannot jointly explain the reported confidence: the permitted latent distribution has MAD at most .185, requiring confidence at least `.845833...`, above the .84 interval's upper endpoint .845. This answer is recorded as **inconsistent**, without claiming the provider's internal computation has been reconstructed.

The coordinator froze this amendment after structural/arithmetic diagnosis and **before inspecting retrieval-quality metrics**. Independent source and baseline reviewers recommended native-field evaluation with uniform diagnostics rather than further tolerance adjustments. No effectiveness result was used to choose which discrepancies to accept. The study report must disclose this amendment and summarize diagnostics across the entire evaluated scope; inconsistent responses must not be described as mathematically validated distributions.

## Current hard requirements

The exact typed Score structure, legend and five probability keys remain mandatory. All probabilities must be finite numeric values in [0,1]; native score must be finite and in [0,4]; confidence must be finite and in [0,1]. Booleans, strings, nulls, nonfinite values, incorrect/missing/extra keys, and out-of-range values remain invalid. The legend must exactly map string keys `0` through `4` to the unchanged ordered rubric. Exact answer IDs, actual model and expected pin, usage, context bounds, request/response hashes, raw transport evidence, ledger provenance and budget gates remain unchanged.

Structural failures still preserve raw evidence and prevent settlement or automatic replay under the existing client. An arithmetic diagnostic alone no longer triggers that failure path. The policy applies uniformly to **all saved and future answers** whenever parsed, including answers previously accepted under strict or cent rules. Existing raw bytes and hashes remain untouched. Any held request still requires the coordinator's explicit reconciliation from its saved response and ledger; this parser change performs no API call, replay, credential access, ledger mutation, or automatic reconciliation.

## Current per-answer diagnostic schema

`SCORE_RESPONSE_POLICY` defines the version. Every parsed candidate's `response_validation` records:

- `policy`, `mode`, `arithmetic_diagnostics_only`, and `provider_values_preserved`.
- `reported_probability_sum`, `recomputed_score`, `recomputed_confidence`, `reported_distribution_first_mode`, `recomputed_mad`, and `uniform_midpoint_mad`.
- `residuals`: probability sum minus one, native score minus recomputed score, and reported confidence minus recomputed confidence.
- `checks`: probability sum, score and confidence within the strict tolerance; combined strict consistency; whether all fields lie on the cent grid; and joint nearest-cent feasibility.
- The unchanged strict absolute tolerance **0.00001** (zero relative tolerance), the original rounding-policy identifier, and its explicit observation-only caveat.

Modes are **`strict`** when all three strict comparisons pass, **`cent`** when strict comparison fails but joint nearest-cent feasibility holds, and **`inconsistent`** otherwise. All modes retain the same reported fields and may be ranked when structural requirements pass. Joint-cent feasibility is recorded independently for strict cases too. The existing cent-grid recognition and exact feasibility mathematics are unchanged; no additional rounding margin was introduced.

Recomputed score is `sum(i*p_i)`. Recomputed confidence is `max(0, 1 - sum(p_i*abs(i-m))/1.2)`, with `m` the first maximum-probability level. For non-normalized displayed probabilities these remain raw-distribution diagnostics: the parser never first rescales probabilities. Ranking remains **native score descending, reported confidence descending, original dense rank ascending**. The native score plus one remains the display relevance value. Diagnostic modes and recomputed quantities never enter ranking or RRF60.

## Amendment verification

Regression tests retain structural rejection cases and change arithmetic-only rejection expectations to diagnostic assertions. They cover all three modes, the pinned 21st-response numeric pattern, native value preservation, unchanged score/confidence ranking, and a mocked HTTP response that settles once and reuses its unmodified cache without another call or charge. All tests are offline and use synthetic credentials only. No real request, response file, or ledger is modified.

The response-only AST comparison checks named functions before and after this amendment. Only `validate_score_answer` changes; `SCORE_RESPONSE_POLICY` is added. Request construction, canonicalization, hashing, rubric, context policy, batching, preparation, ingestion mechanics, ranking, fusion, and prior cent-feasibility mathematics remain unchanged. Real prepared requests and manifests are not rewritten. The coordinator records final test evidence and source hashes.

## Historical strict and cent acceptance rules (superseded)

The remainder preserves the earlier rationale and mathematical review for audit history. Its statements about rejecting arithmetic inconsistencies describe the former policy, not the current native-field policy above.

### Original strict Score response validation

Implemented 2026-10-07 using the previously frozen public-source findings in `CONTEXT_SOURCE_AUDIT.md` and `API_READINESS.md`. No new network request, live credential read or hosted evaluation was used for this change.

The earlier parser checked answer IDs/type and numeric score range but allowed absent probabilities, legend and confidence. It could therefore accept an incomplete or internally inconsistent Score answer. The corrected `validate_score_answer` function is used by both direct responses and cached-response ingestion.

The source contract is the [Score specification](https://docs.typesafe.ai/primitives/score) and [Score confidence formula](https://docs.typesafe.ai/confidence#score), as captured in the independent source audit. Every answer now requires:

- A `score` answer with a legend mapping exactly `"0"` through `"4"` to the frozen rubric descriptions in their original order.
- Exactly five finite numeric probabilities, each in [0,1], whose sum is approximately one. Booleans, strings, missing/extra indices and nonfinite numbers are rejected.
- A finite score in [0,4] consistent with `sum(i * p_i)`.
- Finite confidence in [0,1] consistent with `max(0, 1 - sum(p_i * abs(i - m)) / 1.2)`, where `m` is the first maximum-probability level. The denominator 1.2 is the five-level uniform mean absolute distance from the midpoint. Confidence is not the largest probability.

The original strict path retains an absolute comparison tolerance of **0.00001**, with zero relative tolerance, for probability sum, weighted score and confidence. This is a fixed study validation choice, not a documented provider precision guarantee. The parser retains provider scores, confidence, probabilities and legend without normalization or rounding. A provider response outside both the strict path and the explicitly versioned rounding path below is preserved as raw evidence and fails validation; the client retains its spending reservation and forbids automatic replay.

## Observed nearest-cent compatibility

The first saved HTTP 200 response exposed a gap in the strict path: native scores, confidence and probabilities appeared rounded to hundredths, so recomputing from displayed probabilities could disagree with the separately displayed values. For example, displayed `(0,0,.76,.15,.09)` has weighted mean 2.33, while the service returned score 2.32 and confidence .73. The original raw body remains pinned at SHA256 `c69d7a4175615c73aacbe1966967b265e09a40a27edac1c77fb399eb9559bc4c`. No response was replayed for this parser change.

`observed-nearest-cent-joint-feasibility-v1` is a fallback only after strict consistency fails. It is an **observed rounding hypothesis**, not a vendor precision promise: the official sources describe probabilities summing approximately to one but do not guarantee two-decimal output or a tie-rounding convention. Every observed probability, score and confidence must lie on the .01 grid; only binary floating-point residue is ignored when recognizing that grid. Each value represents a closed half-cent interval, clipped to its permitted range. Closed endpoints accommodate the undocumented tie convention.

The parser requires one common latent five-level probability distribution satisfying all of the following simultaneously: every probability interval, sum exactly one, the reported score interval, the reported confidence interval under the documented MAD formula, and a valid first maximum-probability level. Checking score and confidence separately would be insufficient. For example, displayed probabilities `(.5,.5,0,0,0)` admit score .50 and separately admit confidence .57, but not both together; the validator rejects that combination.

The calculation uses exact integer inequalities. Set `x_i=1000*p_i`; a probability of `k` cents gives `max(0,10k-5)<=x_i<=min(1000,10k+5)` and `sum(x_i)=1000`. A score of `s` cents gives `10s-5<=sum(i*x_i)<=10s+5`. With mode `m` and positive confidence `c` cents, the MAD sum lies between `max(0,1194-12c)` and `1206-12c`. At confidence zero, only the lower bound 1194 applies because confidence is floored at zero.

The implementation eliminates `x_4`, enumerates the bounded polytope's vertices using four active constraints, and tests exact integer adjugate/determinant solutions. Candidate modes respect all probability intervals. For earlier tied levels, separate strictly positive vertex margins can be combined by convex averaging to prove a first-mode-compatible distribution exists. Coefficient geometry is cached; no external optimization package is required. The parser records whether validation used the strict path or this observation-based fallback. It never substitutes a witness distribution or recomputed score/confidence for provider values used in ranking.

Independent mathematical review confirmed the scaling, confidence-zero treatment and joint-feasibility logic. Explicit latent witnesses exist for the discrepant saved cases, but those witnesses are compatibility proofs, not claimed hidden model output. On the five supplied diagnostic patterns, a local benchmark measured 0.0484 seconds for the cold first pass and 0.0584 seconds for 100 cached validations, approximately 0.584 ms per answer. Extrapolations to other response distributions are estimates, not runtime guarantees.

Regression tests cover missing/null fields; malformed levels or legend; nonfinite, negative and nonnumeric probabilities; normalization; inconsistent score/confidence; first-maximum tie handling; point-mass and uniform distributions; preservation of within-tolerance values; cached responses with valid hashes but invalid schemas; and raw persistence plus retained reservations for an incomplete live-shaped mocked response. Rounding tests additionally cover the supplied numeric diagnostics, joint-infeasible score/confidence pairs, hidden mode ties, confidence flooring, non-grid rejection, impossible normalization and 100 deterministic latent distributions rounded to hundredths. All transports are mocked; numeric diagnostics are reused offline without another API call.

The earlier strict-parser AST comparison found changes only to response validation and its tolerance constant. The rounding update adds response-only math helpers and standard-library imports. Request construction, canonicalization, hashing, rubric, context policy, batching and preparation remain unchanged. No prepared real request or manifest was rewritten.

Changed code/test files: `run_jev_rerank.py`, `test_jev_prepare.py`, `test_hosted_jev_client.py`. The coordinator runs the final combined offline suite and records its results with the delivered evidence.
