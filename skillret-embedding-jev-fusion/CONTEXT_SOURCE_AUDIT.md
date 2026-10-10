# Jev context and Score contract: independent primary-source audit

Verified 2026-10-07 UTC using public documentation and the official Python SDK. No TypeSafe API call, authenticated model-list call, credential read, or paid action was performed. This audit edits documentation only.

## Documented context versus our admission policy

The current model page identifies `jev-1.13.0`. Its exact context wording is **“64k tokens per request; 32k tokens for `state` plus the longest question”**. The surrounding explanation applies the larger limit to shared state plus all questions; the smaller applies to shared state plus one question. These are simultaneous limits. The documentation does not define `k` as 1,000 or 1,024. Therefore neither 65,536/32,768 nor exact 64,000/32,000 can be claimed as a verified server boundary. The latter pair is a conservative local interpretation. [Models](https://docs.typesafe.ai/models)

The same page documents `GET https://api.typesafe.ai/v1/models` with bearer authentication. That route currently lists aliases; versioned IDs may be accepted without appearing in the list. `jev-latest` and `jev-preview` currently resolve to `jev-1.13.0`; each evaluation response reports its actual model. No live route was invoked here. [Model discovery](https://docs.typesafe.ai/models#listing-models)

The existing preparer's default admission calculation is:

```text
total estimate = byte length of canonical UTF-8 JSON request + 2,048
pair estimate  = byte length of canonical UTF-8 JSON containing
                 the same state and exactly one question + 2,048
require total estimate + 2,048 output allowance <= 64,000
require every pair estimate + 2,048 output allowance <= 32,000
```

Both 2,048 allowances are study choices, not vendor-specified requirements. Thus a single-question request has a maximum **27,904 serialized UTF-8 bytes** at the defaults, including the query, instruction, rubric, field names, escaping and wrapper. The available document bytes are smaller. Counting escaped JSON bytes also counts transport material that the provider may serialize differently internally. This is deliberately pessimistic admission estimation; it is not native Jev tokenization or a proven token upper bound. Existing fields named `estimated_*_upper_bound` must be interpreted with that limitation.

The previous preparation's 4/10 smoke and 96/200 pilot blocked-query counts were reported by the coordinating task. The code explains them: it first tests every single candidate against the smaller estimate limit, and one estimated overflow marks the entire query blocked before batch packing. A one-candidate fallback cannot reduce that candidate's own full text. These are **local admission failures**, not observed server rejections and not proof that those texts exceed Jev's native context. This audit did not recompute those counts.

## Token counting and error evidence

The public documentation index, HTTP reference, SDK reference and pinned Python package expose no native tokenizer download, tokenizer revision or count-tokens endpoint. The response includes `usage.input_tokens` and `usage.output_tokens` after evaluation. That is not a preflight counting service. The HTTP reference describes generic validation errors but does not document a dedicated context-overflow error code, exact boundary message, billing guarantee on overflow, or free context probe. Do not infer any of those from `400` or `422`. [Documentation index](https://docs.typesafe.ai/llms.txt), [HTTP API](https://docs.typesafe.ai/api), [SDK exceptions](https://docs.typesafe.ai/sdk/python/api/exceptions)

Independent public-source inspection scanned all **27 Python files under `src/` plus `pyproject.toml`** at SDK commit `f078f1e208a0d885154dc758344ae4fce77ac168`, version `0.7.2`. Searches for `tokeniz`, `count.?tokens`, `tiktoken`, `sentencepiece`, `32768`, `32000`, `64000`, and `65536` returned no matches. Endpoint construction exposes evaluation and models listing only. Package dependencies contain no tokenizer library. This bounded inspection supports “not found in inspected sources”; it cannot prove that no private provider facility exists. [Pinned endpoint construction](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/endpoints.py), [pinned constants](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/constants.py), [package metadata](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/pyproject.toml)

## Deterministic batching and any text reduction

TypeSafe recommends combining questions that share state; questions are evaluated in parallel. It also explicitly permits data inside structured `instructions`, so the study's query-only shared state plus one candidate document in each question is supported. The provider does not prescribe our greedy packing order or truncation policy. [Fan-out pattern](https://docs.typesafe.ai/patterns/fan-out), [structured questions](https://docs.typesafe.ai/api#question-types)

Freeze the dense top-20 order and greedily pack its complete questions under both local admission limits. Split before altering text. Keep query state, rubric and IDs unchanged; never reduce a 200-query study to its 104 previously admitted queries. If an individual document still fails the chosen conservative policy, deterministic reduction can be prepared under the user's authorization, but label it **required by the local estimator**, not proven unavoidable for Jev. Preserve original text locally. Record policy version, original/prepared character and byte counts, exact retained spans, original/prepared hashes, omitted counts, affected candidate/query IDs and resulting batch membership. Report text-reduced and full-text subsets separately while retaining the entire fixed query denominator. Do not use relevance labels to choose cuts or revise the rubric.

No source reviewed establishes a native token conversion factor for English, code or non-ASCII text. Qwen/SKILLRET counts, bytes divided by four, or a different public tokenizer can be auxiliary estimates only; none can be relabeled exact Jev tokens. The fixed conservative byte gate is reproducible but may discard substantially more text than the service requires.

## Score and confidence semantics

`POST https://api.typesafe.ai/v1/systemone` accepts shared `state`, `model`, and a question map. A Score question uses `type: "score"`, `instructions`, and 2–10 ordered criterion descriptions. Five descriptions define native levels **0, 1, 2, 3, 4**. The continuous returned score is `sum(i * p_i)`, not an integer class and not a probability of relevance. A displayed 1–5 scale is `score + 1`; preserve the native value for ranking. The answer also includes `legend`, `probabilities`, and `confidence`, keyed by the question ID. [Score specification](https://docs.typesafe.ai/primitives/score)

Score confidence is **not the highest level probability**. For `n` ordered levels, the documented calculation is:

```text
m = most likely level (first maximum in the documentation's example code)
D = sum(abs(i - (n - 1)/2) for i in range(n)) / n
confidence = max(0, 1 - sum(p_i * abs(i - m)) / D)
```

For five levels, `D = 1.2`. Confidence measures concentration around the most likely level and penalizes distant probability more strongly. It does not guarantee correctness. Preserve provider confidence and probabilities alongside the unrounded score; the study's confidence tie-break is a study rule. [Exact confidence formula](https://docs.typesafe.ai/confidence#score)

## Outcome

The context-policy blocker is understood without hosted calls. The documented limits remain rounded; exact native preflight counting and dedicated overflow semantics remain undocumented in the inspected sources. This audit provides no new authorization to read a saved key or execute a paid request. Revised preparation counts and truncation effects must come from the separately generated, frozen manifests.
