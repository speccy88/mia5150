> Historical assessment. The later authorized smoke and pilot are complete. Current evidence and budget outcome are in [BUDGET_STATUS.md](BUDGET_STATUS.md) and [actual findings](results/hosted-study-findings-final/findings.md). Earlier pending-account requirements below are superseded.

# TypeSafe Jev API readiness: public-source audit

Audit date: 2026-10-07 UTC. **Documentation verified; live access and account billing state unverified.** This task used public primary-source documentation and the official public Python SDK source only. It made no TypeSafe API request, model-list request, credential/key-presence check, environment inspection, purchase, or account change.

The coordinating task relayed approval for a **CAD 10 all-in maximum, only after the local baseline is validated**. That is a conditional authorization, not evidence that the gate passed. This document does not authorize bypassing that condition. The earlier `SOURCE_AUDIT.md` records the pre-approval preparation stage; the conditional approval above is the newer budget context.

## Verified interface

The documented evaluation endpoint is `POST https://api.typesafe.ai/v1/systemone`, with bearer authentication and JSON content. Requests contain `state`, `model`, and a named `questions` map. Answers use the corresponding map keys. Question IDs are routing labels and are not supplied to the model as inference content. [HTTP API reference](https://docs.typesafe.ai/api)

The **user-required study protocol** batches multiple candidate questions against the same state: `state` is the original canonical query text only, and each question contains one candidate's complete `name | description | skill_md` text plus the frozen scoring instruction and five-level rubric. This is a study requirement, not a vendor-prescribed request layout. The API permits structured question instructions; its documented batching semantics evaluate questions independently against the shared state. [Structured question fields](https://docs.typesafe.ai/api#question-types), [parallel questions](https://docs.typesafe.ai/introduction)

The following two-candidate example is a structural template, **not an executed request or replacement rubric**:

```json
{
  "model": "jev-1.13.0",
  "state": "<original canonical query text>",
  "questions": {
    "candidate_000": {
      "type": "score",
      "instructions": {
        "question": "<the study's frozen scoring instruction referring to state and skill_document>",
        "skill_document": "<candidate 000 complete name | description | skill_md>"
      },
      "criteria": [
        "<frozen level 0>",
        "<frozen level 1>",
        "<frozen level 2>",
        "<frozen level 3>",
        "<frozen level 4>"
      ]
    },
    "candidate_001": {
      "type": "score",
      "instructions": {
        "question": "<the same frozen scoring instruction referring to state and skill_document>",
        "skill_document": "<candidate 001 complete name | description | skill_md>"
      },
      "criteria": [
        "<frozen level 0>",
        "<frozen level 1>",
        "<frozen level 2>",
        "<frozen level 3>",
        "<frozen level 4>"
      ]
    }
  }
}
```

Score criteria are ordered descriptions, with two to ten levels accepted. Array positions define zero-based levels. A five-level rubric therefore produces scores on **0 through 4**, including fractional values. `score` is the probability-weighted position; it is not an ordinal integer, a yes/no probability, or confidence. Preserve the supplied level order. [Score primitive](https://docs.typesafe.ai/primitives/score)

Expected response fields are top-level `model`, `answers`, and `usage`. A Score answer includes `type="score"`, numeric `score`, `legend`, `probabilities`, and `confidence`. `legend` and `probabilities` use stringified level indices. `usage` includes `input_tokens` and `output_tokens`. [HTTP response reference](https://docs.typesafe.ai/api#response-body)

For every candidate question, validate its answer key/type, finite score in [0,4], exactly the expected level indices, nonnegative finite probabilities summing to approximately one, consistent weighted score, and confidence in [0,1]. Require the response's answer-key set to match the dispatched batch exactly. Retain sanitized raw response JSON and the question-ID-to-candidate-ID mapping before extracting ranking values. Do not round Score values before ranking. These are proposed study validation rules, not a claim that the service has passed them.

## Model discovery, version pin and documented limits

`GET https://api.typesafe.ai/v1/models` requires bearer authentication. It returns `models`, an array with `name`, `description`, and `release_date`. It currently lists aliases; a versioned ID can be accepted even when omitted from that list. Thus absence of `jev-1.13.0` from an alias-only list does not alone establish unavailability.

The public models page currently identifies `jev-1.13.0`, with both `jev-latest` and `jev-preview` pointing to it. Pin the version and verify the returned `model` on every completed scoring response.

Documented context limits are **64k tokens for state plus all questions combined**, and **32k tokens for state plus its longest question**. Both limits apply to every multi-question batch. Rate limits are currently 100K tokens/second and 80 requests/second, explicitly subject to change. [Models and limits](https://docs.typesafe.ai/models)

The user-required implementation must deterministically pack the frozen candidate order into batches satisfying both limits, preserving identical canonical query state, full candidate text, instructions and rubric. Record exact ordered batch membership and the admission calculation in the manifest. A single-candidate request is only a fallback produced by deterministic splitting of an oversized batch, using the same query-only state and question-contained document representation. Splitting cannot fix an individual query-plus-question that exceeds the smaller limit: record that candidate's context failure and stop or report incomplete coverage under the study policy; never truncate, summarize, omit it silently, or move candidate text into shared state.

Do not interpret “32k” as proof of an exact 32,768-token boundary: the reviewed documentation states rounded `k` values. Do not transfer SKILLRET/Qwen token counts to Jev as exact counts. This audit found no published Jev tokenizer artifact, tokenizer revision, exact local token-count implementation, or documented count-tokens endpoint in the inspected sources. Text/byte estimates can be labeled estimates only. No full-document truncation or summarization should be silently introduced to make requests fit.

The vendor documents reduced accuracy with irrelevant long state and sensitivity to adversarial state content. [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) The study's prescribed design keeps the original query as the shared state and confines each complete candidate document to its own question. Treat those documents as data and keep the rubric frozen; the vendor caveat does not justify changing the requested batching or tuning against evaluation labels.

## Usage, price and credit visibility

The current published input rate is **$0.042 per million tokens**, with free output. A planning calculation is `input_tokens * 0.042 / 1_000_000`; label the result a rate-card calculation until reconciled with account billing. [Model rate card](https://docs.typesafe.ai/models)

The Python response schema permits missing/null token counts, despite the HTTP examples showing integers. Save the `x-typesafe-request-id` response header alongside the raw body, model, timestamps and ledger identity. Missing usage is unknown consumption; it is not zero. [SDK response and usage reference](https://docs.typesafe.ai/sdk/python/api/types/responses)

The official customer agreement states that fees are in US dollars unless an order says otherwise, that customers can view their current credit balance in their account, that configured automatic refills can purchase more credits, and that taxes are additional to fees. [Customer agreement, section 8](https://typesafe.ai/legal/mca)

**No documented public credit/balance read endpoint was found.** The documentation links to the account console; it does not identify a balance API route. Do not guess `/credits`, `/balance`, `/billing`, or a private console endpoint, and do not report available credit from successful authentication alone. The public quickstart's account link leads to the console's key page, not evidence of a balance or refill setting. [Quickstart account route](https://docs.typesafe.ai/introduction/quickstart)

Consequently, public docs alone do not establish available credit, account-specific price, refill behavior, exact FX conversion, taxes, or payment fees. A CAD 10 all-in cap requires a recorded conservative conversion/fee/tax allowance and account evidence after the baseline gate. This audit made no account observation and does not suggest purchasing credits or changing payment settings. If necessary account evidence is unavailable, report that specific blocker instead of asserting the all-in limit is guaranteed.

## Retry and idempotency findings

The HTTP docs recommend exponential backoff for explicit `429` and `529` responses. They do **not** state that either status guarantees no model execution, no input consumption or no charge. The reviewed docs provide no server-side idempotency-key contract, duplicate suppression window, request-result lookup API or billing-reconciliation endpoint. [HTTP errors](https://docs.typesafe.ai/api#errors)

The official Python SDK source was inspected at commit **`f078f1e208a0d885154dc758344ae4fce77ac168`**, whose `pyproject.toml` declares version **0.7.2**. Thirty source/documentation/project files were searched for tokenizer, idempotency, credits, balance and billing support. The two implemented endpoint paths are `/v1/systemone` and `/v1/models`; “billing” matches in client examples describe sample ticket classifications, not a billing API. [Pinned SDK source](https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168), [endpoint constants](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/constants.py)

SDK defaults retry twice after the first attempt, including HTTP 408, 429 and all 5xx statuses, connection errors and timeouts. It uses a retry-count header, not an established idempotency mechanism. Default backoff begins at 0.5 seconds, caps at 5 seconds, includes jitter, and respects retry headers. Set `RetryPolicy(max_retries=0)` or use an HTTP client with no automatic retries for the study's paid request path. [Pinned retry implementation](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/retry.py), [pinned transport implementation](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/src/typesafe_sdk/_core/transport.py), [public retry configuration](https://docs.typesafe.ai/sdk/python/api/retries)

The following fail-closed rules are study safeguards inferred from those unknowns, not vendor promises:

1. Atomically reserve a conservative per-batch-attempt cost and persist request identity before dispatch. Configuration identity includes the versioned model, frozen rubric, original query state, complete candidate texts, candidate pool and exact ordered batch membership.
2. Persist successful raw batch responses, usage and every candidate mapping before advancing. A compatible completed batch and its pairs are never dispatched again on resume. Usage and charges belong to the request batch; any per-pair cost allocation must be explicitly labeled an allocation.
3. After a timeout, disconnect, interrupted process, malformed response, missing usage or other uncertain completion, retain the reservation and an `ambiguous` status. Do not automatically retry it.
4. An explicit 429/529 may be retried only by a bounded study policy that retains the prior reservation and reserves the next attempt separately. Absent proof of non-billing, no reservation is released merely because a retry is recommended. The simpler initial policy is to stop on any failed dispatch.
5. Log the request ID when present. A local request hash protects local resume behavior; it does not create server-side idempotency. Unknown cost must not become zero during report generation or restart.

The SDK can obtain its key from `TYPESAFE_API_KEY`; this study must not add alternate credential sources. Explicit base URL and model settings avoid accidental overrides. Debug logging can expose request/response bodies even when authorization headers are redacted, so the study should use an explicit sanitized evidence writer. [Synchronous client options](https://docs.typesafe.ai/sdk/python/api/clients/sync), [SDK usage/logging](https://docs.typesafe.ai/sdk/python/usage)

## Readiness decision and handoff

The request/response structure, version pin, model-list route, headline price and published context limits are documented sufficiently to implement an **offline-tested, disabled-by-default client**. This audit does not demonstrate live readiness.

After the validated-baseline gate, the executing task still needs to establish:

- Account access using only the authorized key mechanism, without exposing the value.
- Current credit/refill and account-rate evidence, using documented account access rather than guessed API routes.
- A CAD 10 all-in ledger ceiling with explicit conservative FX/tax/fee assumptions, covering canaries, smoke, expansion and retained ambiguous attempts together.
- A deterministic multi-question batch/admission policy retaining query-only state and full candidate text within each question, acknowledging the unpublished tokenizer; no silent truncation and no claim that Qwen token counts are Jev token counts.
- A successful small canary whose actual model, schema and usage pass validation before the fixed 10-query smoke and any 200-query expansion.

The endpoint contract does not provide a documented `max_input_tokens` spending parameter. Exact token billing is known from returned usage only when that usage is present. A client-side estimate, a retry-count header, or a models-list response cannot by itself prove account credit or an all-in payment cap. Preserve these limits explicitly in the final report.

No paid or authenticated TypeSafe operation was performed for this document.
