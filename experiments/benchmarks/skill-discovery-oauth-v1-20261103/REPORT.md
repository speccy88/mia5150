# The three models remain close on this test, with measured coverage through 2,000 skills

Skill-discovery listing study conducted on 3 November 2026 with `gpt-6-luna`,
`gpt-6.1-sol` and `gpt-6.1-terra`. This execution is an **OAuth variant of protocol v1**.
It does not reproduce a temperature-0 API run with a 500-token output limit exactly.

## What the results support

- Each model produced **120 valid responses**, covering sizes 50, 100, 250, 500,
  1,000 and 2,000. Exact top-1 scores range from **72.5% to 75.0%** across those
  six sizes. Paired differences remain compatible with zero with only 20 questions.
- At 5,000 skills, each model received **one HTTP 400 `string_above_max_length`
  rejection**. The other 19 cases at that size were not attempted. A string-length
  rejection neither measures the model's token context limit nor represents 0% accuracy.
- The position audit shows a stronger issue than the potential last-position
  replacement rule: **the 20 gold skills already occupy positions 1–20 at every
  size**. Replacement was never needed. These scores are conditional on the
  expected answers being placed near the beginning of each list.
- Original v1 evidence is retained. Comparability with the earlier cloud study is
  **unverified**, because its original runner and ordered candidate lists were not
  supplied. The OAuth settings are a distinct experimental condition in any case.

## Actual coverage and observed performance

| Model | Responses / planned | Exact top-1, measured sizes | Mean / median latency (s) | Reported input / output tokens |
| --- | --- | --- | --- | --- |
| Luna | 120/140 | 88/120 (73.3%) | 2.872 / 2.743 | 3 444 430 / 11 830 |
| Sol 6.1 | 120/140 | 87/120 (72.5%) | 5.952 / 5.896 | 3 444 430 / 11 812 |
| Terra 6.1 | 120/140 | 90/120 (75.0%) | 3.088 / 2.939 | 3 444 430 / 5 194 |

These percentages use the **six fully measured sizes**, rather than all 140 planned
rows. Latency includes the HTTP round trip and generation, excluding the one-second
pause between requests. Network/service conditions, possible prefix caching and
output length also affect it. It is not an isolated model-hardware measurement.

The three CSVs contain **420 rows and the requested 45 columns**: **360 valid
responses, three rejections and 57 `NOT_RUN_AFTER_MODEL_STOP` rows**. There were
363 benchmark requests and three small admission probes, or 366 inference POSTs
requested from the service. Rejections were not retried. No quota or rate-limit
event was observed. No local GPU benchmark was executed as part of this OAuth study.

Numeric zeros in unattempted rows are CSV structural placeholders, **not
observations**. Those rows are excluded from the scores, intervals and plots below.
No API price was assigned: `est_cost_usd` is blank. Usage was requested through the
consented OAuth connection, without a conventional API key, credit purchase or paid
API fallback. The subscription's actual monetary cost was not measured.

### Exact top-1 by library size

Every measured cell uses the **same 20 questions**, repeated at the next size.
Wilson 95% intervals use **n = 20 for each size**.

| Size | Luna, exact and 95% CI | Sol 6.1, exact and 95% CI | Terra 6.1, exact and 95% CI |
| --- | --- | --- | --- |
| 50 | 17/20 = 85.0% [64.0–94.8%] | 18/20 = 90.0% [69.9–97.2%] | 17/20 = 85.0% [64.0–94.8%] |
| 100 | 16/20 = 80.0% [58.4–91.9%] | 14/20 = 70.0% [48.1–85.5%] | 15/20 = 75.0% [53.1–88.8%] |
| 250 | 15/20 = 75.0% [53.1–88.8%] | 14/20 = 70.0% [48.1–85.5%] | 15/20 = 75.0% [53.1–88.8%] |
| 500 | 14/20 = 70.0% [48.1–85.5%] | 14/20 = 70.0% [48.1–85.5%] | 16/20 = 80.0% [58.4–91.9%] |
| 1000 | 13/20 = 65.0% [43.3–81.9%] | 13/20 = 65.0% [43.3–81.9%] | 13/20 = 65.0% [43.3–81.9%] |
| 2000 | 13/20 = 65.0% [43.3–81.9%] | 14/20 = 70.0% [48.1–85.5%] | 14/20 = 70.0% [48.1–85.5%] |
| 5000 | Not measured: 1 rejection, 19 unattempted | Not measured: 1 rejection, 19 unattempted | Not measured: 1 rejection, 19 unattempted |

![Accuracy and latency by library size](figures/scaling.png)

From 50 to 2,000 skills, observed top-1 falls by 20 percentage points for Luna and
Sol and 15 points for Terra. This describes these questions and candidate lists;
it does not establish a general scaling law. Golds always at the beginning do not
test retrieval of an answer buried in the middle of a long context.

### Coverage, latency and tokens by size and model

Tokens come from the server's `response.completed.usage`. Output usage includes
reasoning where the service counts it. Rejections provide no usable token usage;
missing values are not replaced by estimates.

| Model | N | Valid responses | Rejections | Unattempted | Mean latency (s) | Mean input tokens | Mean output tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 50 | 20/20 | 0 | 0 | 2.408 | 2142.75 | 90.85 |
| Luna | 100 | 20/20 | 0 | 0 | 2.697 | 4236.75 | 90.30 |
| Luna | 250 | 20/20 | 0 | 0 | 2.861 | 11034.75 | 95.90 |
| Luna | 500 | 20/20 | 0 | 0 | 3.061 | 21867.75 | 102.15 |
| Luna | 1000 | 20/20 | 0 | 0 | 3.017 | 43843.75 | 111.35 |
| Luna | 2000 | 20/20 | 0 | 0 | 3.190 | 89095.75 | 100.95 |
| Luna | 5000 | 0/20 | 1 | 19 | — | — | — |
| Sol 6.1 | 50 | 20/20 | 0 | 0 | 4.425 | 2142.75 | 67.40 |
| Sol 6.1 | 100 | 20/20 | 0 | 0 | 5.248 | 4236.75 | 83.95 |
| Sol 6.1 | 250 | 20/20 | 0 | 0 | 5.929 | 11034.75 | 109.95 |
| Sol 6.1 | 500 | 20/20 | 0 | 0 | 6.472 | 21867.75 | 117.95 |
| Sol 6.1 | 1000 | 20/20 | 0 | 0 | 6.130 | 43843.75 | 102.20 |
| Sol 6.1 | 2000 | 20/20 | 0 | 0 | 7.507 | 89095.75 | 109.15 |
| Sol 6.1 | 5000 | 0/20 | 1 | 19 | — | — | — |
| Terra 6.1 | 50 | 20/20 | 0 | 0 | 2.721 | 2142.75 | 26.90 |
| Terra 6.1 | 100 | 20/20 | 0 | 0 | 2.865 | 4236.75 | 32.70 |
| Terra 6.1 | 250 | 20/20 | 0 | 0 | 2.946 | 11034.75 | 42.75 |
| Terra 6.1 | 500 | 20/20 | 0 | 0 | 3.312 | 21867.75 | 47.80 |
| Terra 6.1 | 1000 | 20/20 | 0 | 0 | 3.248 | 43843.75 | 50.20 |
| Terra 6.1 | 2000 | 20/20 | 0 | 0 | 3.437 | 89095.75 | 59.35 |
| Terra 6.1 | 5000 | 0/20 | 1 | 19 | — | — | — |

The 360 valid responses report **10,333,290 input tokens** and **28,836 output
tokens**. Separate admission probes report 429 input tokens and 50 output tokens.
No fabricated zero usage is assigned to the three rejections.

### Ranking quality

| Model | N | R@1 | R@3 | R@5 | R@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 50 | 85.0% | 90.0% | 95.0% | 95.0% | 0.8767 | 0.8943 |
| Luna | 100 | 80.0% | 85.0% | 85.0% | 95.0% | 0.8368 | 0.8624 |
| Luna | 250 | 75.0% | 85.0% | 85.0% | 85.0% | 0.7917 | 0.8065 |
| Luna | 500 | 70.0% | 80.0% | 80.0% | 80.0% | 0.7417 | 0.7565 |
| Luna | 1000 | 65.0% | 70.0% | 75.0% | 85.0% | 0.7030 | 0.7376 |
| Luna | 2000 | 65.0% | 75.0% | 85.0% | 85.0% | 0.7117 | 0.7452 |
| Sol 6.1 | 50 | 90.0% | 95.0% | 100.0% | 100.0% | 0.9267 | 0.9443 |
| Sol 6.1 | 100 | 70.0% | 90.0% | 95.0% | 100.0% | 0.8183 | 0.8633 |
| Sol 6.1 | 250 | 70.0% | 80.0% | 90.0% | 90.0% | 0.7558 | 0.7909 |
| Sol 6.1 | 500 | 70.0% | 80.0% | 80.0% | 85.0% | 0.7488 | 0.7732 |
| Sol 6.1 | 1000 | 65.0% | 70.0% | 75.0% | 80.0% | 0.6931 | 0.7181 |
| Sol 6.1 | 2000 | 70.0% | 70.0% | 75.0% | 80.0% | 0.7183 | 0.7372 |
| Terra 6.1 | 50 | 85.0% | 95.0% | 95.0% | 95.0% | 0.8833 | 0.9000 |
| Terra 6.1 | 100 | 75.0% | 85.0% | 90.0% | 90.0% | 0.8017 | 0.8259 |
| Terra 6.1 | 250 | 75.0% | 80.0% | 85.0% | 95.0% | 0.7981 | 0.8326 |
| Terra 6.1 | 500 | 80.0% | 80.0% | 80.0% | 80.0% | 0.8000 | 0.8000 |
| Terra 6.1 | 1000 | 65.0% | 75.0% | 80.0% | 80.0% | 0.7100 | 0.7324 |
| Terra 6.1 | 2000 | 70.0% | 75.0% | 75.0% | 80.0% | 0.7333 | 0.7494 |

The CSVs contain all metrics at K = 1, 3, 5 and 10. With one gold item, a hit at
rank r ≤ K gives R@K = 1, P@K = 1/K, NDCG@K = 1/log2(r+1), and MRR@K = AP@K = 1/r.
P@10 therefore cannot exceed 0.1; it is not a 10% top-1 selection precision.

| Model | Invalid names in ranked lists, total | Additional top-1 hits from case fallback |
| --- | --- | --- |
| Luna | 30 | 0 |
| Sol 6.1 | 4 | 0 |
| Terra 6.1 | 1 | 0 |

A “valid response” here completes without a technical error and contains at least
one recognized name. Other ranks can still contain invalid names. Invalid slots
and duplicates retain their positions: removing a bad first entry never promotes
a correct second entry to first. Case fallback is tracked separately and added
no top-1 success in this run.

## Uncertainty: 20 questions, not 140 independent observations

A model's 120 valid responses are 20 questions repeated at six sizes. Treating
n = 120 or 140 as independent would ignore that dependence. For the overall mean,
the bootstrap resamples **20 question clusters**, keeping all six sizes and all
three models together within each cluster. Model comparisons use the same
resampled indices: 10,000 replicates, analysis seed 20261103, percentile 95% intervals.

| Model | Mean across six sizes | Question-cluster bootstrap interval, 95% |
| --- | --- | --- |
| Luna | 73.3% | 55.0–90.0% |
| Sol 6.1 | 72.5% | 54.2–90.0% |
| Terra 6.1 | 75.0% | 56.7–90.8% |

| Comparison, left minus right | Observed difference (pp) | Paired 95% interval (pp) |
| --- | --- | --- |
| Luna − Sol 6.1 | 0.83 | -2.50 to 4.17 |
| Luna − Terra 6.1 | -1.67 | -5.83 to 1.67 |
| Sol 6.1 − Terra 6.1 | -2.50 | -8.33 to 1.67 |

All paired intervals include zero. This does not prove equal model performance;
the sample cannot establish a reliable superiority claim. Per-size Wilson
intervals assume distinct Bernoulli cases conditional on the candidate list.
The bootstrap describes sensitivity to these 20 cases with observed lists and
settings held fixed. Neither method accounts for position bias, name leakage,
other library seeds, decoding variability or the full task population. These are
descriptive intervals without multiplicity correction or confirmatory claims.

## Protocol audit and interpretation-changing limitations

### Golds are always at the beginning in this run

Independent reconstruction uses the received record order,
`random.Random(42).sample(all_skills, N)`, then replaces the last item only when
the gold ID is absent. For each model, **all 140 ordered candidate sequences and
logical-message hashes match the recorded run manifests**.

| N, IDs | Distinct names | Golds inserted at the last position | Positions of the 20 golds | Instruction string characters |
| --- | --- | --- | --- | --- |
| 50 | 50 | 0 | 1–20, in question order | 10 295 |
| 100 | 100 | 0 | 1–20, in question order | 20 851 |
| 250 | 250 | 0 | 1–20, in question order | 54 255 |
| 500 | 499 | 0 | 1–20, in question order | 108 801 |
| 1000 | 992 | 0 | 1–20, in question order | 217 958 |
| 2000 | 1973 | 0 | 1–20, in question order | 438 935 |
| 5000 | 4856 | 0 | 1–20, in question order | 1 094 067 |

The received gold IDs also exactly match `random.Random(42).sample(all_skills, 20)`
in question order. Question selection and candidate sampling are therefore
effectively coupled. Last-position replacement remains a potential limitation
of the general protocol, but **it never activated here**. An independent-draw
assumption giving about 0.8% gold inclusion at N=50 does not describe this run.
The observed placement concern is beginning-of-list bias. No list was reordered
after discovering the problem.

### Duplicate names, lexical hints and a single relevant label

There are **6,006 IDs but 5,801 distinct names**, comprising **153 duplicate-name
groups and 358 affected records**. Three of the 20 gold names are nonunique in
the full source. No deduplication was performed. A name-only response cannot
distinguish two IDs with the same name, so these scores evaluate the **expected
name**, not proven UUID identification.

A conservative check finds an exact normalized name phrase in **six of the 20
queries**, ignoring case and separator differences. The other 14 can still contain
lexical hints. Name-guided tasks partly test lexical matching. Only one gold is
provided even when multiple skills might be semantically appropriate, so a
reasonable alternative can be scored wrong. No annotation was changed after
examining the outputs.

### The 5,000-skill rejection measures a transport envelope, not token context

At N=5,000, reconstructed `instructions` contains **1,094,067 characters**
(1,095,016 UTF-8 bytes). All three routes returned `string_above_max_length`.
Retained sanitized responses do not state the field's exact maximum, so no
maximum is invented. `context_limit` remains false. This string-length rejection
cannot establish that the models themselves cannot process the corresponding tokens.

The authenticated OAuth catalog advertised **272,000 tokens for Luna** and
**500,000 for Sol and Terra**, versus a 1,050,000 maximum on the public API cards.
The authenticated limits were used. The v1 gate
`44.5 × N + 200 > 90% of the window` blocked no case: it estimated 222,700 tokens
at N=5,000. That is an estimate, not an actual count of the rejected prompt.
No token-context overflow was measured.

### OAuth differences and cloud comparability

The official OAuth route uses streamed Responses, `store=false`, and the system
text in `instructions`. It does not support imposing `temperature=0` or
`max_output_tokens=500`; those fields are omitted explicitly. Reasoning effort
was not requested; all 360 completed responses report `medium`. No agent tools,
external retrieval or skill execution were used. Descriptions are capped at
300 Unicode characters by the protocol.

Models ran sequentially, with a one-second inter-request delay outside measured
latency. Provider prefix caching is uncontrolled. Different output lengths and
service boundaries prevent interpreting these timings as hardware comparisons
against local GPU models. Model aliases were not replaced by invented dated
versions; actual server-returned identity was audited during execution.

The supplied files and literal sampling recipe are verified, but the reference
cloud runner and ordered candidate lists were unavailable. Its RNG scope, exact
messages, tokenizer, settings and denominators need checking before curves can
be combined. Selection precision is TP / valid queries; selection recall across
all cases is TP / 20 at a fully executed size. Recall is not merely a response
rate: a wrong response is not a true positive. Availability is reported separately.

## Validation and reproduction

- Exact 45-column schema, 140 unique size/question pairs per model, matching golds.
- Field-by-field CSV agreement with transactional SQLite snapshots for all
  363 recorded attempts.
- Independent recalculation of every K metric, token totals and denominators;
  finite nonnegative latencies; unknown monetary costs left blank.
- Independent checks of candidate sequences and all 140 logical-message hashes
  for each model.
- **28 synthetic tests passed**, without new inference. A test exposed a transient
  Windows file lock. The distributed runner adds bounded retries for atomic file
  replacement only, never for an inference POST. Exact measured code is archived.

Normalized rankings, ranks and response hashes are retained. Full HTTP response
text was not saved, so every formatting/suffix normalization cannot be replayed
independently from raw text. Arithmetic validation is not semantic validation
of the gold labels.

The public dataset is a minimal derivative: ID, name and the first 300 description
characters in original order. **All 140 logical prompts remain identical** to
the manifests. The unused full skill bodies from the 117.6 MB original are not
redistributed. Original and derivative hashes are recorded in
[data provenance](data/PROVENANCE.json). The reproduction runner explicitly
accepts the derivative hash; exact measured sources remain in `source-original/`.
Packaging and filesystem retries neither fix v1 sampling nor alter measured results.
The [attribution companion](data/ATTRIBUTION.md) restores source author handles,
repository URLs and received MIT/Apache labels without entering any model prompt.

To reproduce calculations and plots without calling a model:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-oauth.txt -r requirements-analysis.txt
.\.venv\Scripts\python.exe analyze_public.py
```

New measurements require installing OAuth dependencies, connecting one's own
account and running admission before the suite, as described in [README.md](README.md).
That consumes the consenting account's usage and may produce different outputs.
No new run was made for reporting. Credentials, OAuth state, installation IDs,
review links, DPAPI tokens and sensitive logs are excluded.

## Proposed scientific v2, kept separate

1. Use **independent random streams** for question selection, candidates and
   ordering. Shuffle after gold inclusion even when the gold was already present.
   Record multiple seeds and exact positions; optionally balance beginning,
   middle and end placements.
2. Write **more independent tasks without skill names**, diversify domains and
   review labels blind. A pilot of at least 100 cases is a starting point;
   determine final sample size from the desired precision, not a guaranteed-power claim.
3. Annotate **multiple relevant answers where appropriate** and use unambiguous
   IDs. Adapt IR metrics explicitly; distinguish formatting, recognized names,
   availability and semantic relevance.
4. Check **transport limits before large runs** using character, byte and token
   counts and an officially supported input format. Any message splitting or
   role change is a separately named and validated condition. No rejected v1
   request was split or retried.
5. Keep **OAuth and API/local conditions separate**. Record supported controls
   and observed reasoning, and repeat decoding to estimate variability. Do not
   assume all models support temperature, seeds or output caps.
6. **Preregister analysis and retain useful audit records**: output text when
   shareable, effective settings, sanitized errors, order/positions, usage and
   caching when provided. Use question-paired comparisons and distinguish
   sampling uncertainty from systematic bias.

These proposals were not applied retrospectively. Original v1 results remain
intact; a revised scientific protocol requires a separate execution.

## Sources and supporting files

- Supplied `test.jsonl` and `test_queries.json`; original hashes in
  [PROVENANCE.json](data/PROVENANCE.json).
- [Luna results](results/gpt-6-luna/details.csv), [Sol](results/gpt-6.1-sol/details.csv),
  [Terra](results/gpt-6.1-terra/details.csv), [reproducible analysis](results/analysis.json),
  [methodology warning](results/METHODOLOGY_WARNING.json).
- [OpenAI: OAuth subscription usage and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).
- [OpenAI: OAuth route limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).
- API cards: [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
  [Sol 6.1](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
  [Terra 6.1](https://developers.openai.com/api/docs/models/gpt-6.1-terra).
  Observed account-catalog limits are retained separately without account identity.

### Admission probes, excluded from benchmark scores

| Model | Result | Duration (s) | Input / output tokens |
| --- | --- | --- | --- |
| Luna | Admitted, exact response | 2.154 | 143 / 32 |
| Sol 6.1 | Admitted, exact response | 2.010 | 143 / 9 |
| Terra 6.1 | Admitted, exact response | 2.038 | 143 / 9 |
