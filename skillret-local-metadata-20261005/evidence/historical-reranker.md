# SkillRet reranker study: Jev and CLM

An executed MIA5150 development study of **metadata-only skill discovery**. The same
BM25 retriever searched 6,006 SkillRet skills and supplied 20 candidates per
query. Jev and CLM independently ranked those candidates; each pipeline returned
five skills. This folder contains the runner code and saved results. It does not
execute downloaded skills.

**Start with the [comparison below](#results-on-the-same-200-queries), then run
the [read-only verifier](#verify-the-published-results).** The earlier
[SkillRet pilot](../skillret-pilot/README.md) contains the Deep Agents,
OpenRouter, and local Ollama selector implementations and their broader
N=50/250/1,000 experiments. This folder adds the Jev and CLM rerankers and
brings their results together with the pilot's five-query BM25 → LLM results.

## Results on the same 200 queries

| Pipeline | Recall@5 | Complete@5 | Change from BM25 |
|---|---:|---:|---:|
| **BM25 → Jev** | **70.92%** | **114/200** | **+10.33 points** |
| BM25 alone | 60.58% | 89/200 | — |
| BGE-small vector retrieval | 49.25% | 67/200 | −11.33 points |
| **BM25 → CLM** | **38.17%** | **51/200** | **−22.42 points** |

Jev exceeded CLM by **32.75 percentage points** on these 200 queries. In the
paired query records, Jev had higher recall on **91** queries, lower recall on
**1**, and the same recall on **108**. All 200 Jev and 200 CLM calls completed.
BM25 candidate Recall@20 was **72.75%** for both rerankers; it measures the
available pool before reducing it to five results.
The [aggregate and per-query comparison CSVs](reference_run/comparison/) are
included alongside the original raw rows.

The Jev and CLM comparison uses identical query IDs and text, dataset revision,
metadata strings, BM25 top-20 IDs and ranks, candidate memberships, and evaluator
relevance labels. The [verification script](scripts/verify_recorded.py) checks
these conditions from saved evidence. BGE-small used the same 200 queries and
capped metadata, but is a different retrieval pipeline; its recorded 512-token
query limit truncated seven queries.

**Metric:** Recall@5 is the fraction of each query's relevant skill IDs found in
the first five, averaged over queries. Complete@5 counts queries where all
relevant IDs were found. A five-result list may contain extra irrelevant IDs;
Complete@5 is therefore different from exact-set selection.

## What the rerankers did

- **Jev:** pinned `jev-1.13.0`; one TypeSafe SystemOne request per query with 20
  independent Noul relevance judgments over the same capped candidate metadata.
  Highest five probabilities win; ties retain BM25 rank. It rescued relevant
  skills from BM25 positions 6–20 on 39 queries and lost a BM25 top-five relevant
  skill on 8 queries. It made 200 requests with no service failures. The
  published-price estimate from observed usage is **US$0.0542**; the service did
  not expose the actual billed dollar amount. The original report is
  [here](reference_run/reports/JEV_TRIAL.md).
- **CLM:** pinned CLM-v0.1-8B with a Qwen3-8B BF16 encoder and vLLM on a
  temporary Runpod NVIDIA A40. The Mac handled preparation, BM25, evaluation,
  tests, and evidence retrieval; CLM inference ran on the GPU pod. It ranked
  the same 20 metadata strings and returned the top five, with zero service
  failures. The pod was verified terminated. The estimated compute charge for
  the completed pod was **US$0.0951**; an invoice was not measured. See the
  [verified CLM report](reference_run/reports/CLM_RUNPOD_TRIAL_verified.md).

The archived CLM report was written before the Jev run was located in this
checkout; its statement that a matched Jev comparison was unavailable is
historical. The paired check and comparison in this README were completed
after both runs using the saved raw data.

These costs describe different services and accounting methods. They are not a
controlled cost comparison. Neither reranker received gold IDs, expected answer
counts, qrels, full skill bodies, or BM25 scores in its model request. A
reranker cannot recover a relevant skill missing from BM25's top 20: at least
one was missing on 80 of the 200 queries.

## Local and OpenRouter LLM comparison

The separate saved **BM25 → LLM** trials used five fixed queries and N=50
gold-preserving libraries. BM25 offered five candidates to a real Deep Agents
selector, which chose zero to five IDs. The numbers below average recall over
**valid model outputs only**; the denominator is shown to expose missing and
invalid cases.

| Model | Runtime | Valid / planned | Recall on valid outputs |
|---|---|---:|---:|
| Qwen 3.5 4B MLX | Local Mac | 4/5 | 100% |
| Qwen 3.5 9B MLX | Local Mac | 5/5 | 80% |
| Qwen 3.8 27B | OpenRouter | 1/5 | 100% |
| Qwen 3.8 Flash | OpenRouter | 4/5 | 87.5% |
| DeepSeek V4.1 Flash | OpenRouter | 5/5 | 90% |
| MiMo V2.6 Pro | OpenRouter | 4/5 | 87.5% |
| GLM 5.3 Flash | OpenRouter | 5/5 | 90% |

Qwen 27B attempted only one BM25 case; four were unrun. The other six models
attempted all five. The [source CSVs](reference_run/llm/) also report precision,
exact-set matches, failures, and the other selection methods. Raw five-query
decisions are included there; selector code remains in
[`skillret-pilot/`](../skillret-pilot/README.md).

**Do not rank Jev against those percentages as if they were the same test.** The
LLM libraries contained all relevant skills by construction, had only 50
skills, and supplied five BM25 candidates rather than 20. Invalid responses
are excluded from their recall means. On those same five query IDs, the saved
Jev and CLM outputs scored 80% and 70% respectively, but their candidate
libraries still differ from the LLM condition. Five requests are too few for
a general LLM ranking.

## Verify the published results

The historical inputs and outputs are under [`reference_run/`](reference_run/).
No API key, remote pod, dataset download, or model weights are needed to check
the recorded comparisons. Its [`SHA256SUMS`](reference_run/SHA256SUMS) records
the exact published evidence bytes:

```bash
python3 scripts/verify_recorded.py
```

For the original Python fixture tests on macOS or Linux, use Python 3.12 and
project-local dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest -q tests/test_jev_reranker.py tests/test_clm_reranker.py tests/test_reference_helpers.py
```

The fixture tests exercise parsers, labels isolation, budgets, failures and
serialization. They do not substitute for the saved live responses. The
read-only verifier compares the latter and reports actual measured aggregates.

## Reproduce a new run

Use the pinned [dataset manifest](manifests/dataset.json), [embedding
manifest](manifests/embedding.json), [200 pilot query IDs](manifests/pilot_query_ids.json),
and [dependency lock](requirements.lock.txt). The dataset and BGE weights are
downloaded separately; the 111 MB skill-body file and model caches are not
committed. From **this directory**:

```bash
.venv/bin/python pilot.py download
.venv/bin/python pilot.py inspect
.venv/bin/python pilot.py baseline --queries 200
.venv/bin/python pilot.py vectors
.venv/bin/python jev_reranker.py prepare
.venv/bin/python scripts/run_clm_trial.py --prepare-only
```

These commands prepare data and candidates locally on the Mac; they do not
start paid inference. For live Jev, privately set `TYPESAFE_API_KEY` and
`ALLOW_PAID_RUNS=1`; use `jev_reranker.py run --queries 200 --live`. For live CLM,
privately set `RUNPOD_API_KEY` and `ALLOW_PAID_RUNS=1`; use
`scripts/run_clm_trial.py --backend auto`. Each runner has its own persistent
budget and call guards. **A fresh clone has no historical paid ledger, so a
new live invocation creates a new billable experiment.** Never copy a key into
the repository or reset a ledger to bypass a limit. CLM requires a temporary
Linux/NVIDIA BF16 pod; it cannot run with the recorded configuration on this
Mac. The pinned remote checkpoint and source are fetched at run time, not
redistributed here.

| File | Role |
|---|---|
| [`jev_reranker.py`](jev_reranker.py), [`jev.defaults.json`](jev.defaults.json) | TypeSafe request, ranking, budgeted run, report |
| [`clm_reranker.py`](clm_reranker.py), [`scripts/run_clm_trial.py`](scripts/run_clm_trial.py) | Local candidate preparation, Runpod lifecycle, CLM evaluation |
| [`scripts/clm_remote_worker.py`](scripts/clm_remote_worker.py) | Pinned GPU-side CLM/vLLM inference |
| `pilot.py`, `trial.py`, `agent_runner.py`, `reference_metrics.py` | Shared pinned retrieval and metrics code from the executed local checkout |
| `prompts/`, `manifests/`, `evidence/` | Model instruction, revision and source pins |
| [`reference_run/`](reference_run/) | Saved raw decisions, summaries, comparison inputs and historical reports |
| `setup_local_key.py` | Optional private key setup helper; never stores keys in version control |

## Scope and limitations

These are **development results** from public SkillRet skills and synthetic
requests, not an untouched final split or production user data. The 200 pilot
queries were previously consumed; the reserved final IDs were not used as
model input or evaluation labels. SkillRet qrels can omit skills that are
semantically useful. The principal result measures metadata-only retrieval,
not task completion or successful execution of a skill. Jev and CLM have
different model interfaces and prompts, so the matched result compares their
saved pipelines rather than isolating architecture. Re-running may differ as
providers, hardware, or dependencies change; no new inference was performed
while publishing this folder.

