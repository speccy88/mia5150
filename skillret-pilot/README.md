# SkillRet skill-discovery pilot

A small, executed MIA5150 feasibility study: retrieve candidate skills, let a real
LangChain Deep Agents model select skill IDs, and measure selection quality and
cost as the library grows. It **does not execute downloaded skills**.

Start with the [three-page PDF](reference_run/MIA5150_SkillRet_Short_Report.pdf)
or the [recorded benchmark](reference_run/OPENROUTER_SCALING.md).

## What has actually been measured

| Experiment | Observed result | Denominator |
|---|---|---|
| Offline, full-document BM25 | Recall@5 **51.67%** | 200 fixed queries; 6,006 skills |
| Five-model OpenRouter scaling trial | **121 valid selections; 24 exact matches** | 180 planned cases; 138 attempted; 53 reused cases |
| Separate native activation checks | **16 actual SKILL.md reads** | 25 cases; 15 completed workflows |

The scaling trial used three synthetic requests, four methods, and libraries of
50, 250 and 1,000 skills. Provider failures and invalid IDs reduced coverage.
These results do not establish a general model ranking or skill-execution success.
The recorded expansion cost US$0.3379; the entire recorded project cost US$0.3865
(estimated CAD$0.5797 using a **planning assumption** of 1.50 CAD/USD).

`reference_run/` is historical evidence, including the original sanitized handoff
ZIP. New runs write to ignored `results/`, `reports/` and `evidence/` directories.
The historical budget ledger is **not** installed as your new run's ledger.

## 1. Install locally

Tested with **Python 3.12 on macOS**. The budget lock uses POSIX `fcntl`; use
macOS or Linux/WSL, not native Windows. No Docker or GPU is needed for retrieval.
Linux/WSL has not been validated in this handoff.

```bash
git clone https://github.com/speccy88/mia5150.git
cd mia5150/skillret-pilot
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Alternatively, use `uv venv --python python3.12 .venv` followed by
`uv pip sync --python .venv/bin/python requirements.lock.txt`.
The dependency lock is from the tested experiment; changing it creates a different
protocol. Dependencies, dataset files and embedding weights need local disk space.
The three dataset JSONL files total about 123 MB; they are downloaded separately.

All following commands run **inside `skillret-pilot/`**. Nothing requires a global
Python install beyond an available Python 3.12 interpreter.

## 2. Run tests and the real offline baseline first

```bash
# Tests use native Deep Agents plus controlled model/HTTP fixtures. No paid calls.
.venv/bin/python scripts/verify.py

# Download the exact revision in manifests/dataset.json and verify its checksums.
.venv/bin/python pilot.py download
.venv/bin/python pilot.py inspect
.venv/bin/python pilot.py baseline --queries 200

# Real native middleware and effective-tool checks, with scripted model control.
.venv/bin/python pilot.py offline-check

# Download the pinned BGE embedding model and run CPU metadata retrieval.
.venv/bin/python pilot.py vectors
.venv/bin/python pilot.py activation-smoke
```

The baseline command prints the observed metrics and saves raw rankings. The
recorded reference is `recall_at_5 = 0.5166666666666666`, on 200 queries and 6,006
skills. Repeated identical baseline commands reuse your own saved result. A mock
test passing is not a new benchmark result.

`vectors` uses the pinned BGE-small model, not a hosted embedding API. It compares
BM25 and vectors on the **same capped metadata**. That comparison has different
inputs from the full-document baseline and must be reported separately.

## 3. Choose how to test a language model

### Option A: installed local Ollama models

Start your existing Ollama service at `http://127.0.0.1:11434` and inspect
`ollama list`. This pilot's local allowlist contains `qwen3.5:4b-mlx` and
`qwen3.5:9b-mlx`, the models installed for the original experiment. The runner
does not download models or silently substitute another name.

```bash
.venv/bin/python pilot.py ollama-smoke --model qwen3.5:4b-mlx
.venv/bin/python pilot.py ollama-run --model qwen3.5:4b-mlx
.venv/bin/python pilot.py ollama-report
```

Use `qwen3.5:9b-mlx` for the other installed model, or omit `--model` to test both.
An absent model is a blocker. Supporting another local model requires an explicit
allowlist/configuration change and a new canary. Local inference has a separate
100-attempt cap and 90-second request timeout. There are no OpenRouter charges.

### Option B: OpenRouter with your own local key

Run this **in your own terminal**, not in a chat or notebook:

```bash
python3 setup_local_key.py
```

It hides key input, creates a private ignored `.env`, and asks whether you authorize
paid calls. It refuses to overwrite an existing file. If one already exists,
edit it privately. Both `OPENROUTER_API_KEY` and `ALLOW_PAID_RUNS=1` are required.
Never paste a key into a chat, commit it, or print `.env`.

The five explicit configurations are in `manifests/configs/`:

| Runner label | OpenRouter model | Pinned provider |
|---|---|---|
| `qwen38-27b` | `qwen/qwen3.8-27b` | `darkbloom/fp4` |
| `qwen38-flash` | `qwen/qwen3.8-flash` | `alibaba` |
| `deepseek41-flash` | `deepseek/deepseek-v4.1-flash` | `inference-net` |
| `mimo-v2-6-pro` | `xiaomi/mimo-v2.6-pro` | `deepinfra/fp8` |
| `glm53-flash` | `z-ai/glm-5.3-flash` | `relace` |

These are recorded configurations, not a promise of current availability. Each
live run refreshes public endpoint metadata and requires a native canary. GLM
uses low reasoning because its recorded endpoint required it; the others request
none. No model or provider fallback is allowed.

Start with one canary, then choose the bounded comparison:

```bash
.venv/bin/python scripts/run_openrouter_models.py --stage canaries --models deepseek41-flash

# Latest protocol: 3 requests x 3 library sizes x 4 methods x 5 models.
# This runs/reuses each model's canary before any SkillRet comparison.
.venv/bin/python scripts/benchmark_openrouter.py --live

# Reports and audit read saved evidence; they do not make model calls.
.venv/bin/python scripts/report_scaling.py
.venv/bin/python scripts/audit_scaling.py

# Optional separate disclosure check; shares the same remaining budget.
.venv/bin/python scripts/run_openrouter_models.py --stage activation
```

For the earlier five-request, N=50-only study instead, use
`scripts/run_openrouter_models.py --stage trial`. A one-model trial can add
`--models deepseek41-flash`. The scaling script always tests all five models.
Do not stack every optional experiment expecting all to fit: **all paid stages
share the same US$3 / CAD$5 ceilings and 300-request cap**.

The historical default in `pilot.defaults.json` is `qwen/qwen3-8b`; the five-model
scripts above explicitly override it. `scripts/resume.sh` is the original
single-model workflow, not the five-model entry point.

## 4. Read outputs, failures and costs

| Location | Purpose |
|---|---|
| `results/budget.jsonl` | Persistent paid reservations and settled USD costs |
| `results/openrouter/scaling/` | Main raw selections, failures and router records |
| `results/openrouter/<label>/` | Per-model canaries, N=50 tests and activation |
| `results/ollama/` | Local inference results and independent request ledger |
| `reports/` | Generated Markdown, JSON, CSVs and figures |
| `evidence/traces/` | Sanitized outgoing requests and returned responses |
| `manifests/` | Dataset/model pins, exact query IDs and run/catalogue identities |

Exit **0** means the command's completion criterion passed. Live commands use
exit **2** for blocked, partial or invalid stages; inspect the saved reason before
continuing. Pytest preserves its own failure status. Reports do not turn missing
cases into successful empty answers.

- `unknown_ids`: keep the invalid output; do not correct it or repeatedly sample.
- Provider errors: the scaling run pauses that provider's remaining cases.
- `selector_prerequisite_failed`: domain-vector is gated on earlier valid outputs;
  this makes its conditional score subject to selection bias.
- `unresolved_reservation`: stop paid work until billing is reconciled from actual
  evidence. See `scripts/reconcile_rejection.py --help`; never declare an ambiguous
  timeout free of charge.
- `budget_limit` / `request_cap`: the bounded run stops. **Never delete or reset a
  ledger to get more calls**, including by making a fresh copy for the same study.

One teammate's clone has its own local budget; this does not cap other clones or
unrelated account usage. Coordinate a shared API account separately. No new paid
experiment was performed while preparing this GitHub folder.

## 5. Open the two notebooks

Run the offline preparation above first, then choose this `.venv` as the notebook
kernel in your editor. To execute both noninteractively:

```bash
.venv/bin/python scripts/execute_notebooks.py
```

1. `01_data_and_retrieval.ipynb`: inspect data and reproduce retrieval metrics.
2. `02_deepagents_trial.ipynb`: native isolation/read checks and saved result tables.

The notebook runner clears model-call opt-ins in its child kernels. Interactive
paid cells require `ALLOW_PAID_NOTEBOOK_RUNS=1` **in addition to** the normal key,
permission and budget guard. Interactive local inference requires
`ALLOW_LOCAL_NOTEBOOK_RUNS=1`. A fresh checkout shows unrun stages as unrun, not as
the reference scores. Historical executed notebooks are inside the reference ZIP.

## Code map and method

See [ARCHITECTURE.md](ARCHITECTURE.md) for the flow diagram and important boundaries.

| File | Responsibility |
|---|---|
| `pilot.py` | Small CLI, pinned download, inspection and BM25 |
| `reference_metrics.py` | Metric definitions and nested library sampling |
| `agent_runner.py` | Real Deep Agents graph, tool isolation and paid budget guard |
| `trial.py` | CPU vectors, canary, selection, routing and activation orchestration |
| `local_runner.py` | Separate Ollama adapter, limits and reporting |
| `scripts/benchmark_openrouter.py` | Three-query, five-model scaling scheduler |
| `scripts/report_*.py`, `handoff.py` | Read saved results, report and package |
| `scripts/make_short_report.py` | Rebuild the fixed historical PDF from its saved run |
| `tests/` | Metrics, native tools, budget/retry behavior and wire-format checks |

Comments explain why labels are withheld, candidate-only state is used and unknown
costs retain reservations. Formatting/comments change source hashes, so this GitHub
edition starts a new run identity. The original code, exact histories and failures
remain intact in the [reference archive](reference_run/README.md).

The skills and synthetic requests are untrusted dataset content. Never execute
their instructions, install their dependencies, or follow embedded links. Dataset
metadata/queries are Apache-2.0; source skill attribution is retained in the archive.
See [SOURCE_NOTES.md](SOURCE_NOTES.md) for pinned sources and protocol differences.
