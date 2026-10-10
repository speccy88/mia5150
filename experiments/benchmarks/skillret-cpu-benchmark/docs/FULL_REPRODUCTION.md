# SkillRet CPU benchmark: fresh recovery and historical observations

This package contains a reproducible, CPU-only short-header adaptation of SkillRet v1.1. It compares BM25, BGE-small, multilingual E5-small, EmbeddingGemma INT8, and four native binary-relevance adapters: Julia-1, GLiNER2.5-small, Verdict ONNX and Laya multilingual. Both Qwen entries are excluded.

## Read first

- `artifacts/report.html`: technical report, when built
- `artifacts/analysis.ipynb`: executable analysis companion; inspect its visible execution status
- `docs/notebook_validation.md`: analysis checks and execution boundary
- `protocol/DATA_PROTOCOL.md` and `data/processed/manifest.json`: exact data/cohort/catalog design and hashes
- `docs/model_sources.json`: immutable model/runtime revisions and native-adapter notes

The original 4,392-query measurements were completed and independently audited before their working tree became unavailable. `docs/retained_observations.json` preserves genuine aggregate readouts, but their original raw rankings, per-query metrics, timing rows and pair scores are unavailable. They remain a separate historical section, cannot be recomputed from this package, and are never used to invent raw rows or bootstrap intervals.

Fresh recovery evaluates the same first 64 outcome-independent frozen queries at catalog sizes 10, 100, 1,000 and 6,006. The primary comparison requires every model and condition to be complete on the identical cohort; otherwise it uses the all-model first48, then first32 fallback. The first48 prefix was preregistered before new extension-quality outcomes, based on measured remaining runtime. At the timing gate, the controller permits64 only if its measured forecast leaves a 15-minute buffer before11:52 UTC; otherwise it targets48. Individual larger completions stay separate. Fresh latency targets the same 32 queries × 2 repetitions at 6,006 skills only. Missing or deadline-stopped conditions remain missing. The authorized run had an inference cutoff of 11:52 UTC and delivery cutoff of 12:12 UTC on 2 October 2026; reproduction commands below are for a new, deliberately initiated run.

## Privacy correction for prospective runs

ONNX Runtime 1.29+ supports POSIX telemetry. Before a new Python model process, source `./setup_privacy.sh` to set the officially documented `ORT_DISABLE_TELEMETRY=1` before initialization. This newly added opt-out was **not** configured in the measured runs. Actual scorer/adapters and measured provenance remain unchanged; no inference was rerun. See [the privacy addendum](PRIVACY_ADDENDUM.md) and its official sources. The final benchmark uses common32 after inference stopped during Verdict48; Julia/GLiNER48 are supplements.

```sh
. ./setup_privacy.sh
```

## Minimal environment setup

The observed platform uses Python 3.12 and Linux x86-64 CPU wheels. Create three separate environments because GLiNER requires Transformers 4.x and Julia requires Transformers 5.0.x. These direct dependency pins reflect the measured runtimes, but are not complete cross-platform/transitive lockfiles. `docs/requirements-*-observed.txt` are environment inventories only. In particular, Julia's observed freeze can contain an absolute local path and must not be used as a portable install command.

```sh
python3.12 -m venv .venv
python3.12 -m venv .venv-gliner
python3.12 -m venv .venv-julia
.venv/bin/python -m pip install -r requirements-main.txt
.venv-gliner/bin/python -m pip install -r requirements-gliner.txt
.venv-julia/bin/python -m pip install -r requirements-julia.txt
```

No GPU/CUDA package or model substitution is required. The runner fixes four CPU threads and affinity where supported. CPU instruction support and available wheels may differ on another host; do not silently switch precision or checkpoints to make installation work.

## Recreate pinned inputs and download models

Run from the project root. Preparation uses the standard library and checks pinned download sizes/hashes. Model download uses exact revisions in `docs/model_sources.json`, writes local file checksums in `docs/download_receipts.json`, and includes Julia's official Python runtime. Review checkpoint licenses in the source manifest, including Gemma's terms.

```sh
python3.12 src/prepare_dataset.py
.venv/bin/python src/download_models.py
.venv-julia/bin/python -m pip install --no-deps ./models/julia-1
```

`python3.12 src/prepare_dataset.py --offline` works when the pinned raw source downloads are already present. Model weights, virtual environments, and caches may be omitted from the delivery archive; the download and preparation scripts recreate them. The analysis of included fresh raw measurements does not need model weights.

The dataset is pinned to `ThakiCloud/SKILLRET` revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. Semantic documents are `name + '\n' + description[:1024]`; full skill instructions are excluded. Smaller catalogs are deterministic nested, per-query, gold-preserving samples, not one global deployed catalog. Candidate membership must be lexicographically sorted before scoring. Query gold labels and query domain/major metadata are evaluation-only. JSONL must be read by file iteration, not Unicode-aware `splitlines()`.

## Run a fresh benchmark serially

Do not run models, analysis, or package compression concurrently with latency measurement. BGE retrieval must finish before any native-adapter stage because it writes the shared frozen shortlist.

```sh
.venv/bin/python src/run_retrieval.py bm25
.venv/bin/python src/run_retrieval.py bge-small
.venv/bin/python src/run_retrieval.py e5-small
.venv/bin/python src/run_retrieval.py embeddinggemma-int8

# Preserve common32 before any extension; finish matched timing next.
.venv-julia/bin/python src/run_decision.py julia quality 32
.venv-gliner/bin/python src/run_decision.py gliner quality 32
.venv/bin/python src/run_decision.py verdict quality 32
.venv/bin/python src/run_decision.py laya quality 32

# Independent fresh request timing; quality score caches are not used.
.venv-julia/bin/python src/run_decision.py julia timing 32
.venv-gliner/bin/python src/run_decision.py gliner timing 32
.venv/bin/python src/run_decision.py verdict timing 32
.venv/bin/python src/run_decision.py laya timing 32


# Choose48 or64 using the predeclared measured-runtime gate, before extension.
N=48  # Use64 only when the forecast preserves the 15-minute cutoff buffer.
.venv-julia/bin/python src/run_decision.py julia quality "$N"
.venv-gliner/bin/python src/run_decision.py gliner quality "$N"
.venv/bin/python src/run_decision.py verdict quality "$N"
.venv/bin/python src/run_decision.py laya quality "$N"
```

Quality caches are fingerprint-bound to source/configuration/data. Do not edit adapters, runtime settings or model manifests during a run, and do not reuse a mismatching score cache. The existing results are evidence; reproduce into a separate extracted copy rather than overwriting them.

## Analyze without inference

After every inference/timing process has stopped:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_notebook.py' -v
.venv/bin/python src/build_notebook.py --python-execute
```

The notebook independently verifies qrels, hashes, frozen IDs, candidate membership, all nine relevance metrics, exact common cohorts, fresh shortlists and matched timing. It executes ordinary Python cells sequentially in-process with captured output. No Jupyter kernel is started; this is explicitly disclosed rather than claiming kernel-based validation. `--build-only` produces an unexecuted scaffold and no fresh findings.

A successful analysis also writes `docs/notebook_paired_bootstrap.json`. Its 2,000-resample, seed-20261002 paired query-bootstrap intervals use fresh per-query differences only. They are descriptive, unadjusted query-sampling intervals, not training/hardware uncertainty or out-of-domain guarantees.

`src/build_report.py` rebuilds the HTML report when the bundled Data Analytics canonical renderer is available. Its renderer is an installed plugin dependency, not a pip package; a delivered HTML report opens directly without it. The notebook and raw CSV/JSON files remain independently readable when that renderer is unavailable.

## Output map and interpretation

- `results/retrieval64/`: fresh 64-query retrieval quality and full-catalog 32×2 timing
- `results/quality32/`, `results/quality48/`, `results/quality64/`: native-adapter quality checkpoints
- `results/timing32/`: native end-to-end full-catalog 32×2 timing
- `results/rerank_shortlists.jsonl`: shared BGE+BM25 RRF shortlist, up to20 candidates
- `results/pair_scores/`: fingerprinted native accuracy scores and token/truncation audits
- `*_run.json`: completed configuration/hardware receipts; a growing CSV without a valid receipt is not a completed result

MRR@5, Recall@1/5, Hit@1/5, binary nDCG@5, Complete@5 and candidate recall/any-hit are query-macro metrics. Candidate-any-hit bounds achievable MRR within the fixed shortlist. Warm latency includes fresh retrieval and uncached reranking; load, index/corpus encoding, cached accuracy work and batch throughput are separate. Serial QPS is reciprocal mean latency, not concurrent serving capacity. Process peak RSS is not model-resident memory.

Negative reranking results concern the fixed binary adapters, truncation/calibration choices and shared shortlist, not general model capabilities. Synthetic English queries, short headers, incomplete relevance judgments, the small fresh sample, one CPU host, and lack of downstream skill execution limit generalization. Historical fast ONNX settings differ from the expanded/recovery spinning-off runtime, so their timings are never pooled.
