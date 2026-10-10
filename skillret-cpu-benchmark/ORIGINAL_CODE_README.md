# SkillRet benchmark code

Compact source-only reproduction package for the CPU SkillRet short-header benchmark. It contains the actual executed preparation, download, retrieval and native-adapter code unchanged, lightweight tests, analysis/report sources, pinned public manifests and direct dependency requirements. No model weights, caches, dataset records, raw results, credentials or private Library identifiers are included.

## Privacy correction before any new model process

The newly provided `setup_privacy.sh` sets `ORT_DISABLE_TELEMETRY=1` before a new process can initialize ONNX Runtime. Source it before optional model commands:

```sh
. ./setup_privacy.sh
```

This is a **prospective correction, not the measured configuration**. The executed scorer/adapters and their recorded hashes remain unchanged. The setup was syntax-checked only; no ORT initialization, inference, denial retry or alternative execution was performed. It does not authorize resuming the stopped run. See [the privacy addendum](docs/PRIVACY_ADDENDUM.md) for official ONNX Runtime1.29/1.30 documentation and the final common32 evidence boundary.

## Safe starting point: tests, then optional BM25

Run from the extracted project root. Linux x86-64 with Python3.12 was used; Windows support is not asserted. The first commands run synthetic checks only, with no model downloads or inference.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 nbformat==5.11.1
.venv/bin/python -m unittest discover -s tests -p test_notebook.py -v
.venv/bin/python src/test_decision_adapters.py

# Optional: download and verify the pinned public dataset, then lexical BM25.
python3.12 src/prepare_dataset.py
.venv/bin/python src/run_retrieval.py bm25

# Independent result audit: read-only; run after preparation and a finished run.
python3.12 tests/audit_skillret_recovery.py
```

`prepare_dataset.py` creates the frozen64 query inputs, their fixed first32 timing prefix, all6,006 skill headers and deterministic 10/100/1,000/6,006 catalog memberships. It verifies pinned source sizes and hashes. BM25 uses no neural model weights. Run audits only after inference/timing has stopped, so the audit does not compete with CPU measurements.

## Optional heavier model runs

See [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md) for all seven exact repository revisions, download command and the three isolated environments. Julia requires Transformers5.0.0; GLiNER requires4.57.6. The pinned official Julia package must be installed from `./models/julia-1`, never an absolute path from a saved pip freeze.

After downloads, complete BGE retrieval before native reranking because it writes the shared shortlist. Run serially:

```sh
.venv/bin/python src/run_retrieval.py bge-small
.venv/bin/python src/run_retrieval.py e5-small
.venv/bin/python src/run_retrieval.py embeddinggemma-int8

.venv-julia/bin/python src/run_decision.py julia quality 32
.venv-gliner/bin/python src/run_decision.py gliner quality 32
.venv/bin/python src/run_decision.py verdict quality 32
.venv/bin/python src/run_decision.py laya quality 32

.venv-julia/bin/python src/run_decision.py julia timing 32
.venv-gliner/bin/python src/run_decision.py gliner timing 32
.venv/bin/python src/run_decision.py verdict timing 32
.venv/bin/python src/run_decision.py laya timing 32
```

To reproduce the preregistered48-query extension, use the same four quality commands with48 instead of32 after deciding a runtime budget. The analysis selects the largest complete shared64/48/32 cohort; it never mixes a larger individual model run with another model's smaller sample. Matched latency stays at the same32 queries ×2 repeats, full catalog only. Larger inference is optional and may take hours.

The exact `src/controller.py` is included for run-history inspection only. It contains a fixed2 October2026 deadline and external checkpoint-acknowledgement gates. It is not a general launcher: do not run it as the default reproduction command. The only newly provided setup is the prospective environment-variable file; it does not launch a process and was not part of the measured run. The ongoing measured run was not changed for this package.

## Analysis and limits

After completed fresh results exist and inference has stopped:

```sh
.venv/bin/python src/build_notebook.py --python-execute
```

This independently verifies raw rankings/qrels, all nine relevance metrics, catalog membership, exact cohorts, native score ordering and matched timing. It executes ordinary cells in-process with explicit validation disclosure, not a Jupyter kernel. HTML/PDF builders are optional presentation tools; HTML uses the canonical Data Analytics plugin renderer, and PDF uses ReportLab. Those optional presentation dependencies are not required to run BM25 or read CSV results.

The original4,392-query measurements were completed and independently audited before their underlying raw artifacts became unavailable. Their retained historical aggregate observations are separate from fresh recovery64/48/32 data. This code archive includes neither old aggregate tables nor new measurements; no raw historical output is reconstructed. It reproduces inputs and new runs, not the lost historical per-query records.

Documents are name plus the first1,024 description characters, not full skill instructions. Small catalogs are per-query gold-preserving controlled samples. Negative binary-adapter results apply to that fixed configuration/shortlist, not general model capability. Query-bootstrap intervals describe query sampling only. See `protocol/DATA_PROTOCOL.md`, `docs/RUNTIME_NOTES.md`, and `docs/FULL_REPRODUCTION.md` for details; the latter is the broader run documentation, so result/report files it names are intentionally absent from this code-only package.

## What was checked for this archive

All included Python sources were syntax-parsed without executing their module bodies; JSON manifests were parsed; import targets and README script paths were checked; every copied executed source was SHA256-compared against the live file; ZIP integrity and the no-data/no-private-identifier allowlist were checked. This packaging check does not claim a clean-room dependency install or an end-to-end inference rerun. `CODE_MANIFEST.json` records exact included file hashes and source status.

### Optional report integration test

The code-only package intentionally omits `docs/retained_observations.json` and all result tables. Consequently, `tests/test_report.py::test_historical_values_are_labeled` requires the full results archive and cannot pass from this code-only package alone. Do not interpret the lightweight notebook/adapter checks above as a clean-room pass of every integration test. The missing historical file must not be fabricated to satisfy that test.
