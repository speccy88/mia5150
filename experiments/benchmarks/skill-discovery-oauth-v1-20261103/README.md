# Skill discovery OAuth v1, 2026-11-03

Read [REPORT.md](REPORT.md) before comparing model scores. The run is position biased:
all 20 gold IDs occupy candidate positions 1–20 at every library size. Each model has
120 completed responses, one HTTP 400 string-length rejection at N=5000, and 19
explicitly unattempted cases. None of those 20 cases is a measured zero-accuracy result.
Comparability with the original cloud runner is unverified.

## Offline reproduction of calculations

Use Python 3.12. No credentials or network requests are needed for analysis/tests.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-oauth.txt -r requirements-analysis.txt
.\.venv\Scripts\python.exe bench.py validate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`analyze_results.py` also cross-checks the original local SQLite checkpoints, which are
intentionally excluded from the public package. To recompute public CSV statistics and
the figure without SQLite or credentials, use:

```powershell
.\.venv\Scripts\python.exe analyze_public.py
```

The complete source dataset is not included. `data/test.jsonl` keeps only ID, name and
the first 300 description characters, preserving source order. Every logical prompt is
identical to the measured prompt. See `data/PROVENANCE.json` for original and distributed
SHA-256 values and the verified equivalence. No names were deduplicated.

The reproduction runner changes only the accepted dataset checksum for this minimal
file and adds bounded retries for transient atomic file-replacement locks. It never
retries a model request. The exact measured runner is retained under `source-original/`;
its implementation hashes match the result manifests. A rerun of that exact version
requires the original dataset bytes whose hash is documented.

## New measurements, only when explicitly intended

New model calls consume the consenting account's usage. The recorded context windows
are observations from the account catalog on the study date, not universal entitlements.
Check the current catalog and supported parameters before a new study. The current
OAuth route omits temperature and max_output_tokens; this is not strict API equivalence.

```powershell
.\Setup.ps1 -InstallOAuth
.\Connect-ChatGPT.ps1
.\.venv\Scripts\python.exe oauth_chatgpt.py models
.\.venv\Scripts\python.exe admission.py --out results-new
.\.venv\Scripts\python.exe run_oauth_suite.py --out results-new
```

The user completes browser consent. Do not reuse another app's credentials. Do not rerun
admission or a stopped call automatically. Requests are sequential, checkpointed, and
stopped on model errors; shared quota failures stop the whole suite. SQLite is the local
transactional authority and CSV output is replaced atomically. Never commit runtime
`private/`, `.venv/`, fresh credentials, review URLs, host IDs, or local logs.

The packaged `results/` are historical measured evidence, so select a new folder for
new calls. No new measurements were made during report and package preparation.

## Local Windows preparation, unexecuted

`Install-LocalRuntime.ps1` and `Download-Models.ps1` prepare pinned llama.cpp CUDA
and Qwen GGUF artifacts. They have not been run for this study. Downloads and new
measurements must be initiated deliberately. `Start-LocalServer.ps1` requires the
explicit server executable and model file paths, uses the selected JSON config,
and checks full GPU layer offload. Hardware OOM stops the run; CPU fallback is
rejected. Then `Run-Local.ps1 -Config configs/local-qwen3-06b.json -Out results-new-local`
runs the same v1 trials. Adjust the named config and model for the 4B option.
The synthetic tests verify controls; they do not establish model fit or GPU speed.

## Contents and integration

- `results/*/details.csv`: unchanged 45-column v1 exports.
- `results/summary_by_size.csv`, `analysis.json`, protocol audit and stop statuses.
- `data/`: minimal prompt-equivalent dataset and exact 20 received queries.
- `source-original/`: exact measured implementation.
- Source, configuration and synthetic tests; safe validation summary and file hashes.

Integrate this folder under `experiments/benchmarks/skill-discovery-oauth-v1-20261103`
and update the repository experiment index. This folder was prepared locally only;
no GitHub publication or email delivery has occurred.
