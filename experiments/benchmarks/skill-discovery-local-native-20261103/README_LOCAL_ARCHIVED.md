# Local skill-discovery suite: historical preparation notes

This suite was prepared independently of the separate OAuth workspace. Its scripts,
results, processes and OAuth credentials were not modified by the local suite.
[README.md](README.md) describes the public layout; [LOCAL_REPORT.md](LOCAL_REPORT.md)
describes the completed initial study. This document retains historical operating
instructions; public sanitized evidence exports cannot resume the original runs.

## Inspect a suite

```powershell
Get-Content .\evidence\live-status.json
Get-Content .\results\local-native-v2\suite-status.json
.\.venv\Scripts\python.exe .\write_native_v2_report.py
```

The last command regenerates the report from available CSV/SQLite files after
independently checking ranks, metrics and persistence. A complete export can include
errors and blocks; measured responses and their denominators remain distinct.
Do not rerun a suite launcher in its existing directory: it deliberately rejects
existing state to prevent replay.

`evidence/AUDIT_VERDICT.md` records sampler compliance and gold-position bias at
positions 1–20. Originals are preserved. Changing the protocol, weights, parameters
or scoring requires a new run/version.

## Reproduce in a new directory

Python 3.12 and PowerShell 7.6.5 were used (7.5+ for `-DateKind String`). The local
engine uses the standard library. `requirements-test.txt` dependencies support only
synthetic OAuth tests retained with the engine; no real login is required.

1. Supply the two validated input files under `data/`. `bench.py` verifies expected
   SHA-256 hashes before any call. The public minimal derivative has the same
   prompt-relevant fields; see the current README and attribution catalogue.
2. Create a local venv and install CUDA runtime with `Install-LocalRuntime.ps1`;
   install conversion tools only if needed. Official llama.cpp b11960 CUDA 12.4 is
   pinned by SHA-256, without global installation.
3. `inspect_sources.py` reads public publisher metadata only. `fetch_official.py REPO FILE`
   downloads a pinned public GGUF; `fetch_batch.py` contains the official selection.
   Transfers verify HTTP ranges and a complete SHA-256 before promoting `.part` files.
4. Prepare models without publisher GGUFs using `prepare_conversion.py converter`,
   then `prepare_conversion.py REPO`. `convert_official_batch.py` uses pinned official
   code offline, CPU PyTorch and a quantizer without CUDA DLLs. These operations
   prepare weights; they are not CPU inference.
5. Review `local-models.json` and set paths to verified weights on the machine.
   Qwen3.5 used installed Ollama files; Ling exactly matches the official file.
   Do not substitute another model with the same name.
6. Use `bench_native_v2.py` and `configs/native-v2/` for corrected final-answer
   parsing. `run_local_suite_v2.py` creates a fresh suite, one model at a time on
   `127.0.0.1:28085`; it stops only its own servers verified by PID/time/executable.
   Unrelated CUDA workloads block startup. `continue_local_suite_v2.py` was used
   for the inspected continuation after the Qwen3-8B diagnostic. It skips complete
   models and rejects incomplete transactional state that could replay a request.

The launcher requires transformer/output layers and KV on CUDA, f16, actual context
32,768, `fit=off`, one slot, no context shift, warmup, RAM cache or checkpoints.
It does not silently reduce context or layers when VRAM is insufficient. Each
server has logs and evidence tied to PID, start time, weight/executable hashes and
a sealed log prefix. Llama 3.2 3B and Gemma 3 4B require manual gated access; no
terms acceptance, login or ungated third-party copy was used to bypass those gates.

## Preserve and integrate

- Original `state.sqlite3` files are transactional authorities; `details.csv` is
  exported atomically with exactly 45 columns. Public SQLite copies are sanitized
  audit exports, explicitly non-resumable.
- `manifest.json` preserves configuration, identities and candidate/prompt plans.
  Resumption must match its fingerprint. Interrupted requests with uncertain
  outcomes are never automatically replayed.
- Original errors remain unchanged. The late 1.7B OOM is documented in
  `evidence/qwen17-runtime-incident.json`; subsequent server-stopped rows are not
  interpreted as measured zero accuracy.
- `.gitignore` excludes weights, archives, tools, environments, conversions and
  private data. No OAuth secret was copied. Results contain questions/names and
  occasional output fragments, reviewed for public release.
- Report/package commands themselves do not push or send messages. Publication
  and delivery are separate actions.

`local-v1/` remains a historical parser diagnostic with non-comparative scores.
Raw v2 responses remain in `private-responses/`, excluded from Git.
`evidence/model-artifact-provenance.json` collects public manifests without weights.
`audit_native_buffers.py` verifies sealed prefixes and budgets.

GPU scope covers transformer/output layers and KV/recurrent states. The input
buffer remains CPU_Mapped by default. The launcher rejects declared CUDA budgets
above physical 8,192 MiB; WDDM can accept allocations exceeding that capacity.
The Qwen3-8B diagnostic preserves 21 separate responses without counting them as
admissible measurements or replaying its interrupted intent.

The 60 exclusions per model at sizes 1,000/2,000/5,000 come from the configured 32K-window
gate. They demonstrate neither native limits nor the VRAM maximum.

## Finalize and package

After resolving all 18 configurations, `finalize_local_handoff.py` performs only
offline checks and generates `evidence/final-handoff.json`, the standalone report
and `HIGH_CONTEXT_PROPOSAL.md`. The latter remains a proposal: no larger-context
load or call is executed by that command.

`package_local_handoff.py` creates archives in `../deliverables/`, verifying ZIP CRCs
and every content SHA-256. Integration archives contain code, tests, configurations,
results, notes and logs, excluding weights, binary tools, environments, private
source data and raw responses. Original SQLite/WAL files are preserved privately.
The interrupted diagnostic also has `state.portable.sqlite3` with unresolved intents.

The separate `PRIVATE-local-response-evidence-*.zip` retains raw responses for local
verification and remains outside GitHub integration and ordinary delivery. Copied
reports, manifests and `SHA256SUMS-*.txt` accompany archives. These commands neither
push nor send messages.
