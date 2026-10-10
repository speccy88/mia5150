# MIA5150 local SkillRet laboratory

Read `REPORT.md` for measured results, `IMPLEMENTATION_MAP.md` for methodological scope, and `EXECUTION_STATUS.md` for completed versus blocked runs. Open `demo.html` and `demo-lora.html` for the classroom comparison. The viewers need no network or model server. Both improvements and regressions are selectable.

The main result is negative: equal fusion was essentially flat on all4,392 public queries, and the small hard-negative LoRA development gain did not transfer. Neither paired interval establishes a reliable improvement. Corrected replay and full-body retrieval encountered resource guards; partial work is not a completed benchmark result.

This is a standalone implementation at `/home/user/code/mia5150-local` in Ubuntu-26.04 WSL2 on Elite-Windows11. It is not a checkout of `speccy88/mia5150`; no fork files, branches, remotes, or pushes were changed. Historical evidence from that fork is pinned separately under `evidence/`. The official SkillRet source is a separate read-only clone pinned in the environment receipt.

## Evidence and claims

- `results/`: raw rankings, metrics, training losses, candidate scores, failures and timing. Different query sets must not be merged into one leaderboard.
- `manifests/`: exact input hashes, query IDs, candidate pools, train/development groups, and nested libraries.
- `runs/`: exact command, source snapshot, source hashes, worker log, telemetry and success/guard-stop receipt for every guarded run. Failed attempts are retained.
- `checkpoints/`: reversible local adapters and resume receipts; base model weights were not modified.
- `evidence/`: original-plan integrity, source/model cards, historical exposure, environment, tests and methodological references.
- `analysis.ipynb`: thin analysis notebook. `training_demo.ipynb`: thin training/evaluation notebook. They do not silently launch downloads or training.

The corrected dataset has 10,123 training skills / 63,259 training queries, and 6,006 test skills / 4,392 test queries. Revision: `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`. The 128-query screen and 32-query reranking subset are public development data. The 32-query LoRA evaluation is a different training-derived development set, grouped by repository/name with duplicate query exclusions. Released task-trained checkpoints may already have seen those public training examples. All-query public scores are not blind confirmation: historical full-test aggregates were previously consumed.

## Reproduce on this host

Keep `.wslconfig` at memory=6GB, processors=2, swap=2GB and autoMemoryReclaim=dropCache. Do not install a Linux NVIDIA driver. Stop any other GPU experiment first. The resource monitor fails closed at host or guest free RAM below2GiB, GPU free memory below1GiB, temperature above80C, stale host telemetry, or any new guest swap activity.

Start the host telemetry loop in a PowerShell terminal. It uses native CIM and does not change execution policy:

```powershell
$telemetryPath = 'C:\Users\user\Documents\Codex\2026-10-05\task-2\host.jsonl'
while ($true) {
  $osState = Get-CimInstance Win32_OperatingSystem
  @{ utc=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0; available_bytes=[long]$osState.FreePhysicalMemory*1024; total_bytes=[long]$osState.TotalVisibleMemorySize*1024 } | ConvertTo-Json -Compress | Add-Content $telemetryPath
  Start-Sleep -Seconds 2
}
```

In Ubuntu, use the isolated environment and run one guarded job at a time:

```bash
cd /home/user/code/mia5150-local
.venv/bin/python -m pytest -q
python3 reclaim.py
python3 guard.py NEW_UNIQUE_RUN .venv/bin/python rerank.py skillret_rerank_06 skillret_embed_06-header-0.json header 32
```

For heavy GPU stages, replace `guard.py` with `gate_run.py`. This waits up to180seconds for additional startup headroom (host4.5GiB, guest4GiB, idle GPU) before launching the same continuous guard. It never lowers the2GiB/1GiB reserves. Small FP16 weight shards are an optional local loading optimization; original model files remain intact and conversion manifests record the derivation.

Never reuse a run directory. Before rerunning a script that writes an existing result filename, copy the project to a fresh experiment directory or change the output name; preserve the delivered raw evidence. The stage queues refuse existing failed runs or explicitly print their blocked status. A guard-stop means the attempted run did not pass, even if partial predictions exist.

The convenience command `.venv/bin/python reproduce.py reranker --name my_reproduction --run` creates a new isolated output root, copies the required candidate artifacts and source, and links the existing input data/models/environment. It does not overwrite delivered results. Choices are `bm25`, `embedding`, `reranker` and `lora-hard`. Omit `--run` to prepare and inspect the command. Do not invoke downloaders or model converters inside that linked reproduction root.

`run_sequence.py`, `pipeline_remaining.py` and `pipeline_extra.py` document the executed phase ordering. Use their individual commands for a selective reproduction; do not blindly rerun the whole sequence over historical outputs. `lora_pilot.py hard` and `lora_pilot.py replay-clean` perform bounded actual training, not a simulation. The older replay run was superseded because capping the shuffled examples omitted eight labelled positive pairs; its source snapshot and result remain for audit. Read the fixed caps before running.

## Portable package

The ZIP includes source, tests, notebooks, reports, manifests, raw results, failed-run evidence and checkpoint metadata. It excludes base model weights, bulk datasets, the Python environment, generated vector caches, credentials and Git metadata. Restore the exact public data/models with `fetch_data.py`, `fetch_models.py`, `fetch_sra.py` and `fetch_qwen8.py`, using the pinned metadata and resource guard. Verify the SHA256 manifests after downloading. Install the exact `requirements-lock.txt` into a fresh `uv` Python3.12 virtual environment.

The GGUF experiments additionally require the pinned llama.cpp b11425 CUDA12.8 runtime, its dependent CUDA runtime libraries, and the separately cached JevK5/Plumb/Qwen3.5 files identified in `evidence/cached-assets/`. Their exact server commands are saved under `evidence/`; adjust the absolute local paths when moving machines. Native JevK5 Python scoring code and its license are bundled under `vendor/`. Standard Python/transformers scripts are relative to the project through `MIA5150_ROOT`; the Windows host-monitor path can be set through `MIA_HOST_TELEMETRY`.

No paid API, RunPod, new account, credential extraction, model upload, external publication or Git push was used. Jina's reviewed custom source is explicitly pinned and offline; its CC BY-NC4.0 restriction limits that experiment to noncommercial use. Other licenses and precise provenance are retained with the corresponding model metadata.


Package exclusions: all base/model/adapter weights and optimizer binaries remain local and are excluded from the ZIP. Local checkpoint location: `/home/user/code/mia5150-local/checkpoints`; see `evidence/local-checkpoint-inventory.json` for hashes. No Library transfer was performed. `EXECUTION_STATUS.md` is authoritative for tested versus blocked/unrun components.
