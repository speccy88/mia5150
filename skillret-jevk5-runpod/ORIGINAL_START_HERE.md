# SkillRet → JevK5 LoRA, on RunPod

**Status: implementation + CPU tests, not a trained 4B model. No Pod has been rented by this package's author.**

Start with **BGE-M3 → 15 skill candidates + NONE → JevK5-4B + a fresh LoRA**. Train a pilot before considering the full official training partition. The aim is to improve the *complete retrieval system*, not to produce a flattering standalone classifier score.

## What you get

- `PROTOCOL.md`: exact task, data split, loss, baselines, metrics, and acceptance rules.
- `HERMES_HANDOFF.md`: instructions for Hermes/Codex, including the paid-resource approval gate.
- `scripts/fetch.py`: pinned official data downloader; training-only by default.
- `scripts/build.py`: split validation, BGE-M3 cache, deterministic top-100 mining, top-15 conversion, source hashes.
- `scripts/train.py`: BF16 backbone, rank-16 LoRA, soft multi-positive loss, checkpoints, resume, validation-only selection.
- `scripts/infer.py` / `scripts/score.py`: label-free inference and independent strict scoring.
- `scripts/pilot_report.py`: estimate the full run from measured pilot timings and a user-supplied CAD hourly rate.
- `scripts/export_results.py`: one portable evidence ZIP before Pod deletion.
- `notebooks/01_data_and_targets.ipynb`, `02_training_and_evaluation.ipynb`: two short explanatory notebooks. GPU commands are shown, not automatically executed.
- `tests/`, `examples/`, `evidence/`: CPU tests, synthetic schema examples, and actual local test results.
- `docs/CLM_LESSONS.md`, `docs/RUNPOD_OPERATIONS.md`, `docs/SOURCES.md`, `docs/IMPLEMENTATION_REVIEW.md`.

**Weights, the 714 MB source dataset repository, and the original CLM raw run are not embedded in this starter ZIP.** Downloaders and immutable source references are supplied. Nothing here claims that 127,190 real qrels have already been converted or that JevK5 has been trained.

## Recommended first GPU

Use a **normal single-GPU Pod**, not Serverless. A **48 GB A40 / RTX A6000 / L40S** is a conservative starting class for BF16 LoRA with reference kernels. This is an engineering recommendation, not a measured minimum. A 24 GB 3090/4090 may work at microbatch 1 with checkpointing, but must pass the pilot; do not silently switch to QLoRA or offload.

Use a current CUDA-enabled PyTorch Linux template, Python 3.11+, enough system RAM (preferably 32 GB+) and roughly 60 GB+ workspace/storage for the environment, two model families, caches, checkpoints and evidence. Those are planning allowances, not measured requirements. Current GPU prices/availability have **not** been checked for this package.

Before renting: agree a **CAD budget**, exact Pod ID tracking, wall-time cap, and an independent cleanup supervisor. The scripts do not provision or terminate Pods. Stopping Python does not stop GPU billing; stopped Pods can still incur storage charges.

## CPU-only verification, no models or GPU

From the extracted project root:

```bash
python -m pytest -q tests
python -m scripts.demo --out workspace/new-synthetic-demo
```

The toy LoRA test verifies the math, frozen parameters, save/reload and resume on a tiny CPU model. It does NOT verify Transformers/PEFT/Qwen3.5 CUDA integration.

## RunPod pilot: execute only after approval

Upload/extract this ZIP on the Pod and enter the project folder:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
export SKILLRET_ENABLE_FLA=0

python -m scripts.fetch --split train --out workspace/raw/train

python -m scripts.build \
  --raw workspace/raw/train \
  --config configs/pilot.json \
  --out workspace/data-pilot \
  --cache workspace/embedding-cache \
  --device cuda

python -m scripts.train \
  --data workspace/data-pilot \
  --config configs/pilot.json \
  --out workspace/runs/pilot-001 \
  --approve-gpu-run
```

To reuse your original CLM query split, supply `--prior-split /path/to/verified_split.json` to **build**. The file must contain `train_query_ids` and `validation_query_ids`. The script checks complete coverage, ID disjointness and exact provenance/text-family overlap. **The original split-ID manifest was not recovered in this session.** Without it, a NEW deterministic family split is generated; its counts will not be advertised as the old 56,950/6,309 split.

Pilot configuration: up to **2,000 train queries**, **256 validation queries**, **100 optimizer updates maximum**, one epoch maximum, microbatch 1, accumulation 16. Therefore up to 1,600 training-query exposures at 100 updates—not necessarily all 2,000. The configured 45-minute process deadline is checked between operations; it is not an external billing watchdog. Interrupted runs retain completed checkpoints and are not marked completed.

Look at `admission.json`, `events.jsonl`, `tokenization_audit.jsonl`, `selection.json`, and `training_summary.json`. The unadapted model is allowed to remain the selected checkpoint.

Validate a fresh adapter reload before trusting the experiment:

```bash
python -m scripts.infer \
  --run workspace/runs/pilot-001 --data workspace/data-pilot \
  --split validation --which selected --out workspace/reload-pilot-001
```

Measure the cost of a larger experiment. Supply the **actual current CAD-equivalent quote**, not a copied historical RunPod price:

```bash
# Set RATE_CAD to your verified current hourly quote before running this command.
python -m scripts.pilot_report \
  --run workspace/runs/pilot-001 --data workspace/data-pilot \
  --hourly-cad "$RATE_CAD" --out workspace/pilot-cost-estimate.json
```

## Full run: separate approval, fresh initialization

Do not automatically continue to this section. Inspect pilot correctness, memory, gradient behavior, token truncation, validation results, reload agreement and measured cost first. Full training can be much more expensive than CLM head-only training.

Adjust the full run's wall-time cap only in a new, approved config if measured timings justify it. `configs/full.json` initially caps the *process* at four hours; that is not an estimate of how long one epoch takes.

```bash
python -m scripts.build \
  --raw workspace/raw/train --config configs/full.json \
  --prior-split workspace/data-pilot/split.json \
  --out workspace/data-full --cache workspace/embedding-cache --device cuda

python -m scripts.train \
  --data workspace/data-full --config configs/full.json \
  --out workspace/runs/full-001 --approve-gpu-run

python -m scripts.infer \
  --run workspace/runs/full-001 --data workspace/data-full \
  --split validation --which selected --out workspace/reload-full-001
```

This starts again from pinned, merged JevK5—not the pilot adapter. It uses the full training fold, the full validation fold, one epoch, and validation at baseline/end of epoch. LoRA architecture, data representation, candidate lists and selection rule remain fixed.

### Resume an interrupted run

Use a new output directory and the **same** config/data:

```bash
python -m scripts.train \
  --data workspace/data-full --config configs/full.json \
  --resume workspace/runs/full-001/checkpoints/step-000050 \
  --out workspace/runs/full-001-resumed --approve-gpu-run
```

Use the actual complete checkpoint path. `training_state.pt` is trusted local pickle data—never load an arbitrary downloaded replacement. RNG and optimizer state are restored; real-GPU restart equivalence still needs validation on the target stack. Preserve the originating run while resuming because a previously selected checkpoint may still be referenced until export.

## Final test: freeze first, then unlock it

Only after selecting the training procedure/checkpoint with validation:

```bash
DATASET_SHA=$(python -c 'import json; print(json.load(open("workspace/raw/train/manifest.json"))["revision"])')
python -m scripts.fetch --split test --revision "$DATASET_SHA" \
  --out workspace/raw/test --allow-test-download

python -m scripts.build --raw workspace/raw/test --train-raw workspace/raw/train \
  --config configs/full.json --out workspace/data-test \
  --cache workspace/embedding-cache --device cuda --allow-final-test

python -m scripts.infer --run workspace/runs/full-001 --data workspace/data-test \
  --split test --which zero --out workspace/test-zero --approve-final-test
python -m scripts.infer --run workspace/runs/full-001 --data workspace/data-test \
  --split test --which selected --out workspace/test-selected --approve-final-test

python -m scripts.score --data workspace/data-test --split test \
  --predictions workspace/test-selected/predictions.jsonl \
  --zero-predictions workspace/test-zero/predictions.jsonl \
  --out workspace/test-report --approve-final-test
```

For a resumed run, substitute its final run path consistently. Do not change the model after looking at these test scores and call the same test unseen again.

## Export and end billing

```bash
python -m scripts.export_results \
  --run workspace/runs/full-001 \
  --data workspace/data-full --data workspace/data-test \
  --extra workspace/reload-full-001 --extra workspace/test-zero \
  --extra workspace/test-selected --extra workspace/test-report \
  --out workspace/SkillRet_JevK5_Results.zip
```

Include pilot/originating resume runs with additional `--run` arguments when applicable. Copy the ZIP off the Pod, verify the archive/hash locally, **then terminate the exact experiment Pod** and inspect RunPod to confirm compute is gone. Check any remaining volumes/storage separately. Do not terminate unrelated Pods. Preserve the latest viable checkpoint even when a run fails.

## Optional Plumb comparison

`configs/plumb_optional.json` pins merged Plumb-4B. It uses the same architecture/interface, candidate files and training schedule. Run it only after the JevK5 pilot, a separate approval and a compatibility smoke test. Do not train both automatically. Keep separate output directories and report both favorable and unfavorable results.
