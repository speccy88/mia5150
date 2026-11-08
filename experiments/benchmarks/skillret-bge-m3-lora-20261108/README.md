# BGE-M3 LoRA on SkillRet

Separate experiment using the immutable completed
[BGE-M3 baseline](../skillret-bge-m3-fulltext-20261108/REPORT.md).
The measured GPU is the existing Ubuntu RTX 2060 SUPER 8 GB server.
Read `results/training_summary.json` and `results/progress.json` for actual
completed scope; a prepared pipeline is not a successful training result.

The base checkpoint and dataset revisions match the baseline. LoRA targets
encoder attention query/value projections, preserving CLS pooling and normalized
dense retrieval. The base weights remain FP16 and frozen. FP32 adapters and
optimizer moments, dynamic gradient scaling and nonreentrant gradient
checkpointing make training practical without a CPU fallback.

Only the pinned training split supplies supervision and negative mining.
Entire shared-ID/source-URL/normalized-body groups stay together in a seeded
training-derived validation split. Shared test identities and lexical query
near-duplicates are quarantined using a prespecified audit. The audit reads
test text for overlap only, never test qrels or scores for training or selection.
See the saved overlap audit for exclusions and the limits of lexical similarity.

Training uses one query, one selected positive source passage and one negative
per physical group. Random and frozen-dense hard negatives alternate; every
known positive skill is excluded. Accumulation over 16 groups adds no in-batch
negatives. Positives and negatives both use the first explicit source passage,
including metadata and the beginning of the body. Passage selection is recorded, with 384-token training
windows and 256-token query windows. Complete final test representations retain
the baseline's exact A/B/C source spans and 8,192-token inference capacity.

Validation is a fixed 128-query heldout retrieval proxy using one bounded first
passage per heldout skill. It is deliberately cheaper than full-source A/B/C.
The best trained adapter is selected by its validation NDCG@10, with earliest
ties. Official test scores never select checkpoints. Smoke updates are discarded
before main training. A 200-update pilot precedes a gated continuation of at most
two eligible-query passes, with early stopping.

## Reproduce

Published results are immutable measurements. For a new training run, use a
separate checkout and move this study's existing `results/` directory into the
external run cache first; acquisition recreates an empty result directory. Keep
the published manifests/protocol for input verification. The default recipe is
the balanced lead-passage recovery; the archived first recipe is evidence rather
than the default training recipe. Vector-only replay needs none of this training.

Use Python 3.12 in a separate environment. Install the pinned CUDA 12.6 PyTorch
wheel from the official PyTorch index, then this study's requirements. The
recorded environment was copied into a separate virtual environment from the
frozen baseline installation; package files are independent and the baseline
environment is unchanged. `runtime-lock.txt` records exact installed versions.
Keep all base weights, raw splits, temporary files and recovery states outside Git.

From this directory, after making the baseline's pinned model/data cache
available as described in its README:

```bash
python3.12 -m venv "$HOME/.local/opt/mia-bge-lora-venv"
source "$HOME/.local/opt/mia-bge-lora-venv/bin/activate"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export SKILLRET_LORA_CACHE="$HOME/.cache/mia-bge-lora-lead"
mkdir -p "$SKILLRET_LORA_CACHE/tmp"
export TMPDIR="$SKILLRET_LORA_CACHE/tmp"
git lfs pull --include='experiments/benchmarks/skillret-bge-m3-fulltext-20261108/vectors/*.npz'
python src/acquire.py
python src/prepare_data.py
python -m pytest -q tests
export HF_HUB_OFFLINE=1
python src/train.py
python src/run.py
python src/validate.py
python src/analyze.py
python src/snapshot.py export
python src/snapshot.py replay
python src/report.py
```

Acquisition contacts Hugging Face explicitly for the pinned training files and
validates Git/LFS object hashes. Inference/training uses local pinned weights.
Model caches are read-only reused through symlinks; adapters and vectors have
separate identities and storage. Recovery checkpoints include adapter weights,
AdamW/scheduler/scaler states, positions and random-generator states outside Git.
Never substitute base vectors for adapter vectors.

After an interrupted main run, recover an existing checkpoint explicitly:

```bash
python src/train.py --resume "$SKILLRET_LORA_CACHE/checkpoints/step-00100"
```

Recovery archives later unfinished journal rows, restores the saved adapter,
optimizer, scheduler, scaler and RNG states, then continues the seeded query
order. It does not repeat the discarded smoke or completed mining.

The initial 1e-4 lexical-positive pilot completed 200 updates but failed its
validation gate (best NDCG@10 0.418016 versus frozen 0.802766). It was preserved
under `results/attempts/01-lexical-positive-pilot/` and received no official
evaluation. The recovery uses matching lead-passage policies and a 1e-5 peak
learning rate, with validation every 50 steps. Both changes were declared using
training-derived validation before any official test evaluation; their separate
effects are not identified. Frozen train-only mining and validation references
are reused. The recipe's actual continuation/stop status is in its saved summary.

The existing official baseline vectors/rankings are reused without rerunning
baseline evaluation. New frozen inference is limited to numeric parity checks,
training-only hard-negative mining and the training-derived validation reference.

Primary references: [official BGE-M3](https://huggingface.co/BAAI/bge-m3),
[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding),
[PEFT dtype guidance](https://huggingface.co/docs/peft/developer_guides/troubleshooting).
