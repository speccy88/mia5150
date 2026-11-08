# Reproduce the published BGE-M3 studies from main

The completed baseline and LoRA studies are published on `main`. The original
experiment branches were not published because the repository restricts branch
creation. The study directories retain their historical bytes, including the
original proposed branch names in archived setup instructions. Use these current
commands for both studies.

The lossless archives include every A/B/C document vector, all official and
scaling query vectors, exact IDs, chunk ownership, source hashes and public
relevance judgments. Each study reproduces 480 scaling cases and 13,176 official
query/configuration cases. Replay uses NumPy 2.2.6 and the saved vectors, with no
model, GPU, original embedding cache or network calls during replay.

## Download the indexed skills and queries

Install Git LFS and Python 3.12, then run from a fresh directory:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch main https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='experiments/benchmarks/skillret-bge-m3-fulltext-20261108/vectors/*.npz,experiments/benchmarks/skillret-bge-m3-lora-20261108/vectors/*.npz' --exclude=''
python3.12 -m venv "$HOME/.local/opt/mia-bge-replay-venv"
source "$HOME/.local/opt/mia-bge-replay-venv/bin/activate"
python -m pip install numpy==2.2.6
```

The baseline archive is 83,421,213 bytes, SHA-256
`18aac5dd024193b0b7f1b43075f8c81d4e750bf3296d63c2c8881beccfbda438`.
The LoRA archive is 83,392,201 bytes, SHA-256
`27f1463f1bce2f7d2f213decf0dc038e1a61f2922b1f4b8a321f46a7fdf5e91d`.
Each archive has an array-level shape, dtype and checksum manifest beside it.

## Replay both complete evaluations

From the cloned repository root:

```bash
OPENBLAS_NUM_THREADS=1 python experiments/benchmarks/skillret-bge-m3-fulltext-20261108/src/snapshot.py replay
OPENBLAS_NUM_THREADS=1 python experiments/benchmarks/skillret-bge-m3-lora-20261108/src/snapshot.py replay
```

The commands verify archive/array hashes, normalized float32 vectors, every
saved ranking and score, and independently recomputed metrics. They write fresh
timing receipts to `results/published_vector_replay.json` in the new checkout.
The baseline and adapter studies remain separate throughout replay.

## Download the selected and final LoRA adapters

```bash
git lfs pull --include='experiments/benchmarks/skillret-bge-m3-lora-20261108/adapters/**/*.safetensors' --exclude=''
```

Both adapters contain only the 786,432 FP32 LoRA parameters. Their configs pin
the base model revision; checksums and tensor validation accompany the study.
Base model weights and raw source corpora are kept outside Git. The selected
adapter worsened official retrieval despite improving the training-derived
validation proxy; consult the paired report before using its results.

- [Baseline report](../../experiments/benchmarks/skillret-bge-m3-fulltext-20261108/REPORT.md)
- [LoRA report](../../experiments/benchmarks/skillret-bge-m3-lora-20261108/REPORT.md)
- [Adapter provenance](../../experiments/benchmarks/skillret-bge-m3-lora-20261108/adapters/README.md)
