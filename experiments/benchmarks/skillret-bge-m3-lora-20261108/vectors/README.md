# Lossless adapted indexes and query vectors

`dense-index-v1.npz` is stored through Git LFS. The manifest binds its exact
SHA-256, size, array shapes/dtypes/hashes, base model revision, selected adapter
hash and frozen protocol. Compression preserves every normalized float32 value.
NPZ loading disables pickle. There are no model checkpoints or raw source texts.

The archive includes all three document matrices (6,006 metadata, 6,913 matched
chunks and 6,258 native chunks), chunk ownership and hashes, all 4,392 official
query vectors, the 20 scaling query vectors, exact IDs, full-source hashes and
7,187 public qrels. Adapted vectors are separate from frozen baseline vectors.

Reproduce all 480 scaling and 13,176 official query/configuration outcomes with
no model, GPU, original cache or network request during replay:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch benchmark/bge-m3-lora-20261108 https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='experiments/benchmarks/skillret-bge-m3-lora-20261108/vectors/*.npz'
python3.12 -m venv "$HOME/.local/opt/mia-bge-lora-replay"
source "$HOME/.local/opt/mia-bge-lora-replay/bin/activate"
python -m pip install numpy==2.2.6
OPENBLAS_NUM_THREADS=1 python experiments/benchmarks/skillret-bge-m3-lora-20261108/src/snapshot.py replay
```

The unchanged baseline's ranking and independent scoring helpers are imported
read-only. Its model/vector payload is not required for this replay. Every
ranking must match exactly and every score must agree within 1e-6. All metrics,
including multi-positive denominators and macro/micro coverage, are recomputed.
The saved replay receipt records actual completed scope and measured time.
