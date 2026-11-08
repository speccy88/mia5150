# Lossless dense vectors

The completed publication contains `dense-index-v1.npz` through Git LFS, with a
SHA-256/shape/dtype manifest. It preserves the original float32 document vectors
for A metadata, B matched chunks and C native context, shared official/scaling
query vectors, exact skill/query IDs, chunk ownership, source hashes and binary
qrels. No model weights or raw source text belong in the archive.

From a checkout with NumPy and Git LFS:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch benchmark/bge-m3-fulltext-20261108 https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='experiments/benchmarks/skillret-bge-m3-fulltext-20261108/vectors/*.npz'
cd experiments/benchmarks/skillret-bge-m3-fulltext-20261108
python3.12 -m venv "$HOME/.local/opt/mia-bgem3-replay"
source "$HOME/.local/opt/mia-bgem3-replay/bin/activate"
python -m pip install numpy==2.2.6
OPENBLAS_NUM_THREADS=1 python src/snapshot.py replay
```

The replay requires all 480 scaling and 13,176 official query/configuration
rankings and their metrics to match exactly. Scores use tolerance 1e-6. It makes
zero model/API/network calls and reads no original runtime cache. Lossless ZIP
compression retains every float32 bit, and `allow_pickle=False` is mandatory.
See `results/published_vector_replay.json` for the actual completed replay receipt.
