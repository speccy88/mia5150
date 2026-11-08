# Lossless public retrieval index

`index-v1.npz` is a Git LFS object containing the actual float32 vectors used by
the completed Qwen3 full-text benchmark. [manifest.json](manifest.json) records
its SHA-256, byte size, model/protocol fingerprints, and each array's dimensions,
dtype and uncompressed-byte checksum. Model weights and raw document/query text
are not part of the archive.

The archive contains 6,913 document-chunk vectors, their 6,006 original skill
IDs and chunk ownership, 4,392 official query vectors, 20 scaling query vectors,
source-text hashes and all 7,187 official binary relevance pairs. IDs retain
duplicate names as separate skills. Each vector has 4,096 dimensions. ZIP
compression preserves every float32 bit, and NumPy loads with `allow_pickle=False`.

From the repository root, with Git LFS installed:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext/vectors/*.npz'
cd experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext
python -m pip install numpy==2.5.4
OPENBLAS_NUM_THREADS=1 python src/vector_snapshot.py replay
```

The replay uses only this archive and the study's public JSON evidence. It
recomputes max-chunk retrieval over every exact candidate pool, reproduces all
160 scaling and 4,392 official top-50 rankings, and reconciles all metrics,
including independently calculated macro/micro candidate coverage. It makes
zero model/API/network calls and does not use the original local vector cache.
Exact ID rankings are required; score comparison tolerance is 1e-6.

The generation command, for the original completed local cache, is
`python src/vector_snapshot.py export`. The optional metadata ablation and
uncheckpointed failure probes are not indexed retrieval conditions and are not
included. A replay proves the published retrieval evidence; it does not rerun
embedding inference or reproduce its GPU timing.
