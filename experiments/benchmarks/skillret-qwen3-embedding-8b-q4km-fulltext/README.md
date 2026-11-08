# Qwen3 Embedding 8B Q4_K_M full-text SkillRet benchmark

Read [REPORT.md](REPORT.md) for measured results and the current execution status.
This separate study retains the entire original `skill_md`, uses 4,096-dimensional
float32-normalized embeddings through the local Ollama `/api/embed` endpoint,
and evaluates exact skill IDs. Historical experiments are unchanged.

The required scaling panel has 20 questions at 50/100/250/500/1,000/2,000/5,000/6,006
skills. The separate official test has 4,392 questions and 7,187 binary qrels.
The existing seven libraries retain their saved IDs and order. The new full-size
condition uses seed 42. The historical gold-at-positions-1–20 bias remains visible.
The saved historical pools are not fully nested, so size comparisons also change
candidate membership. Retrieval restricts the accumulating full-corpus index to
each condition's exact eligible IDs.

The 4,096-token context uses a 4,094-token content budget (two admission slots reserved by Ollama), 128-token overlap and
preferred Markdown boundaries. Complete short documents get one vector; long
documents retain every source character across exact source slices. Retrieval
takes the maximum chunk cosine per skill and deduplicates before ranking.
The protocol is intentionally distinct from an unchunked 32K reproduction.

## Reproduce locally

To repeat **retrieval and evaluation without any embedding**, hydrate the
published vector snapshot through Git LFS, then run:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/speccy88/mia5150.git
cd mia5150
git lfs pull --include='experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext/vectors/*.npz'
cd experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext
python -m pip install numpy==2.5.4
OPENBLAS_NUM_THREADS=1 python src/vector_snapshot.py replay
```

Only NumPy is required for this replay. It needs no Ollama, model
weights, original vector cache, raw corpus download, GPU or network request.
It verifies archive and array checksums, every vector's float32 shape/norm,
all 160 scaling rankings, all 4,392 official rankings, and every metric against
the original measured outputs. The snapshot includes all 6,913 document-chunk
vectors with their 6,006 skill IDs, all 4,392 official query vectors, all 20
scaling query vectors, and the 7,187 public relevance pairs. Compression is
lossless; no float16 conversion, rounding or dimension reduction is applied.
See [vectors/README.md](vectors/README.md) and its checksum manifest.

The following procedure reproduces embedding inference as well:

Keep downloaded data, model weights, environments and vector caches outside Git.
Python 3.12 and the packages in [requirements.txt](requirements.txt) are sufficient.
The measured runtime is pinned in [runtime-lock.txt](runtime-lock.txt).

```bash
python3.12 -m venv "$HOME/.local/opt/mia-qwen-venv"
"$HOME/.local/opt/mia-qwen-venv/bin/pip" install -r requirements.txt
```

Install the official Linux Ollama **0.40.15** distribution, or unpack it in a user-owned
directory. On the measured server it runs only on the loopback interface, with
one loaded model and one concurrent inference. Start it in a separate terminal:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_NO_CLOUD=1 \
ollama serve
```

Then, from this study directory:

```bash
ollama pull qwen3-embedding:8b-q4_K_M
python src/prepare.py
OPENBLAS_NUM_THREADS=1 python src/run.py --stage feasibility
OPENBLAS_NUM_THREADS=1 python src/run.py --stage full
OPENBLAS_NUM_THREADS=1 python src/validate.py
OPENBLAS_NUM_THREADS=1 python src/cache_replay.py
OPENBLAS_NUM_THREADS=1 python src/report.py
python -m pytest -q tests
```

`--stage full` completes scaling first, checkpointing each completed size, then
runs all official queries. `--stage scaling` stops after the scaling panel.
The default asset/cache directory is `Path.home() / '.cache/mia-qwen'`; each
command accepts `--cache` except the report builder, which reads public results.
Pinned downloads fail on a SHA-256 mismatch. Preparing an existing experiment
also refuses changed scientific inputs in a frozen protocol. A later publication
checkout preserves the original recorded source-revision receipt; every actual
execution additionally records the exact source-file hashes it ran.

Every embedding cache entry binds exact text hashes, model digest, complete
protocol/chunking configuration and runtime. Atomic NPY/JSON checkpoints retain
float32 shape, vector hashes and unit-norm checks. A corrupt checkpoint fails
validation. Interrupted or unjournaled embeddings may be recomputed; completed
valid entries are reused. Query text and relevance labels never enter document
embeddings together. Only the original query plus the fixed instruction enters
query embedding.

The pinned HF tokenizer supplies the original-length audit. The GGUF tokenizer
can group source control-token spellings differently. Any inference token count
other than the normal audit length plus EOS is checked against the local native
tokenizer on the entire unchanged input before checkpointing. The retained
parity-stop audit documents this compatibility check; it is not a truncation.

The one-second resource monitor records RAM, swap and GPU memory/utilization/
temperature. Guards stop new inference if available RAM falls below 768 MiB,
swap activity increases, GPU use exceeds 7,680 MiB or temperature exceeds 80 °C.
Preflight must show no competing compute workload. The runner never stops a
different workload. On a guard failure, preserve its receipt and unload only the
benchmark's model before reassessing safe settings.

## Evidence and validation

- [protocol.json](protocol.json): frozen data pins, text transform, context,
  chunking, inference settings, ranking and uncertainty definitions.
- [results/input_audit.json](results/input_audit.json) and
  [document manifest](results/documents.jsonl): original lengths, complete
  source/chunk hashes, exact spans and planned text coverage.
- [candidate pools](results/candidate_pools.json): every eligible ID in order,
  saved-source provenance, gold presence and pool hashes.
- `results/scaling-n*.json`, scaling CSV/JSONL and full rankings/evaluation:
  actual outcomes, saved unique-ID rankings and explicit denominators.
- `results/embedding_calls.jsonl`, telemetry, indexing history and latency CSV:
  uncached inference, cache reuse and resource/performance boundaries.
- [figures/](figures/): reproducible 300-dpi PNG and vector SVG, with sources and
  byte hashes in the figure manifest. Missing measurements remain unmeasured.

The production metric helper is imported from the existing tested
`skillret-embedding-jev-fusion/evaluate.py`. The new validator independently
recomputes saved-ranking precision, recall, MAP, NDCG, MRR and exact-ID hits.
Offline fixtures test Unicode coverage, token limits, max-chunk deduplication,
out-of-pool exclusions, multi-positive denominators, question-cluster bootstrap,
API truncation settings and cache integrity. Fixture results are not benchmarks.

After the complete official evaluation, `cache_replay.py` validates and assembles
the saved document vectors, reloads every saved query vector and reproduces all
4,392 official rankings without a model request. Its separately timed cache
reads and searches measure cached execution without relabeling it as inference.
The operating-system file cache is not evicted for this replay; its file-read
timings do not represent cold storage.

No paid inference service is used. Raw full-text data, model weights, secrets and
personal paths stay outside the repository. The requested lossless vector snapshot
is published through Git LFS; the original per-text runtime cache stays local.

Primary references: [Ollama model entry](https://ollama.com/library/qwen3-embedding:8b-q4_K_M),
[embedding API](https://docs.ollama.com/api/embed),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B), and
[pinned SkillRet data](https://huggingface.co/datasets/ThakiCloud/SKILLRET/tree/6583d7d2ed07644d0fb8938ed8178f3a7dc42a12).
