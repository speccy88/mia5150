# BGE-M3 dense SkillRet comparison

This study compares metadata (A), exact published Qwen source chunks (B), and
native long-context full text (C). The primary contrast is B versus the immutable
[Qwen3 8B Q4_K_M study](../skillret-qwen3-embedding-8b-q4km-fulltext/REPORT.md).
All 6,006 skill IDs, 4,392 official query IDs, 7,187 qrels and eight historical
candidate libraries are retained. See `results/progress.json` for measured scope
while execution is running; an audited plan is not a completed retrieval result.

## Reproduce inference

Use Linux with a CUDA-capable GPU. The measured machine is the existing Ubuntu
RTX 2060 SUPER 8 GB server. Python 3.12, PyTorch **2.8.0+cu126**, FlagEmbedding
**1.3.5**, Transformers **4.51.3**, and batch size one are pinned. Keep model files,
raw data, temporary installation files and persistent vector caches outside Git.
A disk-backed temporary directory avoids small `/tmp` filesystem quotas.

```bash
python3.12 -m venv "$HOME/.local/opt/mia-bgem3-venv"
source "$HOME/.local/opt/mia-bgem3-venv/bin/activate"
mkdir -p "$HOME/.cache/mia-bgem3/tmp"
export TMPDIR="$HOME/.cache/mia-bgem3/tmp"
"$HOME/.local/opt/mia-bgem3-venv/bin/pip" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$HOME/.local/opt/mia-bgem3-venv/bin/pip" install -r requirements.txt
```

From this study directory:

```bash
python src/acquire.py
OPENBLAS_NUM_THREADS=1 python src/prepare.py
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 python src/feasibility.py
OPENBLAS_NUM_THREADS=1 python src/prepare.py --context 8192 --freeze
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 python src/run.py
OPENBLAS_NUM_THREADS=1 python src/validate.py
OPENBLAS_NUM_THREADS=1 python src/analyze.py
OPENBLAS_NUM_THREADS=1 python src/snapshot.py export
OPENBLAS_NUM_THREADS=1 python src/snapshot.py replay
OPENBLAS_NUM_THREADS=1 python src/report.py
```

Use the **actual selected context** from feasibility.json for the freeze command;
8,192 is the intended maximum, not permission to ignore a failed admission test.
Freezing refuses changed scientific inputs. The convenience FlagEmbedding encode
method enables truncation and automatic batch retries. This implementation uses
BGEM3FlagModel's official dense forward/CLS/normalization with explicit complete
input IDs, CUDA FP16 parameters, memory-efficient SDPA and no CPU fallback.
Backbone loading explicitly selects FP16 with reduced CPU memory use. The actual
input IDs are checked against the pinned tokenizer with truncation disabled.

Documents are exactly `name + " | " + description + " | " + skill_md`, retaining
all original whitespace, except A deliberately excludes the body. B retains the
published Qwen character slices; overlength handling, if ever needed at a lower
admitted context, is recorded rather than silently truncating. C uses exact
source slices, native tokenizer lengths, preferred Markdown boundaries and
128-token overlap. Two CLS/SEP slots are reserved. BGE uses raw queries without
Qwen's instruction prefix. Float32-normalized vectors are scored with CPU cosine,
max aggregation per skill, exact-ID deduplication and the same content-hash tie
rule as Qwen. Skill IDs and gold labels do not enter embeddings.

Every vector cache binds exact input text, model/checkpoint/tokenizer hashes,
frozen protocol, runtime, dimensions, precision and normalization. NPY/JSON
writes are atomic and checksummed. Identical text vectors are shared between
B and C. Completed rankings checkpoint after every query, and completed scaling
conditions resume without replacing historical outcomes. Fresh query repeats
exist specifically to measure uncached latency at the same boundary for every
configuration; equality with existing cached vectors is required.

## Evaluation and uncertainty

All conditions use the exact candidate IDs/order. The seven smaller historical
libraries are not fully nested and place gold skills at positions 1–20. The
6,006-skill condition preserves Qwen's seeded full-catalog order. Scaling repeats the same
20 exposed single-positive questions; the official test has one to three relevant
IDs per question. Both macro and micro candidate coverage are reported. Missing
queries remain in the denominator. Existing tested TREC metrics are reused,
while independent formulas recompute every saved score and coverage total.

The prespecified primary comparison is official NDCG@10 for B minus Qwen.
Question bootstrap intervals use 10,000 resamples and seed 42. Repeated scaling
sizes remain linked within 20 question clusters; official differences pair all
4,392 query IDs. Secondary contrasts and metrics are exploratory without
multiplicity correction. Encoder architecture/training, tokenizer, precision,
dimension, query convention and inference runtime remain different even when
source spans match. The public exposed data do not establish blinded
performance or skill execution quality.

Measured index wall time includes cache reuse/resumption. In particular C's
incremental cache-aware cost is not a fresh standalone index measurement.
Query totals include fresh embedding, input checks, vector persistence and search;
search-only timings are separate. Telemetry samples one second, with CUDA
allocator peaks recorded separately. The monitor stops below 1 GiB available
RAM, above 7,168 MiB VRAM, above 80 °C or after 64 MiB host swap traffic. A
64 MiB threshold avoids attributing trivial whole-host page events to the model.
The runner never stops another workload.

## Reproduce retrieval without embedding

After the completed vector snapshot is published, Git LFS supplies
`vectors/dense-index-v1.npz`. With NumPy installed, run:

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

This checks archive/array hashes, float32 dimensions and norms, every A/B/C
scaling and official ranking, and all metrics. It uses no model, GPU, network
request or original runtime cache during replay. Compression is lossless and
pickle is disabled. No model weights, raw text, private paths or credentials
belong in the publication.

Primary sources: [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3),
[official FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding),
[PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/).
