# SkillRet / JevK5 LoRA pilot on RunPod

This folder records a small model-adaptation test completed on **2 October
2026**: does a new SkillRet LoRA adapter improve JevK5's ranking of the same
BGE-M3 skill shortlist? On 256 validation queries, Recall@5 increased from
**47.33% to 50.98%**, compared with **43.03%** for BGE-M3 alone. These are
validation results from one seed. The validation set also selected the adapter;
there was no official test-set evaluation or full-dataset training.

The code, two notebooks, configurations, selected adapter, and compact recorded
evidence were imported from the supplied `SkillRet-JevK5-RunPod` drive folder.
The integration checks are offline. They do not rerun model inference or rent
any resources.

## What the test did

1. Loaded the **training split** of `ThakiCloud/SKILLRET`, revision
   `a050ad233a504a43135bafe8cdf45574052b5729`: 10,123 skills, 63,259 queries,
   and 127,190 positive query–skill judgments. Multiple positive judgments
   belong to the same query; they are not independent training prompts.
2. Grouped queries connected by original ID or exact normalized current/original
   text, then assigned those families deterministically with seed 42. The
   resulting folds contain **56,883 training queries and 6,376 validation
   queries**. This checks exact/provenance overlap, not every semantic paraphrase.
   Training and validation share the official training skill catalog.
3. Selected **2,000 training query groups and 256 validation queries** by a
   deterministic hash. BGE-M3 searched the complete 10,123-skill training catalog
   using `name: description`, unit-normalized FP32 dense embeddings, and a
   512-token limit. It saved top-100 IDs; the decision task used the **natural
   top 15**, without adding missing gold skills.
4. Presented those 15 skills plus **NONE** in a deterministic query-specific
   shuffle. JevK5 scored the 16 answer-letter logits at the final prompt token,
   without generating an answer. Query text and skill metadata are inputs;
   relevance labels and skill IDs are used outside the prompt for supervision
   and grading.
5. Attached a fresh rank-16 LoRA to the already merged JevK5 model and trained
   with categorical cross-entropy against uniform mass over all relevant
   shortlist options. If none was relevant, the target was NONE. Only the new
   LoRA tensors were trained; the backbone stayed frozen.
6. Compared BGE-M3, unadapted JevK5, and the adapted JevK5 on identical validation
   queries and candidates. Checkpoint selection used Recall@5, then MRR@15;
   the unadapted baseline was eligible to win.

The representation is metadata only, rather than the official paper's full
`name | description | skill_md` protocol. This experiment also uses a different
dataset revision and evaluation scope from other studies in this repository.
Their reported scores are not a controlled comparison with this pilot. The
complete original specification is [PROTOCOL.md](PROTOCOL.md).

## Recorded results

All rows cover the same **256 queries**. Recall divides by every labelled
relevant skill, including skills missing from the shortlist, and averages over
queries. MRR is truncated at 15. Ranking removes NONE; abstention is measured
separately. False abstention is a fraction of all 256 queries where the model
chose NONE despite a relevant candidate being present.

| Condition | Recall@5 | MRR@15 | Hit@1 | False abstention |
| --- | ---: | ---: | ---: | ---: |
| BGE-M3 retrieval order | 0.430339 | 0.576059 | 0.484375 | 0.000000 |
| Unadapted JevK5 | 0.473307 | 0.616340 | 0.519531 | 0.011719 |
| Step-50 LoRA, original incomplete run | 0.505859 | 0.730219 | 0.691406 | 0.078125 |
| Selected step-100 LoRA, completed resume | **0.509766** | **0.735584** | **0.695312** | 0.058594 |

The step-100 gain is **3.646 percentage points** over unadapted JevK5, with a
paired query-bootstrap 95% interval of **[2.083, 5.469] points**. Relative to
BGE-M3, the gain is **7.943 points**, interval **[5.273, 10.742]**. The saved
bootstrap uses 2,000 resamples, seed 42. These intervals do not correct family
dependence or checkpoint-selection bias and do not measure variation across
training seeds.

Candidate Recall@15 and oracle Recall@5 are both **0.514974**. The selected
ranker is close to that shortlist ceiling, so retrieval coverage limits further
recall. It still falsely abstained on **15/256** queries, versus **3/256** for
unadapted JevK5. Better forced ranking does not establish better autonomous
action selection. No skills were executed.

The unrounded summary is in
[reference_run/results/summary.json](reference_run/results/summary.json).
Predictions, logits, option mappings, complete validation qrels, and per-query
metrics are retained so these results can be audited without model weights or
downloads.

## Why there are two runs

The first process hit its **2,700-second limit at update 67**. Its last complete
adapter/optimizer/RNG checkpoint was step 50. Its failure record and step-50
validation observations are preserved; it was not a completed 100-update run.

The second process resumed that complete step-50 checkpoint, with an explicitly
recorded **7,200-second administrative allowance**. Model, data, optimizer, seed,
and the 100-update target were unchanged. The source patch added `--wall-seconds`
without changing the checkpoint contract. Updates after the saved step-50
checkpoint had to be repeated. The completed schedule represents 1,600 training
query exposures; that is not 2,000 unique training queries or total work across
both attempts.

Step 100 won selection. A fresh-process reload of the selected adapter reproduced
all **256 rankings**, with maximum absolute logit difference **0.0**. The
[reload receipt](reference_run/reload/reload_check.json) is backed by both saved
prediction files, which the offline verifier compares directly.

## Model, training, and compute

- JevK5: `alibiserikbay/JevK5` at
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d`.
- BGE-M3: `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181`.
- JevK5 runtime: version 0.3.0, Git commit
  `6c6522fe5462a05fdb82bceeb0e8c624c11f1517`.
- NVIDIA A40, unquantized BF16 backbone, reference kernels with FLA disabled.
  Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.21.2.
- LoRA rank 16, alpha 32, dropout 0.05; **14,376,960 trainable parameters**.
  AdamW learning rate 3e-5, weight decay 0.01, gradient clipping 1.0, microbatch
  1 with accumulation 16, gradient checkpointing, seed 42.
- Prompt limits: query 256 tokens, each option 128, total 3,072. Token shortening
  audits are retained. Target projections and zero-LoRA parity are recorded in
  the admission receipts.
- The resumed training stage took **1,971.2 seconds** and peaked at
  **9,956,569,088 allocated GPU bytes**. This includes resumed training and its
  validation work; it excludes the initial attempt, setup, retrieval building,
  reload, and export. It does not establish a controlled speedup or minimum VRAM.

The supplied outcomes estimate original plus resumed GPU/storage cost at
**C$1.880**, using the approved FX factor 1.4243. This is a historical estimate,
not settled billing, and excludes tax. Saved teardown receipts record provider
readbacks confirming the two experiment pods were absent; no separate network
volume was attached. These are retained receipts, not a fresh account inventory.
The saved full-workload projection is a planning estimate, not an executed run
or approval for another run.

## What is included

- `scripts/`, `skillret_decider/`, `tests/`, `notebooks/`, `configs/`, and `docs/`:
  supplied experiment sources, including the resume patch's runtime-budget tests.
- `reference_run/`: original failure, resumed completion, metrics, predictions,
  ID-only split/candidate/selection evidence, tokenization audits, environment
  snapshots, remote test receipts, and teardown receipts.
- `selected_adapter/`: the selected step-100 PEFT adapter, configuration, and
  original model card. Its 57,549,856-byte tensor file uses **Git LFS**; SHA-256
  `2d35306c3f18f99fe337d729a95f5af5cf6eb60cce1a4df376a74319d1bb6338`.
- `provenance/SOURCE_SELECTION.json`: original archive hashes, verification of
  every export member against the local extraction, and source hashes/transforms
  for selected files. `SHA256SUMS` covers this current publication.

`ORIGINAL_START_HERE.md`, `ORIGINAL_STATUS.md`, and the original source docs retain
their **pre-training** statements. They are historical package documentation;
this README and the recorded run evidence describe the completed pilot.
`provenance/ORIGINAL_KIT_SHA256SUMS` applies to the initial kit, before the resume
patch, rather than this publication.

The complete original exports, intermediate adapters, optimizer/RNG states,
tokenizers, raw public dataset text, and embeddings remain local under
`local-experiments/artifacts/skillret-jevk5-runpod-2026-10-02/source-drive/` in the
MIA5150 workspace. They are excluded from Git. The drive remains unchanged.

The [local source audit](provenance/LOCAL_SOURCE_AUDIT.json) checked all 127,190
raw qrels and rederived the 256 validation shortlists from the saved vectors.
All primary top-15 orders matched exactly. Fourteen queries had differences
farther down the top-100 ranking, with score gaps at most 8.35e-7 in the local
FP32 calculation. The recorded top-100 lists remain unchanged; the audit does
not claim exact reproduction of those lower positions. No encoder was rerun.

## Verify the recorded snapshot

From this folder after cloning:

```sh
git lfs install --local
git lfs pull
python3 verify_recorded.py
shasum -a 256 -c SHA256SUMS
```

The standard-library verifier checks hashes/inventory, deterministic selection
and option order, split-family disjointness, natural candidate lists, all 20
ranking/decision metrics for all four conditions, paired bootstrap intervals,
selected adapter hashes, and the fresh-process reload logits/rankings. It does
not unpickle optimizer state or run the models.

The supplied CPU tests need pytest, NumPy, and PyTorch. In the local MIA5150
workspace, use the existing sandbox environment from this folder:

```sh
ALLOW_PAID_RUNS=0 ALLOW_PAID_NOTEBOOK_RUNS=0 HF_HUB_OFFLINE=1 \
  ../.venv/bin/python -m pytest -q tests
```

From `repo/skillret-jevk5-runpod/`, that Python path is
`../../local-experiments/.venv/bin/python`. These tests cover code contracts and
toy CPU training; the saved RunPod receipts provide the real-model evidence.

## Run a new experiment

Use a fresh run directory and the pinned sources, not the recorded evidence
directories. [ORIGINAL_START_HERE.md](ORIGINAL_START_HERE.md) describes the CLI
stages; [docs/RUNPOD_OPERATIONS.md](docs/RUNPOD_OPERATIONS.md) describes resource
supervision. Preserve the existing outputs and agree a separate GPU budget and
teardown plan before a paid run. Full training and official test evaluation need
their own decision after freezing the procedure.

`requirements.txt` contains the kit's proposed version ranges. The observed
RunPod versions are retained separately under `reference_run/runtime/`; the
original bootstrap installs a runtime tag, while the saved environment identifies
its resolved commit. A new installation must pin that commit and validate its
CUDA stack. Local offline checks do not establish a fresh GPU installation.

The text-free repo evidence supports scoring audits. It cannot be passed directly
to the original inference CLI, which needs the hashed query/skill-text candidate
file. The full local import retains that file and the original run layout.
Loading the selected adapter requires the pinned JevK5 backbone and the provided
letter-logit reader; the adapter alone is not a standalone model.
