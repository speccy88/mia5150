# SkillRet full-catalog RunPod benchmark

This folder records the five-method benchmark completed on 28 September 2026.
It is separate from `skillret-pilot/` and `benchmark_bm25/`; their results and
configurations were not changed. The seven-page [report](REPORT.pdf) contains
the method, failure analysis, timings, and limitations.

## Measured official test result

All methods ranked the same 6,006-skill test catalog for all 4,392 test queries.
The denominator is every official test query. Values below are macro means;
`results/aggregate.json` and the five `results/*_per_query.jsonl` files retain
the unrounded values and individual outcomes.

| Full-catalog method | Recall@5 | MRR | Hit@1 |
| --- | ---: | ---: | ---: |
| BM25 | 0.5028 | 0.5701 | 0.4943 |
| BGE-M3 dense | **0.5545** | **0.6374** | **0.5651** |
| Frozen Qwen3-8B raw pooled | 0.0132 | 0.0146 | 0.0066 |
| Original pretrained CLM heads | 0.0582 | 0.0543 | 0.0253 |
| SkillRet-adapted CLM heads | 0.3174 | 0.3151 | 0.2222 |

The adapted heads improved Recall@5 over the original heads by 0.2592 on
this test set. The paired group bootstrap interval for the difference is
[0.2472, 0.2712] (`results/bootstrap.json`). This is one training seed on
synthetic SkillRet queries with incomplete relevance judgments, not a claim
about universal model quality. BGE-M3 and BM25 also differ from the Qwen/CLM
chain in architecture and pretraining; the controlled head comparison is the
last three rows. No shortlist, reranker, Jev method, or generative selector
followed retrieval.

## What is here

- `REPORT.pdf`, `figures/`, `results/aggregate.json`, bootstrap intervals,
  validation history, error examples, and per-query top-10 outcomes.
- Exactly two run notebooks, `src/`, `starter/`, `tests/`, setup scripts,
  pinned CLM source, and the frozen protocol and source notes in `evidence/`.
- Test qrels and ordered query/skill IDs, the saved training-group selection,
  source hashes, stage receipts, checkpoint reload and parity checks, environment
  details, and the completed remote pytest log (`29 passed`).
- The best trained head at `checkpoints/clm_skillret/best_head.pt`, stored with
  Git LFS. Run `git lfs install --local` and `git lfs pull` after cloning.
  Its SHA-256 is
  `45770bd332ece2830ed624bac30c33f3ac3267eed684edf151abd0cc36fa4dd4`.
  The training config and head architecture are next to it; the loader is
  `src/load_clm_head.py` and needs the included `src/` and `clm_source/src/`
  on `PYTHONPATH`, plus PyTorch and NumPy.
- `PUBLISHED_MANIFEST.sha256` records the logical bytes of every published
  file, including the LFS checkpoint after `git lfs pull`.

The full local backup also holds all train/test embeddings, full score and
ranking matrices, the original head and resume checkpoints, logs, and the
sanitized comprehensive ZIP. It is 4.39 GB and is **not** in Git. Its SHA-256
is `c253942ce5db442ce7ff683f8d60ab7f71d849b7d343e415d0cda8886704c411`.
The per-query files here are enough to audit the published aggregate metrics,
but not to reconstruct every full-catalog score without regenerating the
vectors. Raw query and skill text are omitted from Git; `src/data_stage.py`
downloads the pinned public dataset for reproduction. See `MODEL_CARD.md` for
the exact encoder recipe.

## Check the committed snapshot

From this directory:

```sh
git lfs install --local
git lfs pull
shasum -a 256 -c PUBLISHED_MANIFEST.sha256
python verify_published_results.py
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q tests
```

The verification script checks query coverage, test qrels, top-10 rankings,
aggregate metrics, the PDF hash, and the LFS checkpoint bytes. The pytest
suite checks metric definitions, multi-positive loss, ranking ties, and budget
rules. The original RunPod environment used an NVIDIA A40, Python 3.12.3,
PyTorch 2.8.0+cu128, and vLLM 0.10.2; see `evidence/remote_environment.json`
and `evidence/requirements.freeze.txt` for the observed environment.

For a new paid run, start with a **fresh** `/workspace/<run>` directory and
copy the code, `clm_source/`, setup script, and tests there. The stage receipts
in this folder intentionally say `complete`; running `src/remote_run.py`
against this recorded directory would skip those stages. `setup_cu128.sh`
captures the environment used on the RunPod PyTorch CUDA 12.8 image. The data
script downloads SkillRet revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`;
the model revisions are pinned in `src/embedding_stage.py`. Configure a
separate budget guard and deadline before launching another paid run.

The experiment pod and its volume disk were terminated after the complete ZIP
and extracted head passed local checksum checks. No network volume was made.
The observed RunPod credit difference at teardown was $0.7912; provider
billing may post later. The account API exposed a `$` display, but not an ISO
currency code.
