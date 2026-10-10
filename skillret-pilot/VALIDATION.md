# GitHub handoff validation

Validated on 26 September 2026, from this folder in a separate clone of
`Anonymous/mia5150`, using a new project-local Python 3.12.6 environment on macOS.

| Check | Measured outcome |
|---|---|
| Install `requirements.lock.txt` | 153 packages installed; `uv pip check` passed |
| `python -m pytest -q` via `scripts/verify.py` | **54 passed** |
| Pinned dataset download and checksum checks | Passed; 6,006 skills / 4,392 queries / 7,187 qrels |
| New full-document BM25 computation | Recall@5 **0.5166666666666666**; 200 queries / 6,006 skills |
| New matched-metadata BM25 computation | Recall@5 **0.6058333333333333**; same 200 / 6,006 |
| New CPU BGE retrieval computation | Recall@5 **0.4925**; same 200 / 6,006 |
| Native metadata/tool isolation | Passed with scripted model control |
| Separate native read check | Passed offline; no skill execution |
| Notebook 01 | 5 code cells executed successfully |
| Notebook 02 | 14 code cells executed successfully; live/local inference skipped |
| Ruff undefined-name checks and formatting | Passed |
| Core executable-code comparison | Seven core files retain identical Python syntax trees after removing docstrings |
| Historical reference ZIP | CRC and exact SHA-256 verified; all 447 members scanned |

The installation and offline steps were actually run; these are not predicted
results. See [validation/](validation/) for the small saved command/test/notebook
records. Runtime downloads and caches are ignored by Git. New retrieval rankings
were computed without installing historical runtime results or budget ledgers.

**No new OpenRouter or Ollama generation was performed for publication.** Paid-run
and notebook opt-ins were disabled, and no paid ledger was created. Historical
live integration evidence remains in `reference_run/`, clearly separate from
this publication check. Local scripted/native tests do not establish current
provider availability.

The publication changes add comments, readable formatting, fresh-checkout notebook
guards, and reference-fixture loading for tests. Automated notebook execution also
disables model-call opt-ins in child kernels. Experimental prompts, retrieval
algorithms and budget ceilings are retained. Source hashes therefore differ from
the archived run; new live cases will not reuse archived outcomes.

Before pushing, the staged files and the archive contents were checked for key
patterns, private-key blocks, secret/environment paths, caches and symlinks.
The original local `.env` was neither copied nor read during publication. The
existing project and its historical results remain unchanged.
