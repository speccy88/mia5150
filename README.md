# MIA5150 — SkillRet research

Code, recorded results, and documentation for studying how natural-language
requests retrieve and select agent skills. Start with the
[22-study benchmark catalog](docs/SkillRet-Benchmark-Catalog.md) or the
[published EDA report](https://speccy88.github.io/mia5150/).

Authored documentation and presentation text are English. See the [language audit and immutable-data exceptions](docs/publication/ENGLISH_LANGUAGE_AUDIT.md).

Both BGE-M3 studies are published on `main`. Use the
[current Git LFS download and CPU replay commands](docs/publication/BGE-REPRODUCTION.md)
to reproduce their rankings and metrics without embedding again. The archived
study instructions retain the original proposed experiment branch names.

## MetaEncoder v1 on RunPod

The [MetaEncoder v1 study](experiments/benchmarks/skillret-metaencoder-v1-runpod/REPORT.md)
completed full-catalog retrieval for all 6,006 skills and 4,392 test queries:
**0.720593 NDCG@10**, **0.791249 Recall@10**, **0.792056 MRR@10**.
Content-preserving chunking and exploratory dimension reductions are documented.
Secondary decision experiments and final delivery reconciliation are in progress;
this is not yet a completed study across every requested arm.

## SkillRet 0.6B query LoRA checkpoint

The [published checkpoint](experiments/benchmarks/skillret-local-challenger-checkpoint-20261109/README.md) contains two query-only adapters,
all indexed document and query vectors, and CPU replay for five official
conditions. The two pure query adapters reached **0.817349** and **0.817469**
NDCG@10 over all 4,392 official queries, versus the archived **0.810990**.
Official reranker, scaling, and latency measurements are in progress.

## BGE-M3 LoRA: paired before/after retrieval

The [completed LoRA study](experiments/benchmarks/skillret-bge-m3-lora-20261108/REPORT.md)
selects its adapter using training-derived validation, then evaluates the exact
same A/B/C inputs as the frozen BGE-M3 baseline. All 480 scaling and 13,176
official query/configuration outcomes are independently rescored and replayed
from lossless vectors. Official LoRA NDCG@10 A/B/C is **0.382641 / 0.433377 / 0.428759**.
The primary B-minus-matching-base difference is **-0.128022**,
95% paired interval **[-0.137472, -0.118557]**. Training completed 1,100
optimizer steps with validation early stopping; the selected adapter is step
950. The failed first pilot is preserved separately.
[Portable adapters](experiments/benchmarks/skillret-bge-m3-lora-20261108/adapters/README.md) and
[document/query vectors with model-free replay](experiments/benchmarks/skillret-bge-m3-lora-20261108/vectors/README.md)
are included through Git LFS. Exposed public data and the short validation
proxy limit generalization.

## BGE-M3 versus Qwen3 full-text benchmark

The [completed BGE-M3 study](experiments/benchmarks/skillret-bge-m3-fulltext-20261108/REPORT.md)
evaluates metadata, exact Qwen chunk spans and native 8K chunks on the same
6,006 skills, eight scaling libraries and all 4,392 official queries.
Official NDCG@10 is **0.557723 / 0.561399 / 0.565406** for A/B/C, versus
archived Qwen **0.642091**. The primary paired B-minus-Qwen difference is
**−0.080692**, 95% interval **[−0.089143, −0.072258]**.
BGE uses less VRAM and has lower measured query latency; runtime, precision
and query conventions differ. All document and query vectors are included
through [Git LFS with verified offline replay](experiments/benchmarks/skillret-bge-m3-fulltext-20261108/vectors/README.md).

## Qwen3 Embedding 8B Q4 full-text benchmark

The [completed local study](experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext/REPORT.md)
embeds all 6,006 complete skill documents with 100% source coverage and evaluates
all eight 20-query scaling libraries plus all 4,392 official test queries.
Full-test NDCG@10 is **0.642091**, Recall@10 **0.701389**, and MRR@10 **0.732295**.
The historical 20-query panel is exposed and gold-conditioned; protocols differ
from the prior full-text SkillRet 0.6B result of 0.810990 NDCG@10.
[Lossless document and query vectors](experiments/benchmarks/skillret-qwen3-embedding-8b-q4km-fulltext/vectors/README.md)
are included through Git LFS, with an offline command that reproduces every
scaling and official ranking without embedding again.

## Nine-model OpenRouter audit (8 November 2026)

The [audited prompt-results report](experiments/benchmarks/openrouter-prompt-audit-20261108/results/REPORT.md)
recomputes all saved rankings from commit `3fb7e66`: 1,184 completed responses,
40 preflight context exclusions and 36 absent cases across nine models.
Raw data are unchanged. Seventy-two responses lack recoverable invalid-entry
positions, so identifiable and parser-filtered views are separated, with
paired question-cluster intervals. Recorded list-price estimates total
US$35.140929; actual billed spend and cache savings are unverified.
See [reproduction and figures](experiments/benchmarks/openrouter-prompt-audit-20261108/README.md)
and the [conditional offline retrieval comparison](experiments/benchmarks/openrouter-matched-retrieval-20261108/README.md).
No new paid calls were made. Results are exposed, single-gold and not a
fair unqualified leaderboard across unequal coverage.

## Skill discovery listing studies (3 November 2026)

[Compare measured accuracy and coverage](experiments/benchmarks/skill-discovery-local-native-20261103/COMPARISON.md):
three OAuth models produced 360 completed responses. The initial local 32K series
produced 1,120 measured outputs from 14 configurations, with four blocked. A
[separate Qwen8/Mistral context/KV series](experiments/benchmarks/skill-discovery-local-native-20261103/profiles-v3/PROFILE_REPORT.md)
adds 200 responses: 16 distinct local models now have measurements across the
series. Llama 3.2 and Gemma 3 were not run before the study was closed, as
requested. The local final-answer adapter and OAuth settings differ. The same 20 questions recur at each
size, with every expected correct skill (gold label) at positions 1-20; these are
exposed, position-biased results.
Read the [OAuth report](experiments/benchmarks/skill-discovery-oauth-v1-20261103/REPORT.md)
and [local report](experiments/benchmarks/skill-discovery-local-native-20261103/LOCAL_REPORT.md).
Unexecuted cases and context or hardware exclusions have no measured accuracy.

## Latest matched comparison (24 October 2026)

[Read the visual PDF summary](experiments/benchmarks/skillret-gap-closure-20261024/report/MIA5150_Gap_Analysis_and_Results_2026-10-24.pdf). The validated ten-page report includes a dated hosted-results addendum and cites immutable local (`0f4e9818`) and hosted terminal (`d769811`) evidence. The underlying Qwen audit separately targets `497ca150`.

The [proposal gap study](experiments/benchmarks/skillret-gap-closure-20261024/README.md) completes a 200-query exposed primary panel across five methods and the original 10/100/1,000/6,006 catalog sizes. At 6,006 skills, the small local selector exactly recovered 9/200 sets with BM25, 7/200 with vector and 11/200 with hybrid. The paired interval does not establish a hybrid advantage. Full listing exceeds the configured local context at 1,000 and 6,006 skills.

Read the [measured report and charts](experiments/benchmarks/skillret-gap-closure-20261024/report/RESEARCH_DRAFT.md), [raw primary outcomes](experiments/benchmarks/skillret-gap-closure-20261024/results/primary200-canonical.jsonl), and [gap closure matrix](experiments/benchmarks/skillret-gap-closure-20261024/GAP_CLOSURE.md). The separate local seed/model sensitivity and full 4,392-query retrieval pass are complete. The [hosted follow-up](experiments/benchmarks/skillret-gap-closure-20261024/hosted/reports/HOSTED_REPORT.md) adds 120 completed Qwen/SiliconFlow cases on a shared balanced 24-query panel at N=100: exact sets are listing 5/24, BM25 13/24, vector 7/24, domain vector 4/24 and hybrid 10/24. Known reported charges are US$0.02411946; three failed provider canaries retain US$0.19 UNKNOWN holds, so actual total spend remains unknown. Human adjudication and the original hosted matrix remain incomplete.

## Find your way

| Directory | What you will find |
| --- | --- |
| [experiments/](experiments/README.md) | Self-contained studies grouped into [benchmarks](experiments/benchmarks/README.md), [pilots](experiments/pilots/README.md), and [scaling](experiments/scaling/README.md) |
| [EDA/](EDA/README.md) | Dataset analysis, golden-skill distributions, and the GitHub Pages homepage |
| [paper/](paper/README.md) | Editable proposal, bibliography, NeurIPS style, and template example |
| [demos/](demos/README.md) | The separate proposal demonstration notebook and its validation limits |
| [docs/](docs/README.md) | Study catalog, course template, paper guides, publication audit, and supplied-input inventory |
| [scripts/](scripts/README.md) | Repository maintenance and local paper build commands |

```text
.
├── README.md
├── AGENTS.md
├── EXPERIMENTS.md
├── experiments/
│   ├── benchmarks/
│   ├── pilots/
│   └── scaling/
├── EDA/
├── paper/
├── demos/
├── docs/
│   ├── course/
│   ├── inputs/
│   ├── paper/
│   └── publication/
└── scripts/
```

Each study keeps its source, results, manifests, and reproduction instructions
together. The [experiment index](EXPERIMENTS.md) records completion limits;
partial, failed, and unrun arms must not be treated as completed benchmarks.
Scores from different query panels, document views, and candidate pools are not
directly comparable.

## Read or reproduce

- **Dataset:** open the [live EDA report](https://speccy88.github.io/mia5150/).
- **Recorded benchmarks:** use the [catalog](docs/SkillRet-Benchmark-Catalog.md)
  or its [Wiki page](https://github.com/speccy88/mia5150/wiki/SkillRet-Benchmark-Catalog).
- **A particular study:** open its README under [experiments/](experiments/README.md).
  Dependencies and execution requirements belong to that study.
- **Proposal PDF:** edit [paper/project_proposal.tex](paper/project_proposal.tex)
  and follow the [PDF build guide](docs/paper/LATEX.md).
- **Course requirements:** consult the preserved [course template](docs/course/PROJECT-TEMPLATE.md).
- **An old folder link or command:** consult the [old-to-new path map](docs/REPOSITORY-LAYOUT.md).

The golden-label analysis can be checked without downloads or inference:

```sh
python3 EDA/golden_skills.py --check
python3 -m unittest discover -s EDA -p 'test_*.py'
```

GitHub Pages publishes `EDA/eda_report.html` automatically from `main`.
Paper sources also build automatically through
[Build LaTeX PDFs](https://github.com/speccy88/mia5150/actions/workflows/latex-pdf.yml).

## Contributing

Keep all code and documentation on `main`. Follow [AGENTS.md](AGENTS.md) and the
[publication/privacy guide](docs/publication/PUBLICATION-PRIVACY.md).
Routine updates scan changed files only. Large model downloads, environments,
private credentials, and original private execution files stay outside this
public repository.

Team aliases: `deepujoy88`, `speccy88`, `vsimard`.

The one later Mistral/Parasail BF16 arm returned an upstream HTTP429 at its first canary and produced no benchmark cases. The campaign is terminal: 150 generation attempts, US$0.02411946 known charges, US$0.19 UNKNOWN holds, US$0.21411946 guarded. No further model or retry. Only Qwen has successful hosted quality evidence. [Terminal scope and costs](experiments/benchmarks/skillret-gap-closure-20261024/hosted/mistral-reports/terminal-execution.json).
