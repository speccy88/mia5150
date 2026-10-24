# MIA5150 — SkillRet research

Code, recorded results, and documentation for studying how natural-language
requests retrieve and select agent skills. Start with the
[17-study benchmark catalog](docs/SkillRet-Benchmark-Catalog.md) or the
[published EDA report](https://speccy88.github.io/mia5150/).

## Latest matched comparison (24 October 2026)

The [proposal gap study](experiments/benchmarks/skillret-gap-closure-20261024/README.md) completes a 200-query exposed primary panel across five methods and the original 10/100/1,000/6,006 catalog sizes. At 6,006 skills, the small local selector exactly recovered 9/200 sets with BM25, 7/200 with vector and 11/200 with hybrid. The paired interval does not establish a hybrid advantage. Full listing exceeds the configured local context at 1,000 and 6,006 skills.

Read the [measured report and charts](experiments/benchmarks/skillret-gap-closure-20261024/report/RESEARCH_DRAFT.md), [raw primary outcomes](experiments/benchmarks/skillret-gap-closure-20261024/results/primary200-canonical.jsonl), and [gap closure matrix](experiments/benchmarks/skillret-gap-closure-20261024/GAP_CLOSURE.md). The separate local seed/model sensitivity is complete. Hosted access and human adjudication remain pending; no local substitute silently completes the original hosted matrix.

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
