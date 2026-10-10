# SkillRet exploratory data analysis

The [published report](https://speccy88.github.io/mia5150/) uses
[eda_report.html](eda_report.html) as its homepage. The
[Pages workflow](../.github/workflows/eda-pages.yml) copies that file to the
deployment's `index.html` and publishes from `main` whenever the report, label
inputs, or workflow changes. No additional code branch is needed.

## Golden skills per query

A golden skill is a **distinct skill ID with positive relevance** for a query.
The analysis includes every query, including zero-label queries, deduplicates
positive qrel pairs, and checks that each query's `skill_ids` and `k` agree with
its positive qrels. Percentages use all queries within the indicated split;
the combined column pools both splits. The charts share a 0–100% scale.

| Golden skills | Train queries | Test queries | All queries |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 20,776 | 2,143 | 22,919 |
| 2 | 21,035 | 1,703 | 22,738 |
| 3 | 21,448 | 546 | 21,994 |
| Total | 63,259 | 4,392 | 67,651 |

These are labeled benchmark targets, not every possible relevant skill for a
real request. The inputs are the preserved
[query](../skillret-local-metadata-20261005/data/queries) and
[qrel](../skillret-local-metadata-20261005/data/qrels) splits. Their
[manifest](../skillret-local-metadata-20261005/evidence/data-manifest.json) pins
ThakiCloud/SKILLRET revision `6583d7d2ed07644d0fb8938ed8178f3a7dc42a12`.
Public query text was privacy-redacted; label IDs and relevance values were
preserved. [golden_skills_summary.json](golden_skills_summary.json) records the
counts, validation checks, and hashes of the actual public input files.

Refresh or verify the golden-skill section from the repository root with Python
3. No packages, dataset downloads, inference, or paid APIs are required:

```sh
python3 EDA/golden_skills.py
python3 EDA/golden_skills.py --check
python3 -m unittest discover -s EDA -p 'test_*.py'
```

This refresh preserves the existing report's other sections. Those sections
are historical EDA output; they were not recomputed for this label analysis.
The old `plotly-latest` URL was frozen at version 1.58.5, which does not support
the report's typed arrays. Interactive charts now use pinned Plotly.js 4.1.1
from the official CDN. The golden-skill table and SVG charts work without it.

## Rebuilding the entire historical report

`eda_analysis.py` requires pandas, Plotly, and the full pinned dataset export
(large skill-body files remain outside the repository). Supply a directory
containing `skills.jsonl`, `taxonomy.json`, and `skills/`, `queries/`, `qrels/`
train/test JSONL files:

```sh
python3 EDA/eda_analysis.py --data-dir /path/to/pinned/data
```

The full generator checks its golden-label counts against the public label
inputs and includes the same golden-skill section. Review examples, generated
content, and privacy before committing regenerated reports; keep private
corpora, local paths, and credentials outside the repository.
