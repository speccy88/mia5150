# Repository layout

[Repository home](../README.md) · [Experiments](../experiments/README.md)

The root contains navigation, repository instructions, configuration, and clear
entry folders. Studies are grouped by purpose; their internal source/results
layout stays together. `EDA/eda_report.html` remains the Pages homepage source.

## Old-to-new paths

| Previous root path | Current location |
| --- | --- |
| `benchmark_bm25` | [experiments/benchmarks/benchmark_bm25](../experiments/benchmarks/benchmark_bm25) |
| `skillret-cpu-benchmark` | [experiments/benchmarks/skillret-cpu-benchmark](../experiments/benchmarks/skillret-cpu-benchmark) |
| `skillret-full-catalog-runpod` | [experiments/benchmarks/skillret-full-catalog-runpod](../experiments/benchmarks/skillret-full-catalog-runpod) |
| `skillret-reranker-study` | [experiments/benchmarks/skillret-reranker-study](../experiments/benchmarks/skillret-reranker-study) |
| `skillret-embedding-jev-fusion` | [experiments/benchmarks/skillret-embedding-jev-fusion](../experiments/benchmarks/skillret-embedding-jev-fusion) |
| `skillret-local-metadata-20261005` | [experiments/benchmarks/skillret-local-metadata-20261005](../experiments/benchmarks/skillret-local-metadata-20261005) |
| `skillret-pilot` | [experiments/pilots/skillret-pilot](../experiments/pilots/skillret-pilot) |
| `skillret-jevk5-runpod` | [experiments/pilots/skillret-jevk5-runpod](../experiments/pilots/skillret-jevk5-runpod) |
| `skillret-embeddinggemma2-pilot` | [experiments/pilots/skillret-embeddinggemma2-pilot](../experiments/pilots/skillret-embeddinggemma2-pilot) |
| `skillret-jina-nano-pilot` | [experiments/pilots/skillret-jina-nano-pilot](../experiments/pilots/skillret-jina-nano-pilot) |
| `skillret-local-rubrics-20261006` | [experiments/pilots/skillret-local-rubrics-20261006](../experiments/pilots/skillret-local-rubrics-20261006) |
| `skillret-searchjev-update20` | [experiments/pilots/skillret-searchjev-update20](../experiments/pilots/skillret-searchjev-update20) |
| `skillret-seed5150-scaling` | [experiments/scaling/skillret-seed5150-scaling](../experiments/scaling/skillret-seed5150-scaling) |
| `MIA5150_Proposal_Demo` | [demos/MIA5150_Proposal_Demo](../demos/MIA5150_Proposal_Demo) |
| `skillret-user-pdfs` | [docs/inputs/skillret-user-pdfs](../docs/inputs/skillret-user-pdfs) |
| `artifact-publication-audit` | [docs/publication/artifact-publication-audit](../docs/publication/artifact-publication-audit) |
| `PUBLICATION_MANIFEST.sha256` | [docs/publication/PUBLICATION_MANIFEST.sha256](../docs/publication/PUBLICATION_MANIFEST.sha256) |
| `docs/PUBLICATION-PRIVACY.md` | [docs/publication/PUBLICATION-PRIVACY.md](../docs/publication/PUBLICATION-PRIVACY.md) |
| `docs/LATEX.md` | [docs/paper/LATEX.md](../docs/paper/LATEX.md) |
| `docs/LATEX_VALIDATION.md` | [docs/paper/LATEX_VALIDATION.md](../docs/paper/LATEX_VALIDATION.md) |
| `docs/latex-template-provenance.json` | [docs/paper/latex-template-provenance.json](../docs/paper/latex-template-provenance.json) |
| `project_proposal.tex` | [paper/project_proposal.tex](../paper/project_proposal.tex) |
| `references.bib` | [paper/references.bib](../paper/references.bib) |
| `neurips_2026.tex` | [paper/neurips_2026.tex](../paper/neurips_2026.tex) |
| `neurips_2026.sty` | [paper/neurips_2026.sty](../paper/neurips_2026.sty) |
| `checklist.tex` | [paper/checklist.tex](../paper/checklist.tex) |

The original course template is preserved under
[docs/course/PROJECT-TEMPLATE.md](course/PROJECT-TEMPLATE.md). Its example
commands and placeholder tree describe a template, not this repository.

## Reproduction and historical evidence

Open the study's new directory and follow its README. Archived JSON/CSV results,
model files, request records, and source snapshots were moved without rewriting
their contents. Old absolute runtime paths and source-commit references in those
records describe the original execution; the table above gives current locations.

Current file hashes are in
[publication/PUBLICATION_MANIFEST.sha256](publication/PUBLICATION_MANIFEST.sha256).
They use paths relative to the repository root. Run checksum verification from
that root after hydrating any Git LFS files needed for the verification.

Routine publication checks changed content only, as specified in
[the privacy guide](publication/PUBLICATION-PRIVACY.md). Pure file moves are
verified by comparing Git file identities; they do not require rerunning inference
or scanning the complete history.
