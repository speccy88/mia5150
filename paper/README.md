# Paper sources

[Repository home](../README.md) · [PDF build guide](../docs/paper/LATEX.md)

| File | Purpose |
| --- | --- |
| [project_proposal.tex](project_proposal.tex) | Editable SkillRet course proposal |
| [references.bib](references.bib) | Proposal bibliography |
| [neurips_2026.sty](neurips_2026.sty) | Shared NeurIPS formatting |
| [neurips_2026.tex](neurips_2026.tex) | Original template example |
| [checklist.tex](checklist.tex) | Checklist included by the template example |

All five source files were moved together so their local includes and citations
keep working. Source contents were preserved during the reorganization.
Generated PDFs are build artifacts, separate from these editable files.

Edit the proposal and push to `main`, then download the PDFs from
[Build LaTeX PDFs](https://github.com/speccy88/mia5150/actions/workflows/latex-pdf.yml).
With a local TeX installation, run `bash scripts/build_latex.sh` from the
repository root. Output stays under `build/`.
