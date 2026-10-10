# Edit the proposal and get its PDF

The repository builds **two independent PDFs** with the supplied NeurIPS 2026 style:

| Editable source | Generated artifact | Purpose |
|---|---|---|
| [`project_proposal.tex`](../project_proposal.tex) + [`references.bib`](../references.bib) | `project_proposal.pdf` | Custom MIA5150 SkillRet proposal |
| [`neurips_2026.tex`](../neurips_2026.tex), [`neurips_2026.sty`](../neurips_2026.sty), [`checklist.tex`](../checklist.tex) | `neurips_2026.pdf` | Original template example, including the checklist |

The three supplied files are preserved byte-for-byte. Their hashes are in
[`latex-template-provenance.json`](latex-template-provenance.json). `checklist.tex`
is an included fragment, not a standalone document. Its sample instructions and
unanswered items belong to the template example; they are not claims about our study.

## Automatic build

1. Open `project_proposal.tex` in GitHub or your editor and change the proposal.
2. Commit and push to `main`, or open/update a pull request.
3. Open [Actions > Build LaTeX PDFs](https://github.com/speccy88/mia5150/actions/workflows/latex-pdf.yml).
4. Open the newest successful run and download **project_proposal.pdf** from
   **Artifacts**, or use the link in the run summary. The template PDF is separate.

Changes to `.tex`, `.sty`, `.cls`, `.bib`, files under `figures/`, or the build
configuration trigger compilation. Changes only to the Python pilot do not.
You can also select **Run workflow** on the workflow page without editing anything.
Artifacts are retained for **30 days**; rerun to regenerate expired downloads.
Sign in with access to this private repository to download its artifacts.

GitHub compiles both roots with TeX Live 2026 and `latexmk`. Bibliography and
cross-reference passes are automatic. A compile error fails the workflow and
retains diagnostic logs for seven days; the job never substitutes an older PDF.
Generated PDFs are workflow artifacts, not automatic commits, so there is no
commit/build loop. The workflow has read-only repository permission and needs no
API key or additional repository secret.

See [the verified build record](LATEX_VALIDATION.md) for the first successful run,
PDF checksums, visual review and an older GitHub CLI download workaround.

## What to edit

- **Proposal text and tables:** `project_proposal.tex`. Comments mark the sections.
- **Citations:** `references.bib`; add `\citep{key}` or `\citet{key}` in the proposal.
- **Figures:** put them under `figures/`, then use `\includegraphics` if needed.
- **Shared formatting:** the supplied `neurips_2026.sty`; keep it unchanged unless
  an updated official style is deliberately adopted.

The proposal uses `preprint` mode to show the team's names without claiming
conference acceptance. It is a course proposal, so the unfilled conference
checklist is compiled only in the original template PDF. The outline previously
at `project_proposal.tex` remains available in Git history (commit `b9c9df2`).

The draft separates measured pilot results from planned experiments. Student
numbers were not supplied and are not invented. Before course submission, add
them if required, confirm the proposed team roles and next-stage budget, and
review the final rendered PDF. The current paper does not authorize more API calls.

## Optional local build

With a TeX distribution that provides `latexmk`, pdfLaTeX and the standard NeurIPS
packages, run from the repository root:

```bash
bash scripts/build_latex.sh
```

Output: `build/project_proposal.pdf` and `build/neurips_2026.pdf`. The script disables
shell escape and stops on errors. It never installs software or accesses a model.
No local TeX installation is needed when using GitHub Actions.

Implementation references: [latex-action](https://github.com/xu-cheng/latex-action)
and [upload-artifact](https://github.com/actions/upload-artifact). Action revisions
are pinned by commit SHA and the TeX Live image by its immutable digest.
