#!/usr/bin/env bash
# Local equivalent of the GitHub build; no Python or API credentials are needed.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is missing. Use the GitHub 'Build LaTeX PDFs' workflow, or install a TeX distribution locally." >&2
  exit 127
fi

mkdir -p build
# latexmk repeats TeX/BibTeX as needed until citations and cross-references settle.
for source in project_proposal.tex neurips_2026.tex; do
  latexmk -pdf -no-shell-escape -file-line-error -halt-on-error \
    -interaction=nonstopmode -outdir=build "$source"
done
