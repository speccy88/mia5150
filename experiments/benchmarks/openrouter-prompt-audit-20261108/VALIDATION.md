# Publication validation

Validation covers the new prompt audit, conditional reconstructed-input retrieval
study, and their integrated figures. The historical input snapshot is
`3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`.

## Passed checks

- **30 prompt-audit and visualization tests** and **11 offline-retrieval tests**
  pass. Tests cover fixed-K scoring, duplicate/invalid rank slots, empty versus
  missing outcomes, source-hash guards, paired question clusters, name-equivalent
  single-target labels, reconstructed catalog order, and saved-reranker gates.
- An independent execution of `audit.py` into a separate output directory
  reproduces all eight report/data artifacts byte for byte, using the documented
  5,000 bootstrap draws and seed 5150.
- All five frozen input hashes match `source/input_manifest.json`. Eleven
  unchanged original benchmark files, including the raw results and figures,
  match the pinned commit. The historical report backup is byte-identical to its
  original; only the current report and navigation documents are replaced.
- All nine figure pairs match their recorded hashes. PNG dimensions are
  1920 by 1080; SVG viewboxes have the same 16:9 aspect ratio. Figure 3 uses the
  same query subset for every method within each model-size panel.
- Staged changes pass the repository's Gitleaks 8.30.1 command. Three scanner
  matches were an independently verified public tokenizer SHA-256, not secrets.
  The reviewed exception is restricted to that exact hash in its three evidence
  files. No complete-history scan or credential exception was used.
- Relative document links, publication checksums, study manifests, and staged
  whitespace are checked before commit. Commit identity is the repository's
  generic `Research Team` identity.

## Commands

Run from the repository root:

```sh
python -m unittest discover -s experiments/benchmarks/openrouter-prompt-audit-20261108/tests -v
python -m unittest discover -s experiments/benchmarks/openrouter-matched-retrieval-20261108 -p test_offline.py -v
python experiments/benchmarks/openrouter-prompt-audit-20261108/audit.py --output-dir /path/to/separate-output
git diff --cached --check
gitleaks git --pre-commit --staged --config .gitleaks.toml --redact=100 --max-decode-depth 1
```

## What these checks do not establish

Passing arithmetic and publication checks does not recover original response
positions, raw prompts, the missing historical parser/cleaner, or billed usage.
The name-equivalent target remains distinct from strict UUID identity. The new
centroid-domain router is not the historical router. Jev plans and candidate
oracles are not measured reranker results. Exposed single-target queries and
small paired samples do not establish blind generalization or a broad winner.
