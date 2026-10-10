# Verified PDF build

The automatic push build passed on 27 September 2026 UTC (26 September in Toronto).

- [Successful GitHub Actions run](https://github.com/speccy88/mia5150/actions/runs/36288360372)
- Built source commit: `e8cea05aea6197d37e8edec917ab21bbd7d16a98`
- Engine: pdfTeX 1.40.29 / TeX Live 2026, with `latexmk` and BibTeX.
- Both PDF uploads and the diagnostic-log upload passed.

| GitHub-produced PDF | Pages | Bytes | Review |
|---|---:|---:|---|
| `project_proposal.pdf` | 4 (3 body + 1 references) | 117,966 | All pages rendered and visually reviewed |
| `neurips_2026.pdf` | 12 | 171,289 | All pages rendered and visually reviewed, including checklist |

Downloaded file hashes matched the artifact API's digests:

```text
28b1da07cc5810ffca1645c80f34b8cd9f3005181502c41d0bba3883bf5083db  project_proposal.pdf
fa0266f16a5cc967e47d171d388a91930b8336ace1db486304646610ad9cf23a  neurips_2026.pdf
```

The final TeX logs contain no unresolved citations, unresolved references or
overfull boxes. Expected first-pass citation warnings settled automatically.
The proposal has a nonfatal spacing warning and the expected warning that shell
escape is disabled. The original template retains its sample placeholders and
a nonfatal spacing warning; it is a template example, not a completed submission.

Before the push, both documents also compiled locally with Tectonic 0.17.0 in
untrusted mode. Shell syntax, workflow structure and byte-for-byte hashes of the
three supplied files passed checks. No benchmark/model calls were made.

## Download compatibility

GitHub CLI 2.71.2 tried to unpack the direct PDF artifact as a ZIP and returned
`zip: not a valid zip file`. The workflow itself succeeded. Browser downloads and
the authenticated artifact API return the actual PDFs. For this historical run,
the following commands were verified (artifacts expire after 30 days):

```bash
mkdir -p build
gh api repos/Anonymous/mia5150/actions/artifacts/10921616830/zip > build/project_proposal.pdf
gh api repos/Anonymous/mia5150/actions/artifacts/10921576893/zip > build/neurips_2026.pdf
```

Despite the legacy endpoint name `/zip`, these two artifacts are direct PDF bytes.
Prefer the run page's **Artifacts** links for future runs, whose IDs will differ.

## Hosted-runner storage

These measurements came from `docker system df` on GitHub's disposable runner:

| Resource | Before | After |
|---|---|---|
| Images | 6 / 1.856 GB | 7 / 6.692 GB |
| Containers | 0 / 0 B | 0 / 0 B |
| Local volumes | 0 / 0 B | 0 / 0 B |
| Build cache | 0 / 0 B | 0 / 0 B |

The pinned TeX Live image added 4.836 GB of image storage. No pruning was performed.
No active containers or volumes existed before or after. Both required PDFs were
verified after the build. Local Docker and Docker Desktop storage were not used.
