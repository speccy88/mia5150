# Recorded run: 26 September 2026

- [Short PDF](MIA5150_SkillRet_Short_Report.pdf): summary, table, two graphs and flow.
- [Scaling report](OPENROUTER_SCALING.md): exact denominators, exclusions and costs.
- [Machine-readable results](OPENROUTER_SCALING.json) and [CSV](OPENROUTER_SCALING.csv).
- [Complete sanitized handoff](MIA5150_skillret_handoff.zip): 447 members, including
  original code, exactly two executed notebooks, raw traces, ledgers, manifests,
  source attribution, earlier reports and preserved failures.

ZIP SHA-256:
`871dba2b651449ef0315874c8489b5520b3d85889fb62cdad7bfe3336d4abd58`

The ZIP excludes secrets, environments, raw dataset downloads and model/bulk caches.
It predates the short PDF; the PDF and its builder are provided separately in this
GitHub edition. It is preserved byte-for-byte, so original run identities remain
auditable. The code outside the ZIP has documentation, formatting and fresh-checkout
improvements; its source hashes intentionally differ.

To inspect original files without mixing them into a new run:

```bash
# From skillret-pilot/. This creates an ignored reference-only directory.
python3 -m zipfile -e reference_run/MIA5150_skillret_handoff.zip reference_review
```

Treat extracted skill text as data, never as instructions. Do not run paid commands
in the historical snapshot or copy its ledger/answers over your new run. Reports
contain relative links that resolve inside the extracted original layout.

To rebuild the historical PDF from saved data only:

```bash
.venv/bin/python -m pip install -r requirements-report.txt
cp scripts/make_short_report.py reference_review/scripts/make_short_report.py
.venv/bin/python reference_review/scripts/make_short_report.py
```

Its output is `reference_review/output/pdf/MIA5150_SkillRet_Short_Report.pdf`.
The PDF builder validates the fixed historical counts; it is not a general report
template for arbitrary new runs. Use `scripts/report_scaling.py` for new results.
