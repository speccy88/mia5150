# English language audit — 3 November 2026

Repository documentation, reports, navigation, chart labels, working login UI and
authored explanatory code text are English. The translation preserves scientific
meaning, model settings, scores, raw input/output data and Git history.

The complete current tracked tree was inventoried: **16,323 files**, approximately
1.33 GB in the local checkout. Content heuristics scanned 16,223 UTF-8 files,
including raw HTTP bodies and SVG text. Three gzip evidence files were expanded
for inspection. All text members in the historical 447-entry handoff ZIP were
checked; no strict French signal was found. All five PDFs contain extractable
English text and no strict French signal. The two translated plots were regenerated
from unchanged measurements and visually reviewed.

The first broad word scan produced many false positives from English words such as
precision and execution. A refined authored-file scan yielded 34 candidates. Manual
review separated new French presentation text, frozen evidence, the English word
proposition in external code/model-card text, and embedded raw queries. The final
strict whole-tree scan found 71 files before rebuilding the consolidated report:
one authored report subsequently replaced in English and 70 immutable evidence
files. An additional HTML demo with escaped raw multilingual queries is listed.
The exact classified paths and hashes are in [the audit manifest](ENGLISH_LANGUAGE_AUDIT.json).

## Necessary evidence exceptions

- Original skill descriptions, training/evaluation queries and recorded requests
  retain their source language. Translation would change prompts and invalidate
  comparisons. This includes multilingual text embedded in a historical demo.
- Frozen query fixtures and their recorded outputs remain unchanged. English
  documentation surrounding them explains their purpose.
- External tokenizer vocabularies retain all language tokens. Translating entries
  would corrupt model behavior.
- Exact executed-source snapshots retain their measured bytes and hashes,
  including two historical OAuth helpers with French UI strings. Current working
  helpers are English; snapshots are reproducibility evidence, not current UI.

No authored French narrative is retained as an alternative current report. Earlier
French publication versions remain accessible through ordinary Git history; history
was not rewritten. New authored text should use English, while quoted or measured
source evidence must preserve its actual language.

## Preservation and practical limits

The translation checks unchanged bytes for 96 selected new-study evidence files,
including all 38 CSVs, 33 SQLite exports, datasets and executed-source snapshots.
Offline metric and prompt validation and 64 synthetic tests pass without model
inference. The language evidence manifest separately verifies each retained
whole-tree exception against the pre-translation commit.

Language heuristics and source/visual review are not a formal proof that every
possible French substring is absent. Binary model/array/SQLite artifacts remain
evidence; 13 Git LFS pointers were not hydrated. PDF text extraction does not perform
OCR. Those limits do not justify modifying measured data or source snapshots.

## Follow-up correction review — 3 November 2026

Review of translation commit `7b90ba68728fff42c58be6b577ffc3926b9ab9d5`
found omissions in the earlier language pass: French thousands separators in
several reports, diagnostic report labels, generator fallback text and two error
messages in each working OAuth helper. These presentation strings are now English.
The earlier inventory and JSON record describe the original audit, not an
exhaustive guarantee; they remain available as historical evidence.

The follow-up also clarifies 60 context exclusions **per model** (1,080 across
18 models), the initial series' 32K scope, the separate v3 context/KV conditions,
metric recalculation versus stored-result checks, and the uncertain outcome of
pending inference transaction `100:1`, which was not retried. Gold labels and the
P@10 ceiling are explained; study closure wording is made natural.

All 317 tracked scientific input/result/audit files in the two studies were checked
against that commit and remain byte-identical, including all 38 CSVs and 33 SQLite
exports. Measured table values, prompts, configurations, historical validations,
source snapshots and the original 96-file preservation proof are unchanged.
Current file and publication manifests are refreshed; historical archive mappings
and language-audit hashes retain their original meaning.

Local, v3 and public OAuth offline validation passed. Disposable report regeneration
preserved table data; the comparison and supplement reproduce exactly, and all four
consolidated report sections match their standalone reports. Across the synthetic
suites, 62 tests passed and two Windows DPAPI tests were skipped. On Linux, the
serialized listener tests require modeling the Windows login guard; without that
adapter, the same two pre-existing failures occur with both baseline and corrected
helpers. No live OAuth request or model inference was made. Public OAuth database
rechecks and full private raw-response normalization remain outside public replay.
Details and counts are in [the follow-up review record](ENGLISH_CORRECTION_REVIEW.json).
