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
