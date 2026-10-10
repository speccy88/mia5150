# 7 October 2026 artifact publication audit

The source ownership inventory selected the explicitly named MIA5150 local experiment, its retrieval/rubric follow-up, scaling version 2, SearchJev 20-update pilot, EG2/Jina retrieval pilots, and supplied PDF identities. Unrelated speech, vision, category and routing projects were not copied. The source branch was fast-forwarded from `041f2ceccd3f43e9f841289541b68083c795df73` to `34833ac8b5acf980a3c2eeb0d128684ea7586502`, preserving its four completed commits. Old reference studies retain their original files.

* source-provenance.json lists every selected source file, original SHA-256, exported SHA-256, and exclusions.
* source-inventory-summary.json provides per-folder source counts and sizes.
* export-file-manifest.json binds the exported artifact paths/bytes/hashes.
* export-redactions.json documents credential-shaped public input examples. Original source files remain local; IDs, labels, rankings and metrics were not changed. Historical request names/hashes still describe the original request bytes, not redacted exports.
* offline-validation.json records syntax/JSON/JSONL validation and structured credential scans. Trufflehog was not installed; this is a bounded pattern scan, not proof that every possible secret format is detected.
* offline-tests.json records the existing offline suites and the intended Linux/runtime requirements.

The canonical local scaling ZIP matches supplied Library version-2 SHA-256 `35fe8071ed1ecdbd7327e09c130424f9c23d657a27575a9a59f8367239007b6c`; the canonical SearchJev ZIP matches `4c3418dad6f7f34c9ac0a6daf98b402a4d73a5b0663245771cba614f28f70ece`. All their original entries were extracted and CRC checked. Original package manifests are historical evidence; current export hashes account for documented redactions and repository README additions.

## Large and duplicate omissions

No multi-gigabyte base weights, virtual environments, dependency overlays/wheels, or vector caches were published. Small trained adapters and local checkpoint evidence were included; SearchJev's final adapter is 43,340,856 bytes. Original asset pins appear in each experiment's model/dependency manifests. Download upstream model files at those immutable revisions and compare their expected SHA-256 before rebuilding vectors with the recorded encoding scripts/configurations.

The large metadata upstream corpus omissions are `data/skills/train.jsonl` (187,920,814 bytes), `data/skills/test.jsonl` (117,606,037 bytes), and SRA `corpus/corpus.json` (232,255,987 bytes). Use the pinned dataset revisions, evidence/data-manifest.json, manifests/sra-files.json, fetch_data.py and fetch_sra.py to reconstruct original inputs; adapt their original destination paths first. Scaling's exact public query/qrel inputs and corpus metadata projection are included, with documented export redaction. A projection cannot reconstruct full skill bodies or their original file hash. Scaling vector arrays were already excluded from its canonical ZIP and require existing arrays or re-encoding; historical EG2/Jina producer provenance remains incomplete.

Duplicate delivery ZIPs were not committed alongside expanded files. Cached demos, code and raw experiment evidence remain available separately. The three user PDFs remain blocked by Library transfer errors; skillret-user-pdfs/README.md records exact names, identities and byte sizes. This is a documented incomplete PDF upload, not an assertion that all source bytes were published.

No GPU, training, paid API or live benchmark run was performed for consolidation. Offline tests use synthetic fixtures or saved artifacts. No sharing policy, host settings, or source files were changed.
