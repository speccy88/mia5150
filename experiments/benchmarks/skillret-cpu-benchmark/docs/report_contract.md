# Recovery report contract

## Evidence boundary

The original benchmark tree became unavailable. `docs/retained_observations.json` contains genuinely observed, previously audited aggregate values that survived in recorded readouts. It is historical evidence, not invented measurements; original raw rankings, timing observations, vectors, and pair scores are unavailable. Never fabricate or reconstruct those rows from means. Never mix these historical values into fresh paired comparisons, bootstrap intervals, or common-cohort tables.

The new report is generated from canonical `artifact.json` through the installed Data Analytics `report:deliver` helper. No parallel HTML renderer, browser installation, inference or package changes are performed by report tooling.

## Inputs

- `results/retrieval64/<slug>_{per_query.csv,latency.csv,summary.json,run.json}` for BM25, BGE, E5, Gemma
- `results/quality32/<native-key>_{per_query.csv,summary.json,run.json}` then analogous `quality64`
- `results/timing32/<native-key>_{latency.csv,run.json}`
- `data/processed/queries.jsonl`: the exact hash-ordered common64 queries; the full4392 source may be saved separately
- `data/processed/selected_query_ids.json`, `full_catalog.json`, and the pinned dataset manifest

Required per-query fields are model, method, catalog_size, query_id, gold_count, ranked_skill_ids, and the nine metric fields defined in `src/build_report.py`. query_group is checked or derived. Native results must include shortlist_mrr5. Optional pair-truncation counters are preserved.

The loader independently recalculates top-five MRR, Recall, Hit, nDCG and completeness from rankings and gold labels. Ranking IDs must be known and unique. Native rankings must preserve the exact BGE hybrid candidate membership and agree on first-stage coverage and initial MRR. No incomplete condition is converted to a zero result.

## Common-cohort selection

All retrieval models must complete the exact64-query grid: BM25 × four catalog sizes; each BGE/E5/Gemma × vector/hybrid/domain_vector × four catalog sizes. Select common64 only when every native model also completes all four catalog sizes on those exact IDs. Otherwise use the exact shared common32 subset only after every native model completes its own four-size common32 grid. Retrieval rows are filtered to those same32 IDs for comparison. There is no unequal-cohort model ranking.

The full-catalog cross-pipeline table contains exactly eight pipelines: BM25, BGE/E5/Gemma hybrid, and the four BGE-shortlist native pipelines. Candidate scope and MRR ceiling remain explicit.

Latency must contain the same first32 IDs × repeat0/1 at catalog6006. No overlapping observations are pooled. Native timing configurations and fingerprints are checked against selected quality metadata where provided. Smaller-catalog native timing is outside scope. Readiness requires all eight cross-pipeline timing conditions, plus common quality. A running controller remains a checkpoint even if common32 is already available.

## Completion receipts and archive restoration

Read completed CSVs only when their run receipt exists, is non-pilot/non-partial, and is at least as new as the CSV. For compact archive restoration, a newer CSV may be accepted only if one exact relative-path entry in root `rank_restore_manifest.json` matches its SHA256 and declared byte count. Missing/mismatched restoration entries never weaken the live-run guard.

## Build and checkpoints

```sh
python3 src/build_report.py --checkpoint --output-dir artifacts/checkpoint
python3 src/build_report.py --artifact-only --output-dir artifacts/preparation_check
python3 src/build_report.py
python3 -m unittest discover -s tests -p 'test_report*.py'
```

Every build emits canonical artifact.json and a compact report_build_receipt.json. A successful HTML build emits report.html. `report_analysis.json` contains fresh aggregate analysis and exact input hashes; raw rows remain in their original source CSVs.

The current installed helper returns `structural_only`: canonical validation, exact payload equality, and structural/semantic fallback checks passed, but no compatible installed headless-shell was available. Do not claim browser visual QA, responsive rendering, source-dialog interaction, or chart SVG extraction was verified. Retain semantic chart tables and disclose the limitation on delivery. No Sites publication is requested.

## Narrative and QA

Answer first; define metrics/cohorts before their use. Primary fresh quality, paired shortlist changes, multi-positive coverage, matched latency, hardware/setup, limitations and next steps stay visible. Historical4392 aggregates and original fast8 observations stay separate, with unavailable-raw warnings. The old fast8 baseline already equaled its candidate-any-hit ceiling (.75), leaving zero MRR-improvement headroom. Its one-observation timings, initial ONNX defaults, and Laya rounded-log timing are not pooled into recovery results.

Scores are descriptive. Add independently verified uncertainty outputs only after their cohort/source hashes match fresh evidence. Model-specific prefixes, pooling, calibration, finite-choice limits, query-only truncation, candidate preservation and Julia cache settings must be documented from actual fresh code/receipts. Do not infer causal runtime improvements from uncontrolled setting changes. Process RSS is a high-water mark, not resident model memory.

Inference is bounded by11:52UTC and delivery by12:12UTC on2026-10-02. Report code never starts or resumes inference.

## Preregistered common48 fallback

Before any extension quality was observed, the user approved the first48 prefix as an additional runtime-budget fallback. Selection priority is exact all-model64 → exact all-model48 → exact all-model32. All use prefixes of the same frozen64 order; no cohort is chosen based on accuracy. Timing remains the same first32 IDs ×2. The report reads `results/quality48` with the same completion, configuration, rank and gold-set checks. The runtime forecast must preserve at least15 minutes before the inference cutoff when choosing64; report code never makes or executes the inference scheduling decision.

## Approved compact PDF companion

The primary user-facing PDF is an explicitly abridged, separately authored four-page benchmark brief from the same final validated metrics, not a faithful export of every HTML section. Full HTML, notebook, raw fresh data and provenance stay in the ZIP. The selected delivery format is a compact PDF plus the complete supporting archive.

`src/build_compact_pdf.py` is prepared but must not run during measured timing. After inference/timing stops, first build the final canonical artifact, then generate the PDF from that exact artifact. The script verifies the eight-pipeline cohort/grid, uses the same32×2 latency data, and only accepts companion intervals matching the selected cohort hash and independently checked paired delta. Render all PDF pages to PNG and visually inspect every page; repair layout before handoff. Confirm four pages, selectable text and no clipped tables or missing glyphs. Do not equate PDF visual QA with the HTML companion's structural-only browser checks.
