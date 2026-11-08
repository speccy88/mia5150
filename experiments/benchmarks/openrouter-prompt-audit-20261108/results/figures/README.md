# Presentation figures

All figures are **16:9**, with **1920 × 1080 PNG** exports and **editable-text SVG** exports. They are derived from the independent audit outputs, not from the historical report's aggregate values. Source snapshot: `3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45`.

## Figure index

1. **Top-1 accuracy across all nine models**: [PNG](01_accuracy_by_size.png) · [SVG](01_accuracy_by_size.svg). Small multiples preserve all nine models. Solid lines use the subset with no reported invalid skills; dashed lines show the explicitly labelled retained-ranking sensitivity. Shading is the audit's 95% query-bootstrap interval. The second line of each x-axis tick gives primary/completed denominators.
2. **Recall@5 and NDCG@5 versus library size**: [PNG](02_recall_ndcg_at5.png) · [SVG](02_recall_ndcg_at5.svg). Both metrics use the no-reported-invalid subset, with the same denominator annotations and audit intervals. Circles and squares distinguish the metrics without relying only on color.
3. **Query-matched prompt/retrieval comparison**: Recall@5 [PNG](03a_matched_recall5.png) · [SVG](03a_matched_recall5.svg); NDCG@5 [PNG](03b_matched_ndcg5.png) · [SVG](03b_matched_ndcg5.svg). Every method within a model–size panel is restricted to exactly the same archived query IDs as that prompt model's no-reported-invalid subset. Cohorts can differ between model panels. Uses the audited retrieval `name_equivalent_single_target` contract, not strict UUID scoring or multi-positive qrels. Historical candidate-catalog identity is conditional, because the original loader/cleaner and run-specific hashes are absent. The centroid-domain method is a clearly labelled new label-free supplement, not the historical domain router. These are descriptive point estimates, not a significance claim.
4. **Quality versus estimated cost**: [PNG](04_quality_estimated_cost.png) · [SVG](04_quality_estimated_cost.svg). Both axes use the same 68 common size–query cells for every model (19 distinct query IDs; represented sizes 50, 100, 250, 500 and 1000). Estimated costs are recomputed from frozen registry prices and recorded tokens. They are not invoices. Actual billed cost is unavailable, not zero. Vertical lines are clustered-query bootstrap intervals for accuracy.
5. **Median and p95 successful-response latency**: [PNG](05_latency_median_p95.png) · [SVG](05_latency_median_p95.svg). All panels share logarithmic seconds axes. Denominators include retained completed responses, excluding context placeholders and absent rows. Tail estimates from at most 20 responses per cell are descriptive.
6. **Coverage heatmap**: [PNG](06_coverage_grid.png) · [SVG](06_coverage_grid.svg). Every planned model–size cell is displayed. Counts show completed/20 planned. Partial, context-blocked and absent results are distinct states with text and pattern encodings; an absent record is not proof that no attempt occurred.
7. **Ranking quality at K=5/K=10**: [PNG](07_ranking_quality_k5_k10.png) · [SVG](07_ranking_quality_k5_k10.svg). Uses the same 68 common size–query cells across all nine models. It displays Recall, NDCG and MRR without ranking models by noisy point estimates. Under the single-gold contract, Recall equals Hit rate, MRR equals MAP, and Precision equals Recall/K. Full metrics and intervals remain in `../common_panel_summary.csv`.
8. **Jev evidence gap**: [PNG](08_jev_evidence_gap.png) · [SVG](08_jev_evidence_gap.svg). The bounded repository evidence search found no completed cells matching this query panel. Historical results are not reused as a gain, and no zero-gain bar is fabricated. The underlying evidence is the retrieval study's `results/jev-audit.json`.

## Interpretation limits

- `no_reported_invalid` is an explicitly restricted subset, not proof that the original response order was recovered. Historical raw responses and invalid-name positions are missing. Parser deduplication may also have removed information. `retained_ranking` is a sensitivity result, not a corrected original-rank score.
- The nine models' per-size primary subsets need not contain the same queries. Figures 1–2 describe those conditional subsets. Figures 4 and 7 instead use the auditor's common panel intersection.
- Repeated size evaluations of the same 20 queries are not independent samples. The audit resamples query IDs and retains their sizes together; the 95% percentile intervals are not multiplicity-adjusted and do not model parser or historical-candidate uncertainty.
- All absolute rate axes start at zero. Library size is plotted on a true numeric logarithmic axis. Timing and estimated cost have explicitly labelled logarithmic axes.
- Empty cells remain gaps. No plotted metric, confidence bound, charge or Jev gain is invented. Known recorded zero-quality answers remain zero.
- Estimated token cost is not actual provider billing. The historical report's manually stated “actual” totals are not independently attributable to these requests.

## Reproduction

From the repository root, first regenerate this study's audit and the sibling matched-retrieval study's saved results, then run:

```bash
python experiments/benchmarks/openrouter-prompt-audit-20261108/visualizations/plot_audit.py
python -m unittest discover \
  -s experiments/benchmarks/openrouter-prompt-audit-20261108/tests \
  -p test_visualizations.py -v
```

The renderer requires Python, NumPy and Matplotlib; tests also use Pillow. It performs no network or paid API calls and does not import the historical report generator. CLI flags `--summary`, `--common-summary`, `--prompt-rows`, `--retrieval-results`, `--config` and `--output` can point to independently generated inputs without copying them.

`figure_data.json` contains the exact audited source rows read for these figures. `matched_figure_data.json` additionally records each matched mean, denominator and query-index membership for Figure 3. The renderer verifies the retrieval query-file hash and snapshot, rejects duplicate/missing retrieval keys, and cross-checks every matched prompt mean and denominator against the independent audit summary. If the optional sibling comparison ledgers are unavailable, those comparison figures are not generated; they are never replaced with fabricated values. `figure_manifest.json` records source SHA-256 values and each PNG/SVG checksum. SVG text remains editable. The canvas is never cropped with `bbox_inches="tight"`, so the aspect ratio remains exactly 16:9.

## Validation

- Automated tests cover missing-versus-zero semantics, absent required cells, empty quality cohorts, coverage states, all-nine inclusion, exact PNG/SVG dimensions, same-query retrieval joins and refusal to invent a Jev gain.
- The exporter checks visible text bounds against the canvas; off-axis logarithmic locator ticks that Matplotlib does not draw are excluded from this check.
- The final PNGs were inspected for text clipping, crowded labels, denominator visibility, redundant non-color encodings and consistency of units. Marker shape, line style and hatch encodings preserve meaningful distinctions in grayscale.
- Color-vision-deficiency simulation was not performed. No web/mobile interactive layout or production serving latency was tested; these are static presentation exports.
