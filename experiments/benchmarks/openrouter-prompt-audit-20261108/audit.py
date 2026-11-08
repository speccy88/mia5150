#!/usr/bin/env python3
"""Offline, fail-closed audit of the frozen OpenRouter CSV. No provider imports/calls."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

SOURCE_COMMIT = '3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45'
STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
RAW = ROOT / 'benchmark_openrouter/tests_openrouter'
K_VALUES = (1, 3, 5, 10)
SEED = 5150
BOOTSTRAP_REPLICATES = 5000
COHORTS = ('retained_ranking', 'no_reported_invalid', 'invalid_insertion_lower')


def score_ranking(ranking, gold, ks=K_VALUES, valid_ids=None):
    """Score ordered output slots; invalid and duplicate slots NEVER get removed.

    Precision uses fixed K. AP denominator is min(K, number of golds).
    Exact-set uses the entire output, not top-1 or a truncated cutoff.
    Empty gold is an invalid evaluation case; missing outputs are not passed here.
    """
    gold = set(gold)
    if not gold:
        raise ValueError('An evaluation query must have at least one gold label')
    seen = set()
    relevant = []
    valid = []
    duplicate = []
    for item in ranking:
        is_valid = isinstance(item, str) and bool(item) and (valid_ids is None or item in valid_ids)
        is_duplicate = is_valid and item in seen
        valid.append(is_valid)
        duplicate.append(is_duplicate)
        relevant.append(int(is_valid and not is_duplicate and item in gold))
        if is_valid:
            seen.add(item)
    out = {'top1_accuracy': float(bool(relevant and relevant[0])),
           'exact_set_match': float(bool(ranking) and all(valid) and set(ranking) == gold),
           'output_slot_validity': sum(valid) / len(ranking) if ranking else None,
           'duplicate_slot_rate': sum(duplicate) / len(ranking) if ranking else None,
           'returned_count': len(ranking)}
    for k in ks:
        rel = relevant[:k]
        hits = sum(rel)
        precision = hits / k
        recall = hits / len(gold)
        first = next((i for i, value in enumerate(rel, 1) if value), None)
        dcg = sum(value / math.log2(i + 1) for i, value in enumerate(rel, 1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
        ap = sum(sum(rel[:i]) / i for i, value in enumerate(rel, 1) if value) / min(k, len(gold))
        for name, value in {
            'recall': recall, 'precision': precision, 'hit': float(hits > 0),
            'mrr': 1 / first if first else 0., 'ndcg': dcg / idcg,
            'map': ap, 'completeness': float(hits == len(gold)),
            'f1': 2 * precision * recall / (precision + recall) if hits else 0.,
        }.items():
            out[f'{name}@{k}'] = value
    return out


METRICS = ['top1_accuracy', 'exact_set_match'] + [f'{name}@{k}' for k in K_VALUES
    for name in ('recall', 'precision', 'hit', 'mrr', 'ndcg', 'map', 'completeness', 'f1')]


def cluster_bootstrap(values, query_ids, seed=SEED, replicates=BOOTSTRAP_REPLICATES):
    """Percentile interval: resample query IDs, retaining every size of each query.

    values is rows x metrics. A query repeated across sizes stays a single cluster.
    Each draw estimates the row-weighted mean of the resampled full clusters.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(query_ids):
        raise ValueError('Values and query IDs must have the same length')
    if len(values) == 0:
        return None, None
    keys = sorted(set(query_ids))
    sums = np.array([values[np.array([q == key for q in query_ids])].sum(axis=0) for key in keys])
    counts = np.array([query_ids.count(key) for key in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(replicates, len(keys)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    return np.quantile(means, .025, axis=0), np.quantile(means, .975, axis=0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_registry():
    tree = ast.parse((ROOT / 'benchmark_openrouter/benchmark_models_testplan_openrouter.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', None) == 'OPENROUTER_MODELS':
            return ast.literal_eval(node.value)
    raise ValueError('Frozen model registry not found')


def number(value):
    return None if value in ('', None) else float(value)


def load_rows(raw_dir=RAW):
    cfg = json.loads((raw_dir / 'run_config.json').read_text())
    queries = json.loads((raw_dir / 'test_queries.json').read_text())
    query_index = {}
    for i, q in enumerate(queries, 1):
        key = (q['query'][:120], q['gold_skill'])
        if key in query_index:
            raise ValueError('Ambiguous truncated-query/gold join; full request IDs required')
        query_index[key] = f'q{i:02d}'
    with (raw_dir / 'details.csv').open(newline='') as fh:
        original = list(csv.DictReader(fh))
    by_key = {}
    registry = model_registry()
    issues = Counter()
    for index, source in enumerate(original, 2):
        query_key = (source['query'], source['gold_skill'])
        if query_key not in query_index:
            raise ValueError(f'Unmatched query on CSV line {index}')
        model, size, qid = source['model'], int(source['library_size']), query_index[query_key]
        key = (model, size, qid)
        if key in by_key:
            raise ValueError(f'Duplicate evaluation key at CSV line {index}: {key}')
        if model not in cfg['models'] or size not in cfg['library_sizes']:
            raise ValueError(f'Out-of-design row at line {index}')
        ranking = json.loads(source['ranked_skills'])
        invalid = json.loads(source['invalid_skills'])
        if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
            raise ValueError(f'Unsupported ranked_skills at line {index}')
        error = source['error']
        if error == 'CONTEXT_LIMIT_ESTIMATED':
            status = 'context_excluded_not_attempted'
        elif error:
            status = 'attempted_failure'
        else:
            status = 'completed'
        n_invalid = int(source['num_invalid_skills'])
        if n_invalid != len(invalid):
            issues['invalid_list_count_mismatch'] += 1
        if int(source['num_valid_skills']) != len(ranking):
            issues['valid_list_count_mismatch'] += 1
        if len(ranking) != len(set(ranking)):
            issues['duplicate_in_saved_ranking'] += 1
        row = dict(source_line=index, model=model, library_size=size, query_id=qid,
                   gold_skill=source['gold_skill'], status=status, error=error or None,
                   ranking=ranking, invalid_skills=invalid, n_invalid=n_invalid,
                   recorded_selected_skill=source['selected_skill'],
                   original_positions_recoverable=False,
                   no_reported_invalid=(n_invalid == 0),
                   n_valid_before_saved_truncation=int(source['num_valid_skills']),
                   recorded_estimated_cost_usd=number(source['est_cost_usd']), actual_cost_usd=None,
                   latency_s=number(source['latency_s']) if status == 'completed' else None,
                   e2e_latency_s=number(source['e2e_latency_s']) if status == 'completed' else None,
                   input_tokens=number(source['input_tokens']) if status == 'completed' else None,
                   output_tokens=number(source['output_tokens']) if status == 'completed' else None)
        if status == 'completed':
            info = registry[model]
            row['estimated_cost_usd'] = round(info['cost_input_per_1m'] * row['input_tokens'] / 1e6 +
                                             info['cost_output_per_1m'] * row['output_tokens'] / 1e6, 6)
            if abs(row['estimated_cost_usd'] - row['recorded_estimated_cost_usd']) > 1e-8:
                issues['recorded_cost_recompute_mismatch'] += 1
            retained = score_ranking(ranking, [row['gold_skill']])
            row['scores'] = {'retained_ranking': retained,
                             'no_reported_invalid': retained if n_invalid == 0 else None,
                             'invalid_insertion_lower': score_ranking([None] * n_invalid + ranking, [row['gold_skill']])}
            for metric in METRICS:
                if metric in source and source[metric] != '' and not math.isclose(float(source[metric]), retained[metric], abs_tol=1e-9):
                    issues[f'legacy_metric_difference:{metric}'] += 1
            if float(source['exact_match']) != retained['top1_accuracy']:
                issues['legacy_top1_difference'] += 1
        else:
            row['estimated_cost_usd'] = None
            row['scores'] = {c: None for c in COHORTS}
        by_key[key] = row
    rows = []
    for model, size, qid in itertools.product(cfg['models'], cfg['library_sizes'], query_index.values()):
        row = by_key.get((model, size, qid))
        if row is None:
            row = dict(model=model, library_size=size, query_id=qid, status='unattempted_missing',
                       source_line=None, n_invalid=0, estimated_cost_usd=None, actual_cost_usd=None,
                       latency_s=None, e2e_latency_s=None, input_tokens=None, output_tokens=None,
                       scores={c: None for c in COHORTS})
        rows.append(row)
    return rows, cfg, queries, registry, issues


def summarize(rows, cohort, replicates=BOOTSTRAP_REPLICATES):
    completed = [r for r in rows if r['status'] == 'completed']
    scored = [r for r in rows if r['scores'][cohort] is not None]
    latencies = [r['latency_s'] for r in completed if r['latency_s'] is not None]
    e2e = [r['e2e_latency_s'] for r in completed if r['e2e_latency_s'] is not None]
    cost = sum(r['estimated_cost_usd'] for r in completed) if completed else None
    out = dict(cohort=cohort, n_expected=len(rows), n_observed=sum(r['source_line'] is not None for r in rows),
               n_completed=len(completed), n_context_excluded=sum(r['status'] == 'context_excluded_not_attempted' for r in rows),
               n_attempted_failure=sum(r['status'] == 'attempted_failure' for r in rows),
               n_unattempted=sum(r['status'] == 'unattempted_missing' for r in rows),
               n_invalid_order=sum(r['n_invalid'] > 0 for r in completed),
               n_empty_output=sum(not r['ranking'] for r in completed),
               n_scored=len(scored), n_query_clusters=len({r['query_id'] for r in scored}),
               estimated_cost_usd=cost, estimated_cost_per_completed_response_usd=cost/len(completed) if completed else None, actual_cost_usd=None,
               cost_source='historical_registry_x_recorded_tokens; not invoice',
               latency_mean_s=float(np.mean(latencies)) if latencies else None,
               latency_median_s=float(np.median(latencies)) if latencies else None,
               latency_p95_s=float(np.quantile(latencies, .95)) if latencies else None,
               e2e_latency_mean_s=float(np.mean(e2e)) if e2e else None,
               e2e_latency_median_s=float(np.median(e2e)) if e2e else None,
               e2e_latency_p95_s=float(np.quantile(e2e, .95)) if e2e else None,
               input_tokens_total=sum(r['input_tokens'] for r in completed) if completed else None,
               output_tokens_total=sum(r['output_tokens'] for r in completed) if completed else None,
               slot_validity_recorded=(sum(len(r['ranking']) for r in completed) / sum(len(r['ranking']) + r['n_invalid'] for r in completed)) if completed and sum(len(r['ranking']) + r['n_invalid'] for r in completed) else None,
               mean_returned_count=float(np.mean([len(r['ranking']) for r in completed])) if completed else None)
    if scored:
        values = np.array([[r['scores'][cohort][m] for m in METRICS] for r in scored])
        low, high = cluster_bootstrap(values, [r['query_id'] for r in scored], replicates=replicates)
        means = values.mean(axis=0)
        for i, metric in enumerate(METRICS):
            out[metric] = float(means[i])
            out[f'{metric}_ci_low'], out[f'{metric}_ci_high'] = float(low[i]), float(high[i])
    else:
        for metric in METRICS:
            out[metric] = out[f'{metric}_ci_low'] = out[f'{metric}_ci_high'] = None
    return out


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty CSV output: {path}')
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(rows, cfg, replicates):
    records = []
    select = ['top1_accuracy', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10', 'exact_set_match']
    for cohort in ('retained_ranking', 'no_reported_invalid'):
        for size in [*cfg['library_sizes'], 'ALL']:
            lookup = defaultdict(dict)
            for row in rows:
                if (size == 'ALL' or row['library_size'] == size) and row['scores'][cohort] is not None:
                    lookup[row['model']][(row['library_size'], row['query_id'])] = row
            for a, b in itertools.combinations(cfg['models'], 2):
                keys = sorted(lookup[a].keys() & lookup[b].keys())
                delta = [[lookup[a][key]['scores'][cohort][metric] - lookup[b][key]['scores'][cohort][metric]
                          for metric in select] for key in keys]
                if not keys:
                    low = high = means = [None] * len(select)
                else:
                    low, high = cluster_bootstrap(delta, [k[1] for k in keys], replicates=replicates)
                    means = np.mean(delta, axis=0)
                for j, metric in enumerate(select):
                    records.append(dict(cohort=cohort, library_size=size, model_a=a, model_b=b, metric=metric,
                        n_paired_cells=len(keys), n_query_clusters=len({k[1] for k in keys}),
                        difference_a_minus_b=number(means[j]), ci_low=number(low[j]), ci_high=number(high[j]),
                        multiplicity='exploratory; no multiplicity adjustment'))
    return records


def paired_size_comparisons(rows, cfg, replicates):
    """Within-model size effects on identical query IDs (larger minus smaller)."""
    records = []
    metrics = ['top1_accuracy', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10', 'exact_set_match']
    for cohort, model in itertools.product(('retained_ranking', 'no_reported_invalid'), cfg['models']):
        lookup = defaultdict(dict)
        for row in rows:
            if row['model'] == model and row['scores'][cohort] is not None:
                lookup[row['library_size']][row['query_id']] = row
        for smaller, larger in itertools.combinations(sorted(cfg['library_sizes']), 2):
            keys = sorted(lookup[smaller].keys() & lookup[larger].keys())
            delta = [[lookup[larger][q]['scores'][cohort][m] - lookup[smaller][q]['scores'][cohort][m] for m in metrics] for q in keys]
            if keys:
                low, high = cluster_bootstrap(delta, keys, replicates=replicates)
                means = np.mean(delta, axis=0)
            else:
                low = high = means = [None] * len(metrics)
            for j, metric in enumerate(metrics):
                records.append(dict(cohort=cohort, model=model, smaller_library_size=smaller,
                    larger_library_size=larger, metric=metric, n_paired_cells=len(keys),
                    n_query_clusters=len(keys), difference_larger_minus_smaller=number(means[j]),
                    ci_low=number(low[j]), ci_high=number(high[j]),
                    multiplicity='exploratory; no multiplicity adjustment'))
    return records


def fmt(x, digits=3):
    return 'NA' if x is None else f'{x:.{digits}f}'


def report(audit, summaries, path):
    cfg = audit['design']
    overall = {r['model']: r for r in summaries if r['library_size'] == 'ALL' and r['cohort'] == 'retained_ranking'}
    primary = {r['model']: r for r in summaries if r['library_size'] == 'ALL' and r['cohort'] == 'no_reported_invalid'}
    lines = [
        '# Audited OpenRouter prompt-based benchmark', '',
        f"Source: `{SOURCE_COMMIT}`. Offline audit only; no new inference, API requests, or spending.", '',
        '## Evidence and denominators', '',
        f"The planned grid has {audit['counts']['planned_cells']:,} cells: nine models × seven catalog sizes × 20 repeated questions. The CSV has {audit['counts']['observed_rows']:,} rows: {audit['counts']['completed_responses']:,} error-free responses and 40 preflight context exclusions (not API failures). Claude has 36 absent cells. There are no duplicated evaluation keys and no recorded attempted API failures in this retained CSV. Retry/deleted-attempt history is not recoverable from it.", '',
        'The preserved metrics.json is also defective: all 63 model-size entries report zero valid responses, empty retrieval metrics and zero token/latency/cost efficiency, despite positive CSV measurements. Its exact_set_accuracy reproduces top-1 rather than true returned-set equality. The missing aggregate helper prevents a definitive root-cause replay; this audit replaces those aggregates in separate files, using explicit error-string handling and scored denominators.', '',
        'The historical REPORT described seven models/944 rows and US$28.65; it omitted the two newly added models. Its original bytes are preserved in source/historical_REPORT.md. The source details.csv, metrics.json, configuration, query list and historical figures are unchanged.', '',
        '**Original response-order metrics cannot be completely recovered.** The CSV stores parsed valid names separately from invalid names, not raw responses or their positions. Seventy-two completed rows contain reported invalid names; six completed rows have an empty parsed ranking. One stored ranking was truncated from 11 valid names to 10. The imported benchmark_models module, exact runtime version and exact prompt/library manifests are absent from the source commit. Even the zero-reported-invalid subset cannot rule out unrecorded parser normalization/deduplication.', '',
        'Three labeled views are therefore provided, without inventing ranks:',
        '- `no_reported_invalid`: identifiable stored-order subset (1,112 completed rows). This excludes all 72 rows with unrecoverable invalid positions, but includes completed empty outputs as retrieval failures. It is selection-biased and not a full-grid accuracy estimate.',
        '- `retained_ranking`: all 1,184 completed rows scored exactly as stored; filtered-rank sensitivity, not raw-response accuracy.',
        '- `invalid_insertion_lower`: all recorded invalid entries inserted before the stored list. A conditional lower bound for insertion of recorded invalids only, not a bound on unknown normalization, omitted duplicates, malformed prose or unsaved attempts. The retained view is the corresponding upper bound under that assumption.', '',
        'Context exclusions and absent cases have NA quality, latency and actual cost. Availability remains visible, never a false measured zero. Empty completed outputs receive zero retrieval scores. Unverified actual cost is NA even when an estimated cost exists.', '',
        '## Nine-model summary', '',
        'Metrics below use the retained-ranking sensitivity for comparability with saved output. The identifiable subset size and its top-1 value are shown separately. Aggregate sizes differ, so this is **not a leaderboard**.', '',
        '| Model | Completed/planned | No-invalid n | Top-1, retained | Top-1, no-invalid | Recall@5 | NDCG@5 | Median / p95 s | Est. US$ | Actual US$ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for model in cfg['models']:
        r, p = overall[model], primary[model]
        lines.append(f"| {audit['model_registry'][model]['display']} | {r['n_completed']}/{r['n_expected']} | {p['n_scored']} | {fmt(r['top1_accuracy'])} | {fmt(p['top1_accuracy'])} | {fmt(r['recall@5'])} | {fmt(r['ndcg@5'])} | {fmt(r['latency_median_s'])} / {fmt(r['latency_p95_s'])} | {fmt(r['estimated_cost_usd'],6)} | NA |")
    lines += ['', '## Metric definitions and interpretation', '',
        'For a nonempty binary gold set G and the first K output slots: Recall@K is unique gold hits / |G|; Precision@K is unique gold hits / K (short lists are not rewarded); Hit@K is any hit; MRR@K is reciprocal first relevant rank; NDCG@K uses log2(rank+1) discount and ideal DCG up to min(K, |G|); AP@K sums precision at relevant ranks / min(K, |G|), and MAP is its macro mean. Completeness@K means all golds found. F1@K is the harmonic mean of per-query precision and recall, then macro-averaged. All appear at K=1,3,5,10.', '',
        'The generic scorer leaves invalid and duplicate slots in place; a repeated relevant ID gets credit once. Top-1 is membership of the first slot in G. Exact-set match requires the whole saved returned set to equal G with no invalid slot; repeated valid IDs do not change set identity, and duplicate rate is reported separately. Exact-set here describes the saved list, not unavailable pre-truncation output.', '',
        '**Single-gold limitation:** every query has one labeled gold, so Recall = Hit = Completeness and MAP = MRR at each K. Precision = Recall/K, and F1 = 2×Recall/(K+1). These are mathematical identities, not eight independent findings. Plausible alternative skills are unjudged, not proven irrelevant. A valid ten-item list containing the gold can have top-1=1 but exact-set=0. Candidate/list completeness measures gold presence, not whether ten answers were returned.', '',
        'Slot validity in summary.csv is only the recorded parsed-valid share of saved names plus reported invalid names. The original slot denominator, JSON/syntax validity and ID-level catalog membership are unavailable. Name-level outputs cannot disambiguate duplicate-name catalog records. Do not call this end-to-end raw-output validity.', '',
        '## Paired uncertainty and fair comparison', '',
        f"Intervals are exploratory 95% percentile query-cluster bootstrap intervals, {audit['bootstrap']['replicates']:,} draws, seed {audit['bootstrap']['seed']}. Sampling one question resamples all its observed library sizes together. Paired contrasts use intersecting (query ID, size) cells and resample question clusters, never the 1,184 rows as independent observations. CSV output contains the actual paired-cell and independent-query denominators. Only 20 distinct questions exist. Partial cases may reduce this further.", '',
        'results/size_differences.csv contains within-model larger-minus-smaller library-size contrasts on the same queries. results/paired_differences.csv contains descriptive model differences for the identical successful cells of each pair; results/common_panel_summary.csv uses the common cells across all nine models within each view. No multiplicity correction is applied. Missing outcomes need not be missing at random. Neither a raw average nor overlapping point estimates establishes an advantage. All questions are exposed, golds are forcibly included in each sampled library, and source/library equivalence to other experiments is not established.', '',
        '## Costs, latency and token provenance', '',
        f"Recorded/recomputed list-price estimate: **US${audit['cost']['estimated_total_usd']:.6f}** for retained completed rows. Actual billed spend: **NA**. Prices are the frozen historical runner registry, not verified current rates. Tokens are recorded runner fields, potentially containing an estimated-input fallback; no raw usage payload distinguishes those cases. The recomputation reproduces every stored per-row estimate. These estimates exclude unknown deleted retries and are not an invoice or a total-spend assertion.", '',
        'The old US$17 actual-cost total and 83.9% caching claim have no invoice, usage export, cache-token evidence or per-request billed-cost record in this package. They remain historical unsupported assertions and are not used for savings or value claims. Reasoning-token and cache savings cannot be derived from response latency.', '',
        'Latency is the saved llm_latency/latency_s sum of request-attempt durations for completed rows, excluding explicit retry backoff sleeps. It is not TTFT. Separate e2e_latency summaries include the full per-query wall elapsed time, including backoff. The report uses mean, median and interpolated p95. Tokens and latency exclude preflight placeholders, whose input_tokens are estimates, not consumed tokens. Cross-model runs occurred at different times/providers; latency comparisons are descriptive, with no control for load, caching or reasoning settings.', '',
        '## Reproduction and limits', '',
        'Run `python experiments/benchmarks/openrouter-prompt-audit-20261108/audit.py` then `python -m unittest discover -s experiments/benchmarks/openrouter-prompt-audit-20261108/tests -v`. Only NumPy is required for the audit. Use the separate visualization reproduction instructions for charts. All input hashes are recorded in results/audit.json and rechecked before writing outputs. Original paid runners are not invoked.', '',
        'Outputs: metrics.json and summary.csv (corrected aggregates), per_query.jsonl (full planned grid with status/NA), summary.csv (all models/sizes/cohorts/metrics), common_panel_summary.csv, paired_differences.csv, audit.json (counts/checks/hashes) and this report. Fixed seed and sorted rows make numeric artifacts reproducible. Existing raw results are never overwritten.', '',
        'To obtain true original-position, strict-ID or independently billed metrics would require the missing original responses, exact parser and prompts/library manifests, request/attempt IDs, provider usage exports and invoices. This audit does not synthesize them or rerun paid calls. A new controlled experiment would also need held-out queries, multi-gold adjudication, explicit failure accounting, matched settings and a budget.', '',
    ]
    path.write_text('\n'.join(lines))


def run(out_dir=STUDY / 'results', replicates=BOOTSTRAP_REPLICATES):
    inputs = [RAW / name for name in ('details.csv','run_config.json','test_queries.json','metrics.json')]
    inputs += [ROOT / 'benchmark_openrouter/benchmark_models_testplan_openrouter.py']
    before = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    expected = json.loads((STUDY / 'source/input_manifest.json').read_text())
    if before != expected:
        raise RuntimeError('Frozen source hash mismatch; restore the pinned evidence before auditing')
    rows, cfg, queries, registry, issues = load_rows()
    legacy = json.loads((RAW / 'metrics.json').read_text())['per_model_size']
    issues['legacy_aggregate_cells_reporting_zero_valid'] = sum(r['num_valid'] == 0 for r in legacy)
    issues['legacy_aggregate_cells_missing_retrieval_metrics'] = sum(not r['retrieval_quality'] for r in legacy)
    issues['legacy_aggregate_cells_reporting_zero_efficiency'] = sum(all(v == 0 for v in r['efficiency'].values()) for r in legacy)
    summaries = []
    for model, size, cohort in itertools.product(cfg['models'], [*cfg['library_sizes'], 'ALL'], COHORTS):
        selected = [r for r in rows if r['model'] == model and (size == 'ALL' or r['library_size'] == size)]
        summaries.append(dict(model=model, library_size=size, **summarize(selected, cohort, replicates)))
    common_summaries = []
    for cohort in ('retained_ranking', 'no_reported_invalid'):
        keysets = [{(r['library_size'], r['query_id']) for r in rows if r['model']==m and r['scores'][cohort] is not None} for m in cfg['models']]
        common = set.intersection(*keysets)
        for model, size in itertools.product(cfg['models'], [*cfg['library_sizes'], 'ALL']):
            selected = [r for r in rows if r['model']==model and (r['library_size'],r['query_id']) in common and (size=='ALL' or r['library_size']==size)]
            common_summaries.append(dict(model=model, library_size=size, panel='all_nine_shared_cells', **summarize(selected,cohort,replicates)))
    pairs = paired_comparisons(rows,cfg,replicates)
    audit = dict(source_commit=SOURCE_COMMIT, input_sha256=before, design=cfg,
        model_registry={m:registry[m] for m in cfg['models']},
        counts=dict(planned_cells=len(rows), observed_rows=sum(r['source_line'] is not None for r in rows),
                    completed_responses=sum(r['status']=='completed' for r in rows),
                    context_excluded_not_attempted=sum(r['status']=='context_excluded_not_attempted' for r in rows),
                    unattempted_missing=sum(r['status']=='unattempted_missing' for r in rows),
                    attempted_failures_retained=sum(r['status']=='attempted_failure' for r in rows),
                    duplicate_evaluation_keys=0, distinct_queries=len(queries),
                    invalid_position_rows=sum(r['status']=='completed' and r['n_invalid']>0 for r in rows),
                    completed_empty_rankings=sum(r['status']=='completed' and not r['ranking'] for r in rows)),
        checks=dict(issues),bootstrap=dict(unit='query_id; all sizes retained together',seed=SEED,replicates=replicates,interval='95% percentile',multiplicity_adjusted=False),
        cost=dict(estimated_total_usd=round(sum(r['estimated_cost_usd'] or 0 for r in rows),6), actual_total_usd=None,
                  estimate_source='Frozen runner registry x recorded token fields; input fallback possible',
                  actual_cost_evidence='No independently verifiable billing evidence in this input package'),
        limits=['No raw response text or invalid-slot positions','No exact parser module','No original per-run library manifest',
                'Single-gold exposed queries, no multi-gold adjudication','Possible name collisions','Unknown discarded attempts/retries',
                'No billed-usage or cache-token evidence'])
    after = {str(p.relative_to(ROOT)):sha256(p) for p in inputs}
    if after != before:
        raise RuntimeError('Input changed while audit was running; outputs not written')
    out_dir.mkdir(parents=True,exist_ok=True)
    write_csv(out_dir / 'summary.csv',summaries)
    (out_dir / 'metrics.json').write_text(json.dumps(dict(schema_version=1, source_commit=SOURCE_COMMIT, unit='model x library_size x cohort; null means unavailable', summaries=summaries),indent=2,sort_keys=True,allow_nan=False)+'\n')
    write_csv(out_dir / 'common_panel_summary.csv',common_summaries)
    write_csv(out_dir / 'paired_differences.csv',pairs)
    write_csv(out_dir / 'size_differences.csv',paired_size_comparisons(rows,cfg,replicates))
    (out_dir/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    with (out_dir/'per_query.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    report(audit,summaries,out_dir/'REPORT.md')
    print(json.dumps(audit['counts'],sort_keys=True))
    print('Estimated US$',audit['cost']['estimated_total_usd'],'; actual billed cost unavailable')
    return audit


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=STUDY/'results')
    parser.add_argument('--bootstrap-replicates',type=int,default=BOOTSTRAP_REPLICATES)
    args=parser.parse_args()
    if args.bootstrap_replicates<100:
        parser.error('Use at least 100 bootstrap replicates')
    run(args.output_dir,args.bootstrap_replicates)
