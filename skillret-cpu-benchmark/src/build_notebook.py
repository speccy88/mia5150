"""Inference-free analysis of fresh, bounded SkillRet recovery measurements.

No old raw results are reconstructed. Retained original aggregate observations
are a clearly separate, non-recomputable appendix. --build-only is safe during
inference; execute only after the parent confirms inference has stopped.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mrr5', 'recall1', 'recall5', 'hit1', 'hit5', 'ndcg5', 'complete5', 'candidate_recall', 'candidate_hit')
KEYS = ['model', 'method', 'catalog_size']
RETRIEVAL_MODELS = ['bm25', 'bge-small', 'e5-small', 'embeddinggemma-int8']
DECISION_MODELS = ['julia-1', 'gliner2.5-small', 'verdict-151m-onnx', 'laya-multilingual']
MODEL_ALIASES = {'julia': 'julia-1', 'gliner': 'gliner2.5-small', 'verdict': 'verdict-151m-onnx', 'laya': 'laya-multilingual'}
EXECUTION_NOTE = ('The exact ordinary Python cells were executed sequentially in-process and print/display output was captured. '
                  'No Jupyter kernel was started in recovery. The original environment blocked kernel socket binding, so this retains the honest in-process validation boundary. '
                  'No kernel-based execution is claimed, no sandbox control changed, and no model inference or download was performed.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cohort_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def record_hash(root, path, hashes):
    path = Path(path)
    hashes[str(path.relative_to(root))] = sha256(path)


def metric_values(ranked, gold):
    """Independent implementation; never import the runner's metrics."""
    require(bool(gold), 'Gold denominator must be nonempty')
    relevant = [int(skill in gold) for skill in ranked[:5]]
    first = next((rank for rank, value in enumerate(relevant, 1) if value), None)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(gold)) + 1))
    found = len(set(ranked) & gold)
    return dict(mrr5=1 / first if first else 0.0,
                recall1=(relevant[0] if relevant else 0) / len(gold), recall5=sum(relevant) / len(gold),
                hit1=float(bool(relevant and relevant[0])), hit5=float(any(relevant)),
                ndcg5=sum(value / math.log2(rank + 1) for rank, value in enumerate(relevant, 1)) / ideal,
                complete5=float(sum(relevant) == len(gold)), candidate_recall=found / len(gold), candidate_hit=float(found > 0))


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream)


def jsonl_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def completed_result(root, path, hashes, archive_verified=()):
    """Read only committed rows. A growing or partial CSV is not evidence."""
    filename = path.name.replace('_per_query.csv', '_run.json').replace('_latency.csv', '_run.json')
    if filename == path.name:
        filename = 'run.json'
    receipt_path = path.with_name(filename)
    if not path.exists() or not receipt_path.exists():
        return None, 'missing raw CSV or completion receipt'
    receipt = read_json(receipt_path)
    if receipt.get('pilot') or receipt.get('partial') or receipt.get('completed') is False or receipt.get('status', 'completed') not in ('completed', 'complete'):
        return None, 'pilot, partial or explicitly incomplete receipt'
    if path.stat().st_mtime_ns > receipt_path.stat().st_mtime_ns and str(path.relative_to(root)) not in archive_verified:
        return None, 'raw file newer than completion receipt'
    record_hash(root, path, hashes)
    record_hash(root, receipt_path, hashes)
    return receipt, 'completion receipt verified'


def find_result(root, directory, model, kind):
    suffix = 'per_query' if kind == 'quality' else 'latency'
    stems = [model] + [alias for alias, canonical in MODEL_ALIASES.items() if canonical == model]
    candidates = [path for stem in stems for path in [root / 'results' / directory / f'{stem}_{suffix}.csv', root / 'results' / directory / stem / f'{suffix}.csv']]
    existing = [path for path in candidates if path.exists()]
    require(len(existing) <= 1, f'Ambiguous result paths: {directory}/{model}/{suffix}')
    return existing[0] if existing else candidates[0]


def load_context(root):
    """Verify the fresh64 inputs and their pinned reconstruction manifest."""
    dataset_root = root / 'data/processed'
    manifest_path = dataset_root / 'manifest.json'
    manifest = read_json(manifest_path)
    hashes = {}
    record_hash(root, manifest_path, hashes)
    for name, entry in manifest['artifacts'].items():
        path = dataset_root / name
        require(path.exists() and sha256(path) == entry['sha256'], f'Processed input hash mismatch: {name}')
        record_hash(root, path, hashes)
    queries = {row['query_id']: row for row in jsonl_rows(dataset_root / 'queries.jsonl')}
    all_queries = {row['query_id']: row for row in jsonl_rows(dataset_root / 'allqueries.jsonl')}
    require(len(all_queries) == manifest['source_counts']['queries'], 'Full source query count changed')
    catalog = read_json(dataset_root / 'full_catalog.json')
    require(len(catalog) == len(set(catalog)) == manifest['source_counts']['skills'], 'Duplicate/missing catalog skills')
    skill_ids = set(catalog)
    all_gold = {qid: set() for qid in all_queries}
    qrel_count = 0
    for row in jsonl_rows(dataset_root / 'qrels_all.jsonl'):
        qid, sid = row['query_id'], row['skill_id']
        require(qid in all_gold and sid in skill_ids and row['relevance'] == 1, 'Unknown/nonbinary qrel')
        require(sid not in all_gold[qid], 'Duplicate qrel pair')
        all_gold[qid].add(sid)
        qrel_count += 1
    require(qrel_count == manifest['source_counts']['qrels'], 'All-source qrel count changed')
    gold = {qid: set() for qid in queries}
    for row in jsonl_rows(dataset_root / 'qrels.jsonl'):
        qid, sid = row['query_id'], row['skill_id']
        require(qid in gold and sid not in gold[qid], 'Unknown/duplicate selected qrel')
        gold[qid].add(sid)
    for qid, query in queries.items():
        require(bool(gold[qid]) and gold[qid] == all_gold[qid] == set(query['gold_skill_ids']), 'Selected and full released qrels disagree')
        require(query['text'] == all_queries[qid]['text'], 'Selected query text changed')
    frozen = read_json(dataset_root / 'selected_query_ids.json')
    expensive = read_json(dataset_root / 'expensive_query_ids.json')
    target_count = manifest['cohort_counts']['main']
    checkpoint_count = manifest['cohort_counts']['latency_and_fallback']
    require(len(frozen) == target_count and len(frozen) == len(set(frozen)), 'Selected cohort count/uniqueness changed')
    seed = manifest['query_selection']['salt']
    ordered = sorted(all_queries, key=lambda qid: (hashlib.sha256((seed + '\0' + qid).encode()).hexdigest(), qid))
    require(expensive == ordered[:len(expensive)] and frozen == expensive[:target_count], 'Frozen hash order changed')
    require(list(queries) == frozen, 'Saved selected query order changed')
    timing_ids = read_json(dataset_root / 'latency_query_ids.json')
    require(timing_ids == frozen[:checkpoint_count], 'Timing IDs changed')
    sizes = [10, 100, 1000, len(catalog)]
    pools = {}
    for row in jsonl_rows(dataset_root / 'catalogs.jsonl'):
        qid, ids = row['query_id'], row['candidate_skill_ids']
        require(qid not in pools and qid in queries and len(ids) == 1000 and len(ids) == len(set(ids)), 'Invalid per-query sampled membership')
        require(set(ids) <= skill_ids and gold[qid] <= set(ids[:10]), 'Unknown candidates or lost gold in nested pool')
        pools[qid] = ids
    require(set(pools) == set(queries), 'Sampled catalog query coverage differs')
    for pattern in ['src/common.py', 'src/embeddings.py', 'src/decision_adapters.py', 'src/run_retrieval.py', 'src/run_decision.py',
                    'docs/model_sources.json', 'docs/retained_observations.json', 'protocol/DATA_PROTOCOL.md', 'protocol/recovery_plan.json', 'protocol/extension_choice.json', 'docs/notebook_execution_status.json']:
        if (root / pattern).exists(): record_hash(root, root / pattern, hashes)
    terminal_path = root / 'docs/notebook_execution_status.json'
    terminal_status = read_json(terminal_path) if terminal_path.exists() else None
    return dict(protocol=manifest, terminal_status=terminal_status, queries=queries, all_query_count=len(all_queries), gold=gold, catalog=skill_ids, frozen=frozen,
                sizes=sizes, target_count=target_count, intermediate_counts=[48], checkpoint_count=checkpoint_count, pools=pools,
                timing_ids=timing_ids, timing_repeats=2, hashes=hashes, baselines={}, decision_ranks={})


def read_quality(root, path, context, cohort):
    receipt, reason = completed_result(root, path, context['hashes'])
    if receipt is None:
        return pd.DataFrame(), None, reason
    records, seen = [], set()
    for row in csv_rows(path):
        model, method, qid, size = MODEL_ALIASES.get(row['model'], row['model']), row['method'], row['query_id'], int(row['catalog_size'])
        key = model, method, size, qid
        require(key not in seen, f'Duplicate quality key: {path}')
        seen.add(key)
        require('qwen' not in model.lower(), 'Excluded Qwen output cannot enter the analysis')
        require(qid in context['gold'] and size in context['sizes'], f'Unknown quality identity: {path}')
        ranked = json.loads(row['ranked_skill_ids'])
        require(isinstance(ranked, list) and len(ranked) == len(set(ranked)), 'Duplicate/invalid rank IDs')
        require(set(ranked) <= context['catalog'] and len(ranked) <= min(size, 20), 'Invalid candidate IDs or depth')
        if size < len(context['catalog']):
            require(set(ranked) <= set(context['pools'][qid][:size]), 'Ranked ID outside this query/catalog membership')
        gold = context['gold'][qid]
        require(int(row['gold_count']) == len(gold), 'Wrong gold denominator')
        group = 'multi' if len(gold) > 1 else 'single'
        require(row.get('query_group', group) == group, 'Wrong single/multi segment')
        calculated = metric_values(ranked, gold)
        for metric, value in calculated.items():
            require(math.isfinite(float(row[metric])) and math.isclose(float(row[metric]), value, abs_tol=1e-10, rel_tol=1e-10),
                    f'{metric} disagrees with raw ranks and released qrels: {path}/{qid}/{size}')
        result = dict(model=model, method=method, query_id=qid, catalog_size=size, cohort=cohort,
                      gold_count=len(gold), query_group=group, **calculated)
        if row.get('routing_hit') not in ('', None):
            result['routing_hit'] = float(row['routing_hit'])
            require(result['routing_hit'] in (0., 1.), 'Invalid routing-hit flag')
        if model == 'bge-small' and method == 'hybrid':
            context['baselines'][qid, size] = tuple(ranked)
        if method == 'hybrid_rerank':
            baseline = context['baselines'].get((qid, size))
            require(baseline is not None and set(baseline) == set(ranked), 'Reranker shortlist differs from verified BGE hybrid')
            initial = metric_values(baseline, gold)['mrr5']
            require(math.isclose(float(row['shortlist_mrr5']), initial, abs_tol=1e-10), 'Wrong saved shortlist MRR')
            context.setdefault('decision_ranks', {})[cohort, model, qid, size] = tuple(ranked)
            result.update(shortlist_mrr5=initial, mrr5_delta=calculated['mrr5'] - initial,
                          mrr5_headroom=calculated['candidate_hit'] - initial)
        records.append(result)
    return pd.DataFrame(records), receipt, reason


def exact_grid(frame, model, methods, sizes, ids):
    if frame.empty:
        return False
    required = {(model, method, size, qid) for method in methods for size in sizes for qid in ids}
    observed = set(frame[KEYS + ['query_id']].itertuples(index=False, name=None))
    return len(frame) == len(observed) and observed == required


def quality_counts(context):
    """Outcome-independent, preregistered preference: common64, then48, then32."""
    upper, lower = context['target_count'], context['checkpoint_count']
    middle = [count for count in context.get('intermediate_counts', [48]) if lower < count < upper]
    return list(dict.fromkeys([upper, *sorted(middle, reverse=True), lower]))


def choose_common(context, retrieval, decisions):
    coverage, selected = [], None
    for count in quality_counts(context):
        ids = context['frozen'][:count]
        flags = []
        for model in RETRIEVAL_MODELS:
            frame = retrieval.get(model, pd.DataFrame())
            subset = frame[frame.query_id.isin(ids)] if not frame.empty else frame
            methods = ['bm25'] if model == 'bm25' else ['vector', 'hybrid', 'domain_vector']
            good = exact_grid(subset, model, methods, context['sizes'], ids)
            coverage.append(dict(queries=count, model=model, family='retrieval', complete=good))
            flags.append(good)
        for model in DECISION_MODELS:
            good = exact_grid(decisions.get((count, model), pd.DataFrame()), model, ['hybrid_rerank'], context['sizes'], ids)
            coverage.append(dict(queries=count, model=model, family='binary adapter', complete=good))
            flags.append(good)
        if selected is None and all(flags):
            selected = count
    return selected, pd.DataFrame(coverage)


def load_evidence(root, context):
    retrieval, decisions, receipts, status = {}, {}, {}, []
    for model in RETRIEVAL_MODELS:
        path = find_result(root, 'retrieval64', model, 'quality')
        frame, receipt, reason = read_quality(root, path, context, 'fresh-retrieval-available')
        if not frame.empty:
            require(set(frame.query_id) <= set(context['frozen'][:context['target_count']]), 'Fresh retrieval includes unplanned query IDs')
        retrieval[model] = frame
        if receipt is not None:
            receipts['retrieval', model] = receipt
        status.append(dict(stage='retrieval', model=model, receipt_available=receipt is not None, reason=reason))
    for count in reversed(quality_counts(context)):
        for model in DECISION_MODELS:
            path = find_result(root, f'quality{count}', model, 'quality')
            frame, receipt, reason = read_quality(root, path, context, f'fresh-{count}')
            decisions[count, model] = frame
            if receipt is not None:
                receipts[f'quality{count}', model] = receipt
            status.append(dict(stage=f'quality{count}', model=model, receipt_available=receipt is not None, reason=reason))
    selected, coverage = choose_common(context, retrieval, decisions)
    primary = []
    if selected:
        ids = context['frozen'][:selected]
        primary += [frame[frame.query_id.isin(ids)].copy() for frame in retrieval.values()]
        primary += [decisions[selected, model] for model in DECISION_MODELS]
    if context.get('terminal_status'):
        require(selected == context['terminal_status']['primary_query_count'], 'Exact selected cohort differs from final execution scope')
    common = pd.concat(primary, ignore_index=True) if primary else pd.DataFrame()
    if not common.empty:
        common['cohort'] = f'fresh-common-{selected}'
    return dict(retrieval=retrieval, decisions=decisions, receipts=receipts, status=pd.DataFrame(status),
                selected=selected, coverage=coverage, common=common)


def aggregate(frame, extra_keys=()):
    if frame.empty:
        return pd.DataFrame()
    keys = ['cohort'] + KEYS + list(extra_keys)
    metrics = [metric for metric in (*METRICS, 'shortlist_mrr5', 'mrr5_delta', 'mrr5_headroom', 'routing_hit') if metric in frame.columns]
    return frame.groupby(keys, dropna=False).agg(queries=('query_id', 'size'), **{metric: (metric, 'mean') for metric in metrics}).reset_index()


def bootstrap(delta, replicates=2000, seed=20261002):
    delta = np.asarray(delta, dtype=float)
    require(delta.ndim == 1 and len(delta) > 0 and np.isfinite(delta).all(), 'Invalid paired differences')
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(replicates)])
    low, high = np.quantile(samples, [.025, .975])
    return dict(queries=len(delta), mrr5_delta=float(delta.mean()), ci95_low=float(low), ci95_high=float(high),
                wins=int((delta > 1e-12).sum()), ties=int((np.abs(delta) <= 1e-12).sum()), losses=int((delta < -1e-12).sum()),
                seed=seed, bootstrap_replicates=replicates)


def paired_comparisons(evidence):
    common, comparisons = evidence['common'], []
    if common.empty:
        return pd.DataFrame()
    for size in sorted(common.catalog_size.unique()):
        subset = common[common.catalog_size == size]
        baseline = subset[(subset.model == 'bm25') & (subset.method == 'bm25')][['query_id', 'mrr5']]
        for (model, method), frame in subset.groupby(['model', 'method']):
            if method == 'bm25':
                continue
            if method == 'hybrid_rerank':
                differences = frame.mrr5 - frame.shortlist_mrr5
                baseline_id = 'bge-small/hybrid'
            else:
                joined = frame[['query_id', 'mrr5']].merge(baseline, on='query_id', validate='one_to_one', suffixes=('_target', '_baseline'))
                require(len(joined) == len(frame) == len(baseline), 'Paired merge changed query denominator')
                differences = joined.mrr5_target - joined.mrr5_baseline
                baseline_id = 'bm25/bm25'
            comparisons.append(dict(model=model, method=method, catalog_size=int(size), cohort=f'fresh-common-{evidence["selected"]}',
                                    baseline_id=baseline_id, candidate_depth=min(int(size), 20), cohort_hash=cohort_hash(frame.query_id),
                                    **bootstrap(differences)))
    return pd.DataFrame(comparisons)


def exact_timing(frame, ids, repeats):
    required = {(qid, repeat) for qid in ids for repeat in range(repeats)}
    seen = set(frame[['query_id', 'repeat']].itertuples(index=False, name=None)) if not frame.empty else set()
    return len(frame) == len(required) and seen == required


def read_timing(root, path, context, model, quality_receipt=None):
    receipt, reason = completed_result(root, path, context['hashes'])
    if receipt is None:
        return pd.DataFrame(), reason
    if model in DECISION_MODELS:
        require(quality_receipt is not None, f'No completed quality receipt to validate timing configuration: {model}')
        for key in ['configuration', 'score_cache_fingerprint']:
            require(key in receipt and key in quality_receipt and receipt[key] == quality_receipt[key], f'Quality/timing {key} mismatch: {model}')
    records, seen = [], set()
    for row in csv_rows(path):
        qid, repeat, size = row['query_id'], int(row.get('repeat', 0)), int(row['catalog_size'])
        key = row['method'], size, qid, repeat
        require(key not in seen and MODEL_ALIASES.get(row['model'], row['model']) == model, 'Duplicate/incorrect timing identity')
        seen.add(key)
        require(qid in context['timing_ids'] and size == 6006 and 0 <= repeat < context['timing_repeats'], 'Timing observation outside 32×2 full-catalog scope')
        field = 'full_pipeline_ms' if row.get('full_pipeline_ms') not in ('', None) else 'total_ms'
        value = float(row[field])
        require(math.isfinite(value) and value > 0, 'Nonfinite/nonpositive timing')
        record = dict(model=model, method=row['method'], catalog_size=size, query_id=qid, repeat=repeat, total_ms=value)
        for stage in ['stage1_ms', 'rerank_ms', 'query_encode_ms']:
            record[stage] = float(row[stage]) if row.get(stage) not in ('', None) else np.nan
            require(math.isnan(record[stage]) or 0 <= record[stage] <= value + 1e-5, 'Invalid timing stage')
        if model in DECISION_MODELS:
            require(row.get('shortlist_matches_frozen') == 'True', 'Fresh shortlist equality not verified in timing')
            require(tuple(json.loads(row['fresh_shortlist_ids'])) == context['baselines'][qid, size], 'Timed shortlist differs from BGE quality ranking')
        records.append(record)
    return pd.DataFrame(records), reason


def load_timings(root, context, evidence):
    summaries, coverage = [], []
    for model in RETRIEVAL_MODELS + DECISION_MODELS:
        path = find_result(root, 'timing32' if model in DECISION_MODELS else 'retrieval64', model, 'timing')
        if model in DECISION_MODELS:
            quality_stages = [f'quality{evidence["selected"]}'] + [f'quality{count}' for count in quality_counts(context)]
            quality_receipt = next((evidence['receipts'][stage, model] for stage in quality_stages if (stage, model) in evidence['receipts']), None)
        else:
            quality_receipt = evidence['receipts'].get(('retrieval', model))
        frame, reason = read_timing(root, path, context, model, quality_receipt)
        methods = ['bm25'] if model == 'bm25' else (['hybrid_rerank'] if model in DECISION_MODELS else ['vector', 'hybrid', 'domain_vector'])
        if not frame.empty:
            require(set(frame.method) <= set(methods), f'Unexpected timing method: {model}')
        for method in methods:
            part = frame[frame.method == method] if not frame.empty else frame
            good = exact_timing(part, context['timing_ids'], context['timing_repeats'])
            coverage.append(dict(model=model, method=method, complete=good, reason=reason if frame.empty else 'exact query×repeat grid checked'))
            if not good:
                continue
            values = part.total_ms.to_numpy()
            summaries.append(dict(model=model, method=method, catalog_size=6006, queries=len(context['timing_ids']),
                                  repeats=context['timing_repeats'], observations=len(part), cohort_hash=cohort_hash(context['timing_ids']),
                                  mean_ms=float(values.mean()), p50_ms=float(np.quantile(values, .5)), p95_ms=float(np.quantile(values, .95)),
                                  min_ms=float(values.min()), max_ms=float(values.max()), serial_qps=float(1000 / values.mean()),
                                  stage1_p50_ms=float(part.stage1_ms.median()) if part.stage1_ms.notna().any() else None,
                                  rerank_p50_ms=float(part.rerank_ms.median()) if part.rerank_ms.notna().any() else None))
    return pd.DataFrame(summaries), pd.DataFrame(coverage)


def setup_summary(evidence):
    rows = []
    for (stage, model), receipt in evidence['receipts'].items():
        config = receipt.get('model_configuration', receipt.get('configuration', {}))
        rows.append(dict(stage=stage, model=model, runtime=config.get('runtime'), precision=config.get('precision'),
            pooling=config.get('pooling'), max_tokens=config.get('max_length'), threads=receipt.get('hardware', {}).get('threads'),
            load_seconds=config.get('load_seconds', receipt.get('load_seconds')),
            document_encoding_seconds=config.get('documents', {}).get('encoding_seconds'),
            query_batch_encoding_seconds=config.get('queries', {}).get('encoding_seconds'),
            bm25_index_seconds=receipt.get('bm25_index_seconds'), peak_rss_mib=receipt.get('peak_rss_mib'),
            new_pairs_encoded=receipt.get('new_pair_count'), batch_inference_seconds=receipt.get('new_pair_seconds')))
    return pd.DataFrame(rows)


def pair_audits(root, context, evidence):
    """Count unique pairs once and verify native score sorting without inference."""
    count = evidence['selected']
    if not count:
        return pd.DataFrame()
    selected = context['frozen'][:count]
    expected = {(qid, sid) for qid in selected for size in context['sizes'] for sid in context['baselines'][qid, size]}
    rows = []
    for model in DECISION_MODELS:
        alias = next(key for key, value in MODEL_ALIASES.items() if value == model)
        path = root / 'results/pair_scores' / f'{alias}.jsonl'
        metadata_path = path.with_name(f'{alias}_metadata.json')
        if not path.exists() or not metadata_path.exists():
            rows.append(dict(model=model, expected_unique_pairs=len(expected), available=False))
            continue
        metadata = read_json(metadata_path)
        receipt = evidence['receipts'][f'quality{count}', model]
        require(metadata['fingerprint'] == receipt['score_cache_fingerprint'], f'Pair-source fingerprint mismatch: {model}')
        require(metadata['configuration'] == receipt['configuration'], f'Pair-source configuration mismatch: {model}')
        pairs = {}
        for row in jsonl_rows(path):
            key = row['query_id'], row['skill_id']
            if key in expected:
                require(key not in pairs and math.isfinite(float(row['score'])), f'Duplicate/nonfinite selected native score: {model}')
                pairs[key] = row
        require(set(pairs) == expected, f'Missing selected native score pairs: {model}')
        for qid in selected:
            for size in context['sizes']:
                original = context['baselines'][qid, size]
                sorted_ids = tuple(sorted(original, key=lambda sid: -float(pairs[qid, sid]['score'])))
                require(sorted_ids == context['decision_ranks'][f'fresh-{count}', model, qid, size], f'Native ranking differs from saved score/tie policy: {model}/{qid}/{size}')
        audits = [row.get('audit', {}) for row in pairs.values()]
        limit = 512 if alias == 'verdict' else 1024
        rows.append(dict(model=model, expected_unique_pairs=len(expected), available=True,
            audits_present=sum(bool(audit) for audit in audits),
            query_shortened=sum(bool(audit.get('query_truncated')) for audit in audits),
            candidate_cut=sum(audit.get('candidate_preserved') is False for audit in audits),
            candidate_preservation_not_recorded=sum('candidate_preserved' not in audit for audit in audits),
            residual_truncation=sum(bool(audit.get('stateTruncated') or audit.get('optionsTruncated') or audit.get('tokens', 0) > limit) for audit in audits),
            mean_abstain_probability=float(np.mean([audit['p_abstain'] for audit in audits if 'p_abstain' in audit])) if any('p_abstain' in audit for audit in audits) else None))
        record_hash(root, path, context['hashes']);record_hash(root, metadata_path, context['hashes'])
    return pd.DataFrame(rows)


def retained_history(root):
    candidates = [root / 'docs/retained_observations.json', root.parent / 'skillret-recovery/retained_observations.json']
    path = next((path for path in candidates if path.exists()), None)
    return read_json(path) if path else {}


def observations(context, evidence, quality, latency):
    lines = [f"Fresh recovery snapshot: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}.",
             'Original 4,392-query measurements were completed and independently audited before their raw outputs became unavailable. Retained aggregates are genuinely observed historical results, but cannot now be recomputed or bootstrapped here.']
    if context.get('terminal_status'):
        lines.append(context['terminal_status']['stop_reason'] + ' The attempted48 extension is not a completed all-model comparison; its Julia/GLiNER completions remain separate supplements.')
    if not evidence['selected']:
        lines.append('No complete common fresh 64-, 48- or 32-query panel exists yet. Available completed checkpoints are shown separately; no missing model result is inferred.')
    else:
        lines.append(f"Primary fresh comparison: the same first {evidence['selected']} frozen queries across all seven models plus BM25 and all four catalog sizes.")
        full = quality[quality.catalog_size == 6006]
        retrieval = full[full.method != 'hybrid_rerank'].sort_values('mrr5', ascending=False)
        if len(retrieval):
            best = retrieval.iloc[0]
            lines.append(f"Highest observed fresh full-catalog retrieval MRR@5: {best.model}/{best.method}, {best.mrr5:.4f} (Recall@5 {best.recall5:.4f}); this small-sample maximum is descriptive.")
        for row in full[full.method == 'hybrid_rerank'].itertuples():
            lines.append(f"{row.model}: MRR@5 {row.mrr5:.4f}, paired shortlist change {row.mrr5_delta:+.4f}; initial MRR {row.shortlist_mrr5:.4f}, candidate-any-hit ceiling {row.candidate_hit:.4f}.")
    lines.append(f"Complete matched latency conditions: {len(latency)} on the same 32 IDs × 2 repeats at 6,006 skills. Unmeasured timing is never filled in.")
    lines.append('Negative results describe the fixed binary adapters and shortlist, not general model capabilities. Query-bootstrap intervals are descriptive, with no training/hardware or out-of-domain guarantee.')
    return '\n'.join('- ' + line for line in lines)


def markdown(text):
    return dict(cell_type='markdown', metadata={}, source=text)


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text)


def helper_source(functions):
    return '\n\n'.join(inspect.getsource(function) for function in functions)


def make_notebook():
    cells = [markdown('# SkillRet CPU recovery benchmark\n\n## tl;dr\n\nAnalysis not yet executed. No fresh results claimed.'),
    markdown('''## Context & Methods
### Recovery and evidence boundary
The original full-test run's working tree became unavailable. This notebook evaluates newly saved raw measurements from a bounded recovery run. The previous 4,392-query aggregates and eight-query checkpoint, where retained, are historical observations in a separate appendix. They were genuinely measured and independently audited before interruption; their original raw files are now unavailable. They are not silently substituted for fresh observations or used for paired uncertainty.

### Key assumptions
- Pinned SkillRet v1.1: 6,006 headers, 4,392 synthetic English queries and binary qrels. Documents use name plus 1,024 description characters, without full instructions. This is not an official full-text leaderboard reproduction
- Fresh primary: first 64 outcome-independent frozen IDs for every model, otherwise the complete common first48, otherwise common32. The first48 option was preregistered from the same frozen order before new extension-quality results, based on measured runtime budget. Retrieval-only or individual larger checkpoints remain separate from the primary panel
- Catalogs 10/100/1,000 are nested per-query gold-preserving samples; 6,006 is the full catalog. Catalog membership order is not a scoring feature
- BM25 uses k1=1.2/b=0.75 with local corpus statistics. Hybrid uses equal BM25+dense RRF, k=60, depth 100; ranking returns up to 20 candidates. Domain-vector selects one metadata major-category centroid without gold query routing
- BGE: publisher retrieval query instruction, CLS pooling, L2. E5: query:/passage: prefixes, attention-mask mean, L2. Gemma: task: search result | query: and title: none | text: prefixes, exported sentence_embedding with projection, L2. Exact runtime sources are pinned by the runner
- Four decision models apply the fixed native binary-relevance adapters to the shared BGE-hybrid shortlist. Candidate/label text is preserved; query-only truncation fits token budgets. Score ties retain shortlist order. Negative results are configuration-specific, not general capability judgments
- Warm CPU timing uses the same 32 queries × 2 repeats at the full catalog. It includes fresh retrieval and uncached candidate scoring. Setup, cached/batch accuracy work and model loading are separate. Historical timings and fresh timings are never pooled

### Metrics and uncertainty
MRR@5 is reciprocal first relevant rank within five; Recall@1/5 is recovered fraction of each query's gold set; Hit@1/5 is any-gold success; binary nDCG@5 uses an ideal list of min(5, gold count); Complete@5 requires every gold skill. Candidate recall/hit use the entire saved shortlist. Candidate-any-hit is the optimistic MRR ceiling. All reported quality values are per-query macro means with explicit denominators.

p50/p95 are linear percentiles; serial QPS is 1,000/mean milliseconds, not concurrent capacity. Peak RSS is process high-water memory. Paired percentile query-bootstrap intervals use 2,000 replacement resamples, seed 20261002; they are unadjusted for multiple comparisons and do not express training/hardware uncertainty or out-of-domain guarantees.

### Reproduction
Run `python src/build_notebook.py --python-execute` only after inference stops. This executes ordinary Python cells in-process without a socket-binding Jupyter kernel or any model import/download. A build-only scaffold is unexecuted. Rebuild to refresh the top narrative and source hashes.'''),
    markdown('## Data\n### 1. Local setup and independent validation helpers'),
    code("from pathlib import Path\nimport csv, datetime as dt, hashlib, json, math\nimport numpy as np\nimport pandas as pd\nROOT=Path.cwd()\nif not (ROOT/'data/processed').exists() and (ROOT.parent/'data/processed').exists(): ROOT=ROOT.parent\nMETRICS="+repr(METRICS)+"\nKEYS="+repr(KEYS)+"\nRETRIEVAL_MODELS="+repr(RETRIEVAL_MODELS)+"\nDECISION_MODELS="+repr(DECISION_MODELS)+"\nMODEL_ALIASES="+repr(MODEL_ALIASES)),
    code(helper_source([require, read_json, sha256, cohort_hash, record_hash, metric_values, csv_rows, jsonl_rows, completed_result, find_result, load_context, read_quality, exact_grid, quality_counts, choose_common, load_evidence, aggregate])),
    code("context=load_context(ROOT)\nevidence=load_evidence(ROOT,context)\nprint('Fresh selected queries:',len(context['queries']),'Full source queries:',context['all_query_count'],'Skills:',len(context['catalog']),'Qrels:',sum(map(len,context['gold'].values())))\nprint('Selected fresh common queries:',evidence['selected'])\ndisplay(evidence['status'])\ndisplay(evidence['coverage'])"),
    markdown('## Results\n### 2. Fresh exact-cohort quality and scaling\nAll nine metrics are independently recomputed from ranked IDs and released qrels. Primary comparisons require every model and catalog to contain the identical selected query IDs.'),
    code("quality=aggregate(evidence['common'])\ndisplay(quality.round(5))\nif not quality.empty:\n    display(quality.pivot(index=['model','method'],columns='catalog_size',values='mrr5').round(5))"),
    markdown('### 3. Separate completed checkpoint evidence\nThese larger or earlier individual results are not pooled with another model’s smaller sample.'),
    code("checkpoints=[]\nfor model,frame in evidence['retrieval'].items():\n    if not frame.empty: checkpoints.append(aggregate(frame))\nfor (count,model),frame in evidence['decisions'].items():\n    if exact_grid(frame,model,['hybrid_rerank'],context['sizes'],context['frozen'][:count]): checkpoints.append(aggregate(frame))\ndisplay(pd.concat(checkpoints,ignore_index=True).round(5) if checkpoints else pd.DataFrame())"),
    markdown('### 4. Paired uncertainty and candidate ceiling\nDense retrieval compares with BM25 on the same fresh IDs. Binary adapters compare with the original verified BGE-hybrid shortlist. No bootstrap is reconstructed from old aggregate-only observations.'),
    code(helper_source([bootstrap, paired_comparisons])),
    code("paired=paired_comparisons(evidence)\ndisplay(paired.round(5))"),
    markdown('### 5. Single- and multi-gold segments\nRecall@1 differs from Hit@1 for multi-positive queries. Counts are query denominators, not independent replications across systems.'),
    code("segments=aggregate(evidence['common'],['query_group'])\ndisplay(segments[segments.catalog_size==6006].round(5) if not segments.empty else segments)"),
    markdown('### 6. Exact full-catalog 32×2 timing\nQuality and latency are different observations. Completion and source/configuration compatibility are checked separately; no smaller-catalog timing is invented.'),
    code(helper_source([exact_timing, read_timing, load_timings, setup_summary, pair_audits])),
    code("latency,timing_coverage=load_timings(ROOT,context,evidence)\ndisplay(timing_coverage)\ndisplay(latency.round(3))\nif not quality.empty and not latency.empty:\n    joined=quality[quality.catalog_size==6006].merge(latency,on=KEYS,how='left',validate='one_to_one',suffixes=('_quality','_timing'))\n    display(joined[['model','method','queries_quality','mrr5','recall5','complete5','queries_timing','observations','p50_ms','p95_ms','serial_qps']].round(4))"),
    markdown('### Setup, memory and pinned model sources\nLoad/index/corpus-encoding work is excluded from warm latency. Peak RSS is process high-water memory; repeated quality-stage receipts are not additive. Missing setup fields remain unavailable.'),
    code("display(pair_audits(ROOT,context,evidence))\nprint('Pair counts are unique across all catalog conditions; query shortening is distinct from candidate loss. Where pair sources are available, saved native ranking is independently reconciled to scores and stable tie ordering.')\ndisplay(setup_summary(evidence).round(3))\nsources_path=ROOT/'docs/model_sources.json'\nif sources_path.exists():\n    source_models=read_json(sources_path).get('models',[])\n    display(pd.DataFrame([{key:row.get(key) for key in ['key','repo','revision','task','pooling','query_prefix','document_prefix','max_tokens']} for row in source_models]))"),
    markdown('### 7. Historical observations, original raw data unavailable\nThe following are retained aggregate observations from the interrupted original run. They cannot be independently revalidated from raw rankings here, and are not primary fresh evidence. Original and recovery latency values are not pooled. The original common-eight shortlist had MRR=0.75 and candidate-any-hit=0.75, so no positive MRR headroom existed.'),
    code(helper_source([retained_history])),
    code("history=retained_history(ROOT)\nprint(history.get('status','No retained historical observations available'))\ndisplay(pd.DataFrame(history.get('full_catalog_aggregates',[])))\ndisplay(pd.DataFrame(history.get('fast_exploratory_common8',[])))\nprint('Original unequal 16/10-query raw observations are unavailable; no replacement rows or missing metrics are invented.')"),
    markdown('## Takeaways\nAnalysis execution pending.'),
    code(helper_source([observations])),
    code("summary_text=observations(context,evidence,quality,latency)\nprint(summary_text)\nprint('Limits: synthetic English short-header benchmark; incomplete qrels; one CPU host; fixed binary adapters; no downstream skill execution; small fresh cohort; original raw files unavailable.')\nsource_hashes=pd.DataFrame([dict(path=path,sha256=value) for path,value in sorted(context['hashes'].items())])\ndisplay(source_hashes)")]
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'), language_info={'name':'python'}), cells=cells)


def validate_notebook(notebook):
    require(notebook.get('nbformat') == 4 and isinstance(notebook.get('cells'), list), 'Invalid notebook structure')
    for index, cell in enumerate(notebook['cells']):
        cell.setdefault('id', f'analysis-{index:03d}')
        require(cell['cell_type'] in ('markdown', 'code') and isinstance(cell['source'], str), 'Invalid cell structure')
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'analysis-cell-{index}', 'exec')
            require(isinstance(cell['outputs'], list), 'Invalid outputs')
    # Use the official schema validator if the environment provides it. Do not
    # install software solely to hide the recorded validation boundary.
    try:
        import nbformat
    except ImportError:
        notebook['metadata']['schema_validation'] = 'Version-4 required structures and all code-cell syntax checked locally; nbformat is not installed in this recovery environment.'
    else:
        nbformat.validate(nbformat.from_dict(notebook))
        notebook['metadata']['schema_validation'] = 'Validated with nbformat.'


def execute_notebook(notebook, root):
    namespace, count = {}, 0
    previous = Path.cwd()
    try:
        os.chdir(root)
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            outputs, stream = [], io.StringIO()
            def flush():
                if stream.tell():
                    outputs.append(dict(output_type='stream', name='stdout', text=stream.getvalue()))
                    stream.seek(0); stream.truncate(0)
            def display(value):
                flush()
                data = {'text/plain': value.to_string() if hasattr(value, 'to_string') else repr(value)}
                if hasattr(value, 'to_html'):
                    data['text/html'] = value.to_html()
                outputs.append(dict(output_type='display_data', data=data, metadata={}))
            namespace['display'] = display
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell['source'], f'analysis-cell-{count}', 'exec'), namespace)
            flush()
            cell.update(outputs=outputs, execution_count=count)
    finally:
        os.chdir(previous)
    notebook['metadata']['execution_note'] = EXECUTION_NOTE
    notebook['cells'][0]['source'] = '# SkillRet CPU recovery benchmark\n\n## tl;dr\n\n' + namespace['summary_text']
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown' and cell['source'].startswith('## Takeaways'):
            cell['source'] = '## Takeaways\n\n' + namespace['summary_text']
    notebook['cells'][1]['source'] += '\n\n### Execution status\n' + EXECUTION_NOTE
    notebook['metadata']['analysis_snapshot'] = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selected_fresh_query_count=namespace['evidence']['selected'], input_sha256=namespace['context']['hashes'])
    return namespace


def export_bootstrap(namespace, output):
    evidence, context = namespace['evidence'], namespace['context']
    selected = evidence['selected']
    cohorts = {f'fresh-common-{selected}': context['frozen'][:selected]} if selected else {}
    payload = dict(schema_version=2, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Fresh recovery raw results only. Original 4392-query aggregate-only observations are excluded.',
        method='Paired query differences resampled with replacement; N draws per replicate; linear 2.5% and 97.5% percentiles of mean differences.',
        seed=20261002, bootstrap_replicates=2000, confidence_level=.95,
        uncertainty_scope='Descriptive query sampling only; no training/hardware uncertainty, causal evidence, out-of-domain guarantees or multiplicity adjustment.',
        cohort_hash_convention='SHA256 of sorted query IDs joined by newline, no trailing newline',
        cohort_query_ids=cohorts, selected_decision_query_count=selected,
        comparisons=namespace['paired'].to_dict('records'), source_sha256=namespace['source_hashes'].to_dict('records'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build-only', action='store_true')
    mode.add_argument('--python-execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/analysis.ipynb')
    args = parser.parse_args()
    notebook = make_notebook()
    validate_notebook(notebook)
    if args.python_execute:
        namespace = execute_notebook(notebook, ROOT)
        export_bootstrap(namespace, ROOT / 'docs/notebook_paired_bootstrap.json')
    else:
        notebook['metadata']['execution_note'] = 'Unexecuted build-only scaffold; no fresh observations claimed.'
    validate_notebook(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
