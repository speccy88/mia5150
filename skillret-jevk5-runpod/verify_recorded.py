"""Independently audit the recorded pilot using only the Python standard library."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NONE = '__none__'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def rows(path, key):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        require(row[key] not in result, f'Duplicate ID in {path.name}')
        result[row[key]] = row
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-12,
            f'Numeric mismatch: {label}')


def score(prediction, gold, candidate):
    ranked = prediction['ranked_skill_ids']
    require(len(candidate) == 15 and len(set(candidate)) == 15, 'Invalid shortlist')
    require(prediction['candidate_skill_ids'] == candidate, 'Candidate order changed')
    require(len(ranked) == 15 and set(ranked) == set(candidate), 'Candidate set changed')
    require(bool(gold), 'Empty relevance denominator')
    chosen = prediction['top_decision_skill_id']
    require(chosen in set(candidate) | {NONE}, 'Unknown top decision')
    out = {}
    for k in (1, 5, 10):
        found = set(ranked[:k]) & gold
        out[f'recall@{k}'] = len(found) / len(gold)
        out[f'hit@{k}'] = float(bool(found))
        out[f'completeness@{k}'] = float(found == gold)
        dcg = sum((s in gold) / math.log2(i + 2) for i, s in enumerate(ranked[:k]))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        out[f'ndcg@{k}'] = dcg / ideal
    out['mrr@15'] = next((1 / (i + 1) for i, s in enumerate(ranked) if s in gold), 0.0)
    present = len(set(candidate) & gold)
    out.update({
        'candidate_recall@15': present / len(gold),
        'candidate_hit@15': float(present > 0),
        'oracle_recall@5': min(5, present) / len(gold),
        'none_correct': float((chosen == NONE) == (present == 0)),
        'decision_correct': float(chosen in gold or (chosen == NONE and present == 0)),
        'abstained': float(chosen == NONE),
        'false_abstention': float(chosen == NONE and present > 0),
    })
    return out


def check_logits(prediction, mapping):
    logits = prediction['option_logits']
    probs = prediction['option_probabilities']
    require(set(logits) == set(mapping) == set(probs) and len(mapping) == 16,
            'Invalid answer option coverage')
    require(list(logits) == list(mapping), 'Answer presentation order changed')
    require(all(math.isfinite(x) for x in logits.values()), 'Nonfinite logit')
    maximum = max(logits.values())
    weights = {k: math.exp(v - maximum) for k, v in logits.items()}
    total = sum(weights.values())
    for k in weights:
        close(weights[k] / total, probs[k], 'option probability')
    skill_probs = {mapping[k]: probs[k] for k in mapping}
    require(prediction['ranked_skill_ids'] ==
            sorted((s for s in skill_probs if s != NONE), key=lambda s: (-skill_probs[s], s)),
            'Ranking disagrees with option probabilities')
    require(prediction['top_decision_skill_id'] == mapping[max(probs, key=probs.get)],
            'Top decision disagrees with option probabilities')
    close(prediction['none_probability'], skill_probs[NONE], 'NONE probability')


def check_metrics(predictions, labels, candidates, recorded):
    require(set(predictions) == set(labels) == set(candidates) and len(labels) == 256,
            'Missing, extra, or wrong number of queries')
    observed = {q: score(predictions[q], set(labels[q]), candidates[q]['retrieval_order'])
                for q in sorted(labels)}
    require(recorded['n'] == 256 and len(recorded['per_query']) == 256, 'Wrong metric denominator')
    retained = {r['query_id']: r for r in recorded['per_query']}
    require(len(retained) == 256 and set(retained) == set(labels), 'Duplicate or missing metric IDs')
    means = {k: sum(r[k] for r in observed.values()) / 256 for k in next(iter(observed.values()))}
    for q, metric in observed.items():
        for k, v in metric.items():
            close(v, retained[q][k], f'{q} {k}')
    for k, v in means.items():
        close(v, recorded['mean'][k], k)
    return observed, means


def bootstrap(before, after):
    differences = [after[q]['recall@5'] - before[q]['recall@5'] for q in sorted(before)]
    rng = random.Random(42)
    sampled = sorted(sum(differences[rng.randrange(256)] for _ in range(256)) / 256
                     for _ in range(2000))
    return sum(differences) / 256, [sampled[50], sampled[1950]]


def verify(root=ROOT):
    hashes = root / 'SHA256SUMS'
    paths = set()
    for line in hashes.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe manifest path')
        require(relative not in paths, 'Duplicate manifest member')
        paths.add(relative)
        require(sha(root / path) == expected, f'Hash mismatch: {relative}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and not any(x in p.parts for x in ['__pycache__', '.pytest_cache', '.venv-tests'])
              and p.name not in ['SHA256SUMS', '.DS_Store']}
    require(paths == actual, 'Published inventory differs from checksum manifest')
    run = root / 'reference_run'
    labels = read_json(run / 'data/validation_labels.json')
    candidates = rows(run / 'data/validation_candidates.jsonl', 'id')
    split = read_json(run / 'data/split.json')
    train = set(split['train_query_ids'])
    val = set(split['validation_query_ids'])
    families = split['family_by_query_id']
    require(len(train) == 56883 and len(val) == 6376 and not train & val, 'Invalid source split')
    require(train | val == set(families) and not {families[q] for q in train} &
            {families[q] for q in val}, 'Family leakage or incomplete split')
    require(set(labels) <= val, 'Validation rows outside validation fold')
    selected_train = read_json(run / 'data/training_ids.json')
    require(len(selected_train) == len(set(selected_train)) == 2000 and set(selected_train) <= train,
            'Invalid training selection')
    def selected(ids, n):
        def digest(q):
            return hashlib.sha256(json.dumps([42, q], separators=(',', ':')).encode()).hexdigest()
        return sorted(ids, key=lambda q: (digest(q), q))[:n]
    require(selected_train == selected(train, 2000), 'Training selection is not frozen hash sample')
    require(list(candidates) == selected(val, 256), 'Validation selection is not frozen hash sample')
    retrieved = rows(run / 'data/validation_top100.jsonl', 'query_id')
    require(set(retrieved) == set(labels), 'Missing retrieval provenance')
    for q, c in candidates.items():
        require(c['gold_injected'] is False and c['split'] == 'validation', 'Gold injection or wrong split')
        require(c['retrieval_order'] == retrieved[q]['skill_ids'][:15], 'Shortlist changed from natural top15')
        mapping = c['option_skill_ids']
        require(len(mapping) == 16 and len(set(mapping.values())) == 16 and
                set(mapping.values()) == set(c['retrieval_order']) | {NONE}, 'Invalid option mapping')
        displayed = c['retrieval_order'] + [NONE]
        seed = hashlib.sha256(json.dumps([42, q, 'order'], separators=(',', ':')).encode()).hexdigest()
        random.Random(int(seed[:16], 16)).shuffle(displayed)
        require(list(mapping.values()) == displayed, 'Option shuffle changed')
    configurations = read_json(run / 'resumed/config.json')
    selection = read_json(run / 'resumed/selection.json')
    config_hash = hashlib.sha256(json.dumps(configurations, sort_keys=True,
                                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    require(config_hash == selection['config_sha256'], 'Changed training contract')
    manifest = read_json(run / 'data/manifest.json')
    data_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                         separators=(',', ':')).encode()).hexdigest()
    require(data_hash == selection['data_sha256'] and not manifest['is_test'], 'Changed data contract')
    require(selection['selected']['step'] == 100 and not selection['test_used_for_selection'],
            'Wrong selection step or official test selection')
    require(selection['status'] == 'completed_requested_schedule', 'Incomplete requested schedule')
    require(selection['dataset_revision'] == 'a050ad233a504a43135bafe8cdf45574052b5729', 'Wrong data pin')
    for name, expected in selection['adapter_files'].items():
        require(sha(root / 'selected_adapter' / name) == expected, f'Selected adapter changed: {name}')
    summary = read_json(run / 'results/summary.json')
    results = {}
    all_predictions = {}
    for name, prediction_file, metric_file, summary_key in [
        ('bge', 'results/bge.predictions.jsonl', 'results/bge.metrics.json', 'bge_only'),
        ('zero', 'original/validation.zero-shot.predictions.jsonl', 'original/validation.zero-shot.metrics.json', 'zero_shot'),
        ('step50', 'original/validation.step-000050.predictions.jsonl', 'original/validation.step-000050.metrics.json', None),
        ('selected', 'resumed/selected_validation_predictions.jsonl', 'results/selected.metrics.json', 'selected_model'),
    ]:
        predictions = rows(run / prediction_file, 'query_id')
        observed, means = check_metrics(predictions, labels, candidates, read_json(run / metric_file))
        if summary_key:
            for k, v in means.items():
                close(v, summary[summary_key][k], f'{name} summary {k}')
        if name != 'bge':
            for q, p in predictions.items():
                check_logits(p, candidates[q]['option_skill_ids'])
        else:
            require(all(p['ranked_skill_ids'] == candidates[q]['retrieval_order']
                        for q, p in predictions.items()), 'BGE baseline is not original retrieval order')
        results[name] = observed
        all_predictions[name] = predictions
        if name == 'selected':
            for k, v in means.items():
                close(v, selection['selected']['metrics'][k], f'Selection metric {k}')
    selection_keys = [(sum(v['recall@5'] for v in result.values()) / 256,
                       sum(v['mrr@15'] for v in result.values()) / 256)
                      for name, result in results.items() if name != 'bge']
    require(tuple(selection['selected']['key']) == max(selection_keys), 'Selection rule not respected')
    for before, key in [('bge', 'paired_query_bootstrap_vs_bge'), ('zero', 'paired_query_bootstrap_vs_zero')]:
        delta, interval = bootstrap(results[before], results['selected'])
        close(delta, summary[key]['mean_difference'], 'Bootstrap mean')
        for a, b in zip(interval, summary[key]['bootstrap_95_ci']):
            close(a, b, 'Bootstrap interval')
    reloaded = rows(run / 'reload/predictions.jsonl', 'query_id')
    require(set(reloaded) == set(labels), 'Reload query mismatch')
    maximum = 0.0
    agreement = 0
    for q, p in reloaded.items():
        check_logits(p, candidates[q]['option_skill_ids'])
        require(p['candidate_skill_ids'] == candidates[q]['retrieval_order'], 'Reload candidate mismatch')
        original = all_predictions['selected'][q]
        maximum = max(maximum, max(abs(p['option_logits'][k] - original['option_logits'][k])
                                   for k in p['option_logits']))
        agreement += p['ranked_skill_ids'] == original['ranked_skill_ids']
    reload = read_json(run / 'reload/reload_check.json')
    close(maximum, reload['max_abs_logit_difference'], 'Reload logits')
    close(agreement / 256, reload['full_ranking_agreement'], 'Reload rankings')
    require(maximum == 0 and agreement == 256 and reload['passed_tolerance'], 'Reload parity failed')
    require(read_json(run / 'original/failure.json')['step'] == 67, 'Original failure not preserved')
    training = read_json(run / 'resumed/training_summary.json')
    require(training['steps'] == 100 and training['train_query_exposures'] == 1600, 'Wrong completed schedule')
    return {'hashes': len(paths), 'queries': 256, 'ranking_conditions': 4,
            'reload_logit_difference': maximum, 'inference_rerun': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
