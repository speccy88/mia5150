"""Audit matched holdout controls before the official benchmark is opened."""
from common import STUDY, data, np, read, rows, save, sha
from freeze_pipeline import lookup


def counts(start, end, gold, ids):
    result = {}
    for positive_count in sorted({len(gold[q]) for q in ids}):
        population = [q for q in ids if len(gold[q]) == positive_count]
        changes = [end['per_query'][q]['NDCG@10'] - start['per_query'][q]['NDCG@10'] for q in population]
        result[str(positive_count)] = {
            'queries': len(changes),
            'mean_ndcg10_change': float(np.mean(changes)),
            'improved': sum(d > 1e-12 for d in changes),
            'degraded': sum(d < -1e-12 for d in changes),
            'unchanged': sum(abs(d) <= 1e-12 for d in changes),
        }
    return result


def main():
    results = STUDY / 'results'
    assert not (STUDY / 'FINAL_EVAL_LOCK.json').exists()
    confirmation = results / 'CONFIRMATION.json'
    receipt = read(confirmation)
    assert not receipt['used_for_retuning'] and receipt['all_settings_unchanged_after_confirmation']
    membership = read(STUDY / 'manifests/split_membership.json')
    ids = membership['confirmation_query_ids']
    assert len(ids) == len(set(ids)) == 512
    skills, queries, labels = data('train')
    del skills, queries
    assert all(membership['skill_partition'][sid] == 'confirmation' for q in ids for sid in labels[q])
    query = read(results / 'query/confirmation-all-preregistered-mixtures.json')
    query_lock = read(results / 'query/SELECTION_LOCK.json')
    weight = query_lock['mixture_weight']
    weight_key = str(int(weight)) if float(weight).is_integer() else str(weight)
    assert query['part'] == 'confirmation'
    assert all(set(m['per_query']) == set(ids) for m in query['mixtures'].values())
    base = query['mixtures']['0']
    cfg = read(results / 'reranker/SELECTION_LOCK.json')
    k, w = cfg['candidates'], cfg['fusion_weight']
    released_path = results / 'reranker/released-confirmation/panel.jsonl'
    trained_path = results / 'reranker/trained-confirmation/panel.jsonl'
    released = {r['query_id']: r for r in rows(released_path)}
    trained = {r['query_id']: r for r in rows(trained_path)}
    assert set(released) == set(trained) == set(ids)
    score_differences = []
    for q in ids:
        a, b = released[q], trained[q]
        assert a['status'] == b['status'] == 'ok'
        assert a['candidates'] == b['candidates'] == query['rankings']['0'][q]
        assert a['dense_scores'] == b['dense_scores']
        assert a['selections'][:k] == b['selections'][:k]
        score_differences.extend(abs(x-y) for x,y in zip(a['reranker_scores'][:k], b['reranker_scores'][:k], strict=True))
    ref = lookup(read(released_path.with_suffix('.summary.json')), k, w)
    adapted = lookup(read(trained_path.with_suffix('.summary.json')), k, w)
    assert ref['candidate_recall'] == adapted['candidate_recall']
    assert ref['ideal_candidate_ndcg10'] == adapted['ideal_candidate_ndcg10']
    extra = {}
    if read(results/'PIPELINE_SELECTION_LOCK.json')['primary_condition']=='trained-combined':
        combined_release = results/'reranker/released-combined-confirmation/panel.jsonl'
        combined_train = results/'reranker/trained-combined-confirmation/panel.jsonl'
        arows, brows = rows(combined_release), rows(combined_train)
        assert len(arows)==len(brows)==512
        assert [r['query_id'] for r in arows]==[r['query_id'] for r in brows]==ids
        for a,b in zip(arows,brows,strict=True):
            q=a['query_id']
            assert a['status']==b['status']=='ok'
            assert a['candidates']==b['candidates']==query['rankings'][weight_key][q]
            assert a['dense_scores']==b['dense_scores'] and a['selections'][:k]==b['selections'][:k]
        c0=lookup(read(combined_release.with_suffix('.summary.json')),k,w)
        c1=lookup(read(combined_train.with_suffix('.summary.json')),k,w)
        assert c0['candidate_recall']==c1['candidate_recall'] and c0['ideal_candidate_ndcg10']==c1['ideal_candidate_ndcg10']
        extra={'combined_released_and_trained_exact_inputs_equal':True,
               'combined_score_records_sha256':{'released':sha(combined_release),'trained':sha(combined_train)},
               'combined_reranker_effect_by_positive_count':counts(c0['evaluation'],c1['evaluation'],labels,ids),
               'combined_query_effect_by_positive_count':counts(adapted['evaluation'],c1['evaluation'],labels,ids)}
    effects = receipt['local_training_effects']
    output = {
        'status': 'passed',
        'confirmation_receipt_sha256': sha(confirmation),
        'query_count': len(ids),
        'no_official_test_inference': True,
        'reranker_candidates_and_exact_inputs_equal': True,
        'matched_score_records_sha256': {'released': sha(released_path), 'trained': sha(trained_path)},
        'candidate_pool': {'k': k, 'recall': ref['candidate_recall'], 'ideal_ndcg10': ref['ideal_candidate_ndcg10']},
        'pure_query_effect_by_positive_count': counts(base, query['mixtures']['1'], labels, ids),
        'mixture_effect_by_positive_count': counts(base, query['mixtures'][weight_key], labels, ids),
        'reranker_effect_by_positive_count': counts(ref['evaluation'], adapted['evaluation'], labels, ids),
        'reranker_score_change_median_absolute': float(np.median(score_differences)),
        'reranker_score_change_max_absolute': max(score_differences),
        'positive_own_start_effects': sorted(name for name, x in effects.items() if x['positive_interval']),
        'unchanged_research_evaluation_ready': True,
        'repair_decision': 'No automatic recipe change. With matched inputs and no method failure, retain the locked research comparison even if neither local effect is confirmed.',
        'scope': 'Descriptive error audit on the frozen confirmation panel; no parameter changes or test-derived features. Small positive-count strata are not precise.',
        **extra,
    }
    save(results / 'development_review.json', output)
    print('Verified matched development controls:', output['positive_own_start_effects'], flush=True)


if __name__ == '__main__':
    main()
