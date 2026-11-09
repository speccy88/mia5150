"""Separate the pilot's trained-weight effect from changes to fusion."""
from common import STUDY, TRAIN_CACHE, np, read, rows, save, sha
from replay_development_reranker import fusion
from verify import metrics_reference


def paired(start, end):
    ids = sorted(start['per_query'])
    assert ids == sorted(end['per_query'])
    d = np.array([end['per_query'][q]['NDCG@10']-start['per_query'][q]['NDCG@10'] for q in ids])
    rng = np.random.default_rng(42)
    boot = np.concatenate([d[rng.integers(len(ids), size=(100,len(ids)))].mean(axis=1) for _ in range(100)])
    return {
        'delta_ndcg10': float(d.mean()),
        'descriptive_paired_ci95': np.percentile(boot,[2.5,97.5]).tolist(),
        'improved': int((d>1e-12).sum()), 'degraded': int((d< -1e-12).sum()),
        'unchanged': int((np.abs(d)<=1e-12).sum()),
    }


def main():
    root = STUDY/'results/reranker'
    policy = read(root/'policy_selection.json')['selected']
    released_summary = STUDY/policy['summary']
    released_path = released_summary.with_name(released_summary.name.replace('.summary.json','.jsonl'))
    trained_path = root/'bce-pilot/selection-00100.jsonl'
    trained_summary = trained_path.with_suffix('.summary.json')
    released, trained = rows(released_path), rows(trained_path)
    assert len(released)==len(trained)==512
    gold = {row['query_id']:set() for row in trained}
    assert len(gold)==512
    for label in rows(TRAIN_CACHE/'train-qrels.jsonl'):
        if label['query_id'] in gold:gold[label['query_id']].add(label['skill_id'])
    for a,b in zip(released,trained,strict=True):
        assert a['status']==b['status']=='ok' and a['query_id']==b['query_id']
        assert a['candidates']==b['candidates'] and a['dense_scores']==b['dense_scores']
        k = b['scored_candidate_count']
        assert a['selections'][:k]==b['selections']
    control, candidate = read(released_summary), read(trained_summary)
    for setting, evidence in candidate.items():
        prefix, weight = setting.split('-')
        k,w = int(prefix[1:]),float(weight[1:])
        ranks = {}
        for row in trained:
            order = fusion(row['dense_scores'][:k],row['reranker_scores'][:k],w)
            ranks[row['query_id']] = [row['candidates'][i] for i in order]+row['candidates'][k:]
        assert metrics_reference(gold,ranks)==evidence['evaluation'], setting
    best = read(root/'bce-pilot/best.json')
    matched = best['configuration']
    released_choice = policy['configuration']
    save(root/'pilot_matched_comparison.json', {
        'status':'passed','queries':512,'all_trained_policy_metrics_independently_verified':True,
        'all_candidates_and_matched_inputs_identical':True,
        'trained_score_records_sha256':sha(trained_path),'released_score_records_sha256':sha(released_path),
        'trained_selected_configuration':matched,'released_selected_configuration':released_choice,
        'released_selected_ndcg10':policy['ndcg10'],'trained_selected_ndcg10':best['ndcg10'],
        'training_effect_at_trained_policy':paired(control[matched]['evaluation'],candidate[matched]['evaluation']),
        'training_effect_at_released_policy':paired(control[released_choice]['evaluation'],candidate[released_choice]['evaluation']),
        'scope':'Development selection; descriptive interval does not correct for choosing a policy on these queries.',
    })
    print('Pilot policies independently verified against exact matched input evidence.',flush=True)


if __name__ == '__main__':
    main()
