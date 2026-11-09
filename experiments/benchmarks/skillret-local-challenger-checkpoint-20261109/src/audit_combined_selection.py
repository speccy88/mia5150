"""Independently verify the completed combined development selection panel."""
from common import CACHE, STUDY, TRAIN_CACHE, mix_query_vectors, np, read, rows, save, sha
from replay_development_reranker import fusion
from audit_reranker_pilot import paired
from verify import metrics_reference


def main():
    root = STUDY/'results/reranker'
    source = root/'combined-query/selection.jsonl'
    scored = rows(source)
    policy = read(root/'SELECTION_LOCK.json')
    k,w = policy['candidates'],policy['fusion_weight']
    configuration = f'k{k}-w{w:g}'
    selected = read(root/'combined-query/selection.summary.json')[configuration]
    original = read(root/f'{policy["trial"]}/selection-{policy["best_step"]:05d}.summary.json')[configuration]
    released_policy = read(root/'policy_selection.json')['selected']
    released = read(STUDY/released_policy['summary'])[released_policy['configuration']]
    ids = read(STUDY/'manifests/split_membership.json')['selection_query_ids']
    assert [r['query_id'] for r in scored]==ids and len(ids)==len(set(ids))==512
    gold = {q:set() for q in ids}
    for label in rows(TRAIN_CACHE/'train-qrels.jsonl'):
        if label['query_id'] in gold:gold[label['query_id']].add(label['skill_id'])

    document_folder = CACHE/'vectors/train-documents'
    query_folder = CACHE/'vectors/train-queries'
    documents = np.load(document_folder/'vectors.npy', mmap_mode='r')
    document_ids = read(document_folder/'contract.json')['ids']
    source_order = np.arange(len(document_ids))
    base = np.load(query_folder/'vectors.npy', mmap_mode='r')
    base_rows = {qid:i for i,qid in enumerate(read(query_folder/'contract.json')['ids'])}
    adapted = np.load(CACHE/'query-selected/selection-adapted.npy')
    assert read(CACHE/'query-selected/selection-ids.json')==ids
    weight = read(STUDY/'results/query/SELECTION_LOCK.json')['mixture_weight']
    rankings = {}
    for i,row in enumerate(scored):
        qid = row['query_id']
        vector = mix_query_vectors(base[base_rows[qid]],adapted[i],weight)
        scores = documents@vector
        top = np.lexsort((source_order,-scores))[:50]
        assert [document_ids[j] for j in top]==row['candidates']
        assert [float(scores[j]) for j in top]==row['dense_scores']
        order = fusion(row['dense_scores'][:k],row['reranker_scores'][:k],w)
        rankings[qid] = [row['candidates'][j] for j in order]+row['candidates'][k:]
    assert metrics_reference(gold,rankings)==selected['evaluation']
    result = {
        'status':'passed','queries':len(ids),'independent_dense_and_metric_replay':True,
        'source_sha256':sha(source),'reranker_settings_match':configuration,
        'quality_scope':'Development selection only; descriptive intervals do not account for policy choice',
        'query_update_in_fixed_trained_reranker':paired(original['evaluation'],selected['evaluation']),
        'combined_vs_best_released':paired(released['evaluation'],selected['evaluation']),
        'frozen_candidate_recall':original['candidate_recall'],'combined_candidate_recall':selected['candidate_recall'],
        'frozen_ideal_candidate_ndcg10':original['ideal_candidate_ndcg10'],
        'combined_ideal_candidate_ndcg10':selected['ideal_candidate_ndcg10'],
    }
    target = root/'combined_selection_audit.json'
    if target.exists():assert read(target)==result
    else:save(target,result)
    print('Combined selection independently reproduced on all 512 queries.',flush=True)


if __name__ == '__main__':
    main()
