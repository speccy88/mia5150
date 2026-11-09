"""Explain reconstructed reference differences using saved vectors only."""
from common import CACHE, OLD, STUDY, np, read, rows, save, sha, write_rows


def main():
    assert read(STUDY / 'FINAL_EVAL_LOCK.json')['locked']
    reference = read(OLD / 'results/baseline_skillret_06b.json')
    cache = CACHE / 'vectors'

    def checked_array(name):
        folder = cache / name
        done = read(folder / 'complete.json')
        path = folder / 'vectors.npy'
        assert sha(path) == done['array_sha256']
        return np.load(path, mmap_mode='r', allow_pickle=False), read(folder / 'contract.json')['ids']

    documents, document_ids = checked_array('test-documents')
    queries, query_ids = checked_array('test-base-queries')
    index = {sid: i for i, sid in enumerate(document_ids)}
    reconstructed = {r['query_id']: r for r in rows(STUDY / 'results/final/reconstructed-base.jsonl')}
    assert len(queries) == len(reconstructed) == len(reference['rankings']) == 4392
    assert set(query_ids) == set(reconstructed) == set(reference['rankings'])
    differences = []
    error_by_query = []
    for i, qid in enumerate(query_ids):
        old = reference['rankings'][qid]
        new = reconstructed[qid]['ranking']
        assert len(old) == len(new) == 50
        scores = documents @ queries[i]
        old_score = dict(zip(old, reference['scores'][qid]))
        rebuilt_score = dict(zip(new, reconstructed[qid]['scores']))
        assert all(abs(float(scores[index[sid]]) - score) < 1e-6 for sid, score in rebuilt_score.items())
        aligned_error = max(abs(float(scores[index[sid]]) - old_score[sid]) for sid in old)
        error_by_query.append(aligned_error)
        if old == new:
            continue
        new_pos = {sid: rank for rank, sid in enumerate(new)}
        swaps = [(a, b) for j, a in enumerate(old) if a in new_pos for b in old[j+1:] if b in new_pos and new_pos[a] > new_pos[b]]
        lost = [sid for sid in old if sid not in new_pos]
        gained = [sid for sid in new if sid not in old_score]
        old_margin = max((old_score[a] - old_score[b] for a, b in swaps), default=0.)
        new_margin = max((float(scores[index[b]]) - float(scores[index[a]]) for a, b in swaps), default=0.)
        differences.append({
            'query_id': qid, 'top10_identical': old[:10] == new[:10],
            'top50_overlap': 50 - len(lost), 'changed_common_pairs': len(swaps),
            'max_old_score_margin_for_reordered_common_pair': old_margin,
            'max_rebuilt_score_margin_for_reordered_common_pair': new_margin,
            'lost_ids': lost, 'gained_ids': gained, 'max_aligned_score_error': aligned_error,
            'max_archived_score_margin_of_lost_id_above_cutoff': max((old_score[sid] - old_score[old[-1]] for sid in lost), default=0.),
            'max_rebuilt_score_margin_of_gained_id_above_lost_id': max((float(scores[index[b]]) - float(scores[index[a]]) for a in lost for b in gained), default=0.),
        })
    old = reference['evaluation']['aggregate']
    new = read(STUDY / 'results/final/reconstructed-base.metrics.json')['aggregate']
    report = {
        'scope': 'All archived query rankings, score errors aligned on the same skill IDs, no model inference',
        'query_count': len(query_ids), 'identical_top50_count': len(query_ids) - len(differences),
        'identical_top10_count': len(query_ids) - sum(not row['top10_identical'] for row in differences),
        'changed_top50_membership_count': sum(row['top50_overlap'] != 50 for row in differences),
        'max_aligned_archived_score_error': max(error_by_query),
        'p95_query_max_aligned_score_error': float(np.percentile(error_by_query, 95)),
        'archive_ndcg10': old['NDCG@10'], 'reconstructed_ndcg10': new['NDCG@10'],
        'reconstructed_minus_archived_ndcg10': new['NDCG@10'] - old['NDCG@10'],
        'reconstructed_minus_archived_recall10': new['Recall@10'] - old['Recall@10'],
        'ranking_differences_require_review': bool(differences),
    }
    write_rows(STUDY / 'results/reconstruction_ranking_differences.jsonl', differences)
    save(STUDY / 'results/reconstruction_diagnostics.json', report)
    print(report)


if __name__ == '__main__':
    main()
